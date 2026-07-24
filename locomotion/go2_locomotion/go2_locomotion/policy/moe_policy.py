"""Discrete-MoE (ActorCriticMoE) policy plug-in for the MuJoCo joystick demo.

Unlike :class:`OnnxPolicy`, this loads the rsl_rl PyTorch ``ActorCriticMoE`` directly so it can expose the
**currently selected expert (routing mode)** every control step -- read it from ``self.current_expert``.

Why not ONNX: the MoE actor is two-stage --

    encoder_obs (5-frame history, 225-dim)  --MLP-->  latent (32) --.
                                                                     |--> MoE gate argmax picks 1 of 8 experts
    head_obs    (current frame, 45-dim)  --------------------------'      -> that expert MLP emits the 12 actions

We want the argmax expert index out, and we need to feed the encoder the history stack. Both are awkward through the
baseline single-input ONNX path, so we run the torch module and read ``actor.moe.last_mode``.

Observation contract (must match training exactly)
--------------------------------------------------
The controller hands us the SAME 45-dim single-frame obs the ONNX MLP uses, in the trained term order:

    [0:3]   base_ang_vel * 0.2      [3:6]   projected_gravity     [6:9]   velocity_commands
    [9:21]  joint_pos_rel           [21:33] joint_vel_rel * 0.05  [33:45] last_action

The training MoE env adds an ``obs_history`` group = the SAME policy terms, each stacked over the last 5 frames.
IsaacLab stacks history PER TERM then concatenates the terms (NOT 5 whole frames back to back), with each term's
frames ordered oldest -> newest (``CircularBuffer.buffer``). So we keep a 5-deep ring of the 45-dim obs and
re-assemble it per term:

    obs_history = concat_over_terms( concat_over_frames_oldest_to_newest( frame[term_slice] ) )   # -> 225-dim

Because the term boundaries are identical to the single-frame obs (the history group subclasses the policy group),
we derive the slices straight from the controller's validated layout -- no separate ordering to get wrong.

Normalization is OFF in the MoE run (``actor_obs_normalization=False``, ``empirical_normalization=False``), so the
raw obs is fed as-is (the model's normalizers are Identity).

Dependencies: this needs ``torch`` and the ``rsl_rl`` package importable (point ``rsl_rl_path`` at the repo). The
ONNX demo deliberately avoids torch; run the MoE demo in an env that has both torch and mujoco/pygame.
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np

from .base_policy import BasePolicy

# Single-frame term boundaries (must match NNPolicyController._build_observation and the trained PolicyCfg order).
TERM_SLICES = [(0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45)]
OBS_DIM = 45
HISTORY_LENGTH = 5
NUM_ACTIONS = 12

# Architecture -- MUST match MoEPPORunnerCfg.policy in unitree_rl_lab (rsl_rl_ppo_cfg.py).
_MOE_ARCH = dict(
    actor_hidden_dims=[512, 256, 128],   # gating; also the shared action head when moe_stage="encoder"
    critic_hidden_dims=[512, 256, 128],  # unused at inference (loaded loosely); kept for shape parity
    activation="elu",
    expert_num=8,
    encoder_hidden_dims=[512, 256, 256],
    expert_hidden_dims=[512, 256, 256],
    encoder_obs_groups=["obs_history"],
    vel_head_hidden_dim=128,
    vel_dim=4,
    # moe_stage and latent_dim are INFERRED from the checkpoint (see _infer_arch) so that both
    # the older action-stage runs and the newer encoder-stage runs load with no caller change.
)


def _infer_arch(sd: dict) -> dict:
    """Read the FULL MoE actor architecture off a checkpoint's state_dict, so old and new runs load
    with no caller change.

    Read from weight shapes (not hardcoded): moe_stage, expert count, latent width, every hidden-dim
    list, the optional shared trunk (capacity layout "C": a shared encoder trunk + thin per-expert
    heads), and the aux velocity head. Encoder-stage runs have ``actor.action_head.*`` (experts emit
    the latent); action-stage runs have ``actor.student_encoder.*`` (experts emit the action).
    """
    def outs(prefix: str) -> list[int]:
        # out_features of each Linear under prefix, in layer order (hidden dims = all but the last)
        ws = sorted((k for k in sd if k.startswith(prefix) and k.endswith(".weight")),
                    key=lambda k: int(k[len(prefix):].split(".")[0]))
        return [sd[k].shape[0] for k in ws]

    n_exp = 1 + max((int(k.split(".")[3]) for k in sd if k.startswith("actor.moe.experts.")), default=0)
    gate_hidden = outs("actor.moe.gating.")[:-1]
    expert_outs = outs("actor.moe.experts.0.")
    trunk_outs = outs("actor.shared_trunk.")  # [] when there is no shared trunk

    if any(k.startswith("actor.action_head.") for k in sd):
        stage, latent_dim = "encoder", expert_outs[-1]
    elif any(k.startswith("actor.student_encoder.") for k in sd):
        stage, latent_dim = "action", outs("actor.student_encoder.")[-1]
    else:
        raise RuntimeError("[MoEPolicy] cannot infer moe_stage: checkpoint has neither "
                           "actor.action_head.* nor actor.student_encoder.*")

    arch = dict(
        actor_hidden_dims=gate_hidden,       # gate (and the shared action head in encoder stage)
        critic_hidden_dims=gate_hidden,      # unused at inference; kept for shape parity
        activation="elu",
        expert_num=n_exp,
        moe_stage=stage,
        latent_dim=latent_dim,
        encoder_obs_groups=["obs_history"],
    )
    if trunk_outs:  # capacity layout C: shared trunk feeds the gate + thin expert heads
        arch["shared_trunk_dim"] = trunk_outs[-1]
        arch["encoder_hidden_dims"] = trunk_outs[:-1]   # sizes the shared trunk
        arch["expert_hidden_dims"] = expert_outs[:-1]   # sizes the thin per-expert heads
    else:           # independent experts (the expert MLP hidden IS the encoder hidden)
        arch["shared_trunk_dim"] = 0
        arch["encoder_hidden_dims"] = expert_outs[:-1]
        arch["expert_hidden_dims"] = expert_outs[:-1]
    vel_outs = outs("vel_head.")  # always present on ActorCriticMoE; built + asserted at load
    if vel_outs:
        arch["vel_head_hidden_dim"] = vel_outs[0]
        arch["vel_dim"] = vel_outs[-1]
    return arch


class MoEPolicy(BasePolicy):
    """rsl_rl ActorCriticMoE loaded for MuJoCo inference; exposes the live routing expert.

    Attributes:
        current_expert (int): expert index chosen on the LAST ``__call__`` (argmax routing).
        expert_counts (np.ndarray): cumulative per-expert selection counts since the last ``reset``.
    """

    def __init__(
        self,
        model_path: str,
        rsl_rl_path: str = "/home/chung/workspace/rsl_rl",
        device: str = "cpu",
        deterministic: bool = True,
        route_hysteresis: float = 0.0,
        route_min_dwell: int = 0,
        force_expert: int | None = None,
        **kwargs,
    ):
        import torch  # local import so the ONNX demo path never requires torch

        self._torch = torch
        self._device = torch.device(device)
        # deterministic (default): argmax routing -- the same obs always picks the same expert, and the
        # action is the policy mean (no Gaussian noise). False: sample the expert from the gate softmax
        # each step (routing exploration; action is still the mean). Deployment normally uses deterministic.
        self._deterministic = deterministic

        if rsl_rl_path and rsl_rl_path not in sys.path:
            sys.path.insert(0, rsl_rl_path)
        # import the module directly to avoid the modules/__init__ import chain
        from rsl_rl.modules.actor_critic_moe import ActorCriticMoE

        # A sample obs dict just for dim inference in the constructor. The critic group dim is
        # arbitrary: we load weights by matching NAME+SHAPE, so a wrong critic dim simply means the
        # (unused) critic weights are skipped -- the actor still loads exactly.
        obs_sample = {
            "policy": torch.zeros(1, OBS_DIM),
            "obs_history": torch.zeros(1, OBS_DIM * HISTORY_LENGTH),
            "critic": torch.zeros(1, OBS_DIM),  # placeholder; real privileged dim not needed for inference
        }
        obs_groups = {"policy": ["policy"], "critic": ["critic", "obs_history"]}

        # read the checkpoint once up front so the architecture can be matched to it
        ckpt = torch.load(model_path, map_location=self._device)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        arch = _infer_arch(sd)
        print(f"[MoEPolicy] inferred arch: stage={arch['moe_stage']!r} K={arch['expert_num']} "
              f"latent={arch['latent_dim']} trunk={arch['shared_trunk_dim']} "
              f"encoder_hidden={arch['encoder_hidden_dims']} expert_hidden={arch['expert_hidden_dims']}")

        self._model = ActorCriticMoE(
            obs_sample,
            obs_groups,
            NUM_ACTIONS,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            **arch,
        ).to(self._device)
        self._expert_num = arch["expert_num"]
        self.moe_stage = arch["moe_stage"]

        self._load_weights(sd, model_path)
        self._model.eval()
        # Force past the warmup/partition windows so routing is the deterministic argmax over all
        # experts (update_count defaults to 0 -> warmup_active -> everything would route to expert 0).
        self._model.actor.moe.update_count = 10**9
        # Deployment-side routing persistence (works on any checkpoint, incl. ones trained without
        # it): hold the mode unless the challenger's logit beats the held one by > hysteresis and
        # the mode was held >= min_dwell steps. Suppresses phase-timescale mode chatter.
        self._model.actor.moe.route_hysteresis = float(route_hysteresis)
        self._model.actor.moe.route_min_dwell = int(route_min_dwell)

        # force EVERY step to a fixed expert (bypasses the gate via mode_override) -- lets you drive
        # one expert on the MuJoCo course and see its gait in isolation, exactly like eval.py's #K.
        if force_expert is not None and not (0 <= force_expert < self._expert_num):
            raise ValueError(f"force_expert must be in [0, {self._expert_num}), got {force_expert}")
        self._force_expert = force_expert
        if force_expert is not None:
            print(f"[MoEPolicy] FORCING every step to expert {force_expert} (gate bypassed)")

        self._hist: deque | None = None
        self.current_expert: int = -1
        self.expert_counts = np.zeros(self._expert_num, dtype=np.int64)

    # ------------------------------------------------------------------
    def _load_weights(self, sd: dict, model_path: str) -> None:
        own = self._model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own and hasattr(v, "shape") and own[k].shape == v.shape}
        self._model.load_state_dict(filtered, strict=False)

        # sanity: every actor (encoder + moe + vel_head) WEIGHT must have loaded, or the policy is garbage.
        # Exclude non-tensor extra-state keys (e.g. actor.moe._extra_state = update_count/revival_events,
        # a dict handled separately -- we override update_count below regardless).
        actor_keys = [
            k for k in own
            if (k.startswith("actor.") or k.startswith("vel_head.")) and not k.endswith("_extra_state")
        ]
        not_loaded = [k for k in actor_keys if k not in filtered]
        print(f"[MoEPolicy] loaded {len(filtered)}/{len(own)} tensors from {model_path}")
        if not_loaded:
            raise RuntimeError(
                f"[MoEPolicy] {len(not_loaded)} ACTOR tensors did not load (name/shape mismatch) -- the "
                f"architecture here does not match the checkpoint. First few: {not_loaded[:5]}"
            )

    # ------------------------------------------------------------------
    def _build_history(self, obs: np.ndarray) -> np.ndarray:
        """Re-assemble the 225-dim per-term history stack from the 45-dim frame ring (oldest->newest)."""
        if self._hist is None:
            # first frame: IsaacLab's first push fills every slot with it
            self._hist = deque([obs.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
        else:
            self._hist.append(obs.copy())
        frames = list(self._hist)  # oldest -> newest
        blocks = [frame[a:b] for (a, b) in TERM_SLICES for frame in frames]  # term-major, frame oldest->newest
        return np.concatenate(blocks).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        assert obs.shape[0] == OBS_DIM, f"MoEPolicy expects a {OBS_DIM}-dim obs, got {obs.shape[0]}"

        hist = self._build_history(obs)
        obs_td = {
            "policy": torch.from_numpy(obs).unsqueeze(0).to(self._device),        # (1, 45)
            "obs_history": torch.from_numpy(hist).unsqueeze(0).to(self._device),  # (1, 225)
        }
        with torch.no_grad():
            # normalizers are Identity (normalization off in the MoE run); this equals act_inference's
            # front half, but keeps the routing branch explicit.
            enc = self._model.encoder_obs_normalizer(self._model.get_encoder_obs(obs_td))   # (1, 225)
            head = self._model.actor_obs_normalizer(self._model.get_actor_obs(obs_td))      # (1, 45)
            if self._force_expert is not None:
                # pin routing to one expert (mode_override bypasses gate/persistence entirely)
                mode = torch.full((enc.shape[0],), self._force_expert, dtype=torch.long, device=self._device)
                action = self._model.actor(enc, head, mode_override=mode)
            else:
                action = self._model.actor(enc, head)  # argmax routing; also caches moe.last_logits
            if self._force_expert is None and not self._deterministic:
                # sample the expert from the gate softmax instead of taking the argmax (routing exploration).
                # action is still the expert's mean output -- only WHICH expert is stochastic.
                logits = self._model.actor.moe.last_logits
                mode = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
                action = self._model.actor(enc, head, mode_override=mode)

        self.current_expert = int(self._model.actor.moe.last_mode.item())
        self.expert_counts[self.current_expert] += 1
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        self._hist = None
        self.current_expert = -1
        self.expert_counts[:] = 0
        # clear routing-persistence state (held mode + dwell) along with the obs history
        self._model.reset()
