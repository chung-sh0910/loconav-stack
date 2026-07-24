"""History-MLP (ActorCriticVel) policy plug-in for the MuJoCo demo -- the matched baseline to
:class:`MoEPolicy`.

Loads the rsl_rl ``ActorCriticVel`` .pt directly and drives it with the SAME observation contract
as the MoE policy: a 5-frame, 45-dim history re-assembled per term into the 225-dim ``obs_history``
group (see :mod:`moe_policy` for the exact stacking). The only difference is the network -- a plain
MLP over the 225-dim history (no routing, no experts) -- so running this and the MoE policy on the
same MuJoCo scene is an apples-to-apples sim2sim comparison.

``current_expert`` is always -1 (no experts) so the shared demo harness can treat both policies
uniformly. Needs ``torch`` + ``rsl_rl`` importable, like :class:`MoEPolicy`.
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np

from .base_policy import BasePolicy
from .moe_policy import HISTORY_LENGTH, NUM_ACTIONS, OBS_DIM, TERM_SLICES


def _infer_mlp_arch(sd: dict) -> dict:
    """Read the ActorCriticVel actor hidden dims (and the aux velocity head) off a checkpoint, so
    the network matches whatever the run used without a hardcoded shape."""
    def outs(prefix: str) -> list[int]:
        ws = sorted((k for k in sd if k.startswith(prefix) and k.endswith(".weight")),
                    key=lambda k: int(k[len(prefix):].split(".")[0]))
        return [sd[k].shape[0] for k in ws]

    actor_outs = outs("actor.")  # e.g. [512, 256, 128, 12]  (hidden = all but the last)
    if not actor_outs:
        raise RuntimeError("[MlpPolicy] checkpoint has no actor.* MLP weights -- is this an MLP run?")
    arch = dict(actor_hidden_dims=actor_outs[:-1], critic_hidden_dims=actor_outs[:-1], activation="elu")
    vel_outs = outs("vel_head.")  # ActorCriticVel always builds vel_head -> must match at load
    if vel_outs:
        arch["vel_head_hidden_dim"] = vel_outs[0]
        arch["vel_dim"] = vel_outs[-1]
    return arch


class MlpPolicy(BasePolicy):
    """rsl_rl ``ActorCriticVel`` (history-MLP) loaded for MuJoCo inference."""

    def __init__(
        self,
        model_path: str,
        rsl_rl_path: str = "/home/chung/workspace/rsl_rl",
        device: str = "cpu",
        **kwargs,
    ):
        import torch  # local import so the ONNX demo path never requires torch

        self._torch = torch
        self._device = torch.device(device)
        if rsl_rl_path and rsl_rl_path not in sys.path:
            sys.path.insert(0, rsl_rl_path)
        from rsl_rl.modules.actor_critic import ActorCriticVel

        # the MLP actor consumes the "policy" group, which for the Hist-MLP arm IS obs_history (225-d)
        obs_sample = {
            "obs_history": torch.zeros(1, OBS_DIM * HISTORY_LENGTH),
            "critic": torch.zeros(1, OBS_DIM),  # placeholder; the critic is unused at inference
        }
        obs_groups = {"policy": ["obs_history"], "critic": ["critic", "obs_history"]}

        ckpt = torch.load(model_path, map_location=self._device)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        arch = _infer_mlp_arch(sd)
        print(f"[MlpPolicy] inferred actor_hidden={arch['actor_hidden_dims']} "
              f"vel_head_hidden={arch.get('vel_head_hidden_dim')} vel_dim={arch.get('vel_dim')}")

        self._model = ActorCriticVel(
            obs_sample,
            obs_groups,
            NUM_ACTIONS,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            **arch,
        ).to(self._device)
        self._load_weights(sd, model_path)
        self._model.eval()

        self._hist: deque | None = None
        self.current_expert: int = -1        # no routing; kept so the demo harness is uniform
        self.expert_counts = np.zeros(1, dtype=np.int64)

    # ------------------------------------------------------------------
    def _load_weights(self, sd: dict, model_path: str) -> None:
        own = self._model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own and hasattr(v, "shape") and own[k].shape == v.shape}
        self._model.load_state_dict(filtered, strict=False)
        # every actor + vel_head WEIGHT must have loaded (critic/terrain_head are unused at inference)
        actor_keys = [
            k for k in own
            if (k.startswith("actor.") or k.startswith("vel_head.")) and not k.endswith("_extra_state")
        ]
        not_loaded = [k for k in actor_keys if k not in filtered]
        print(f"[MlpPolicy] loaded {len(filtered)}/{len(own)} tensors from {model_path}")
        if not_loaded:
            raise RuntimeError(
                f"[MlpPolicy] {len(not_loaded)} ACTOR tensors did not load (name/shape mismatch) -- the "
                f"architecture here does not match the checkpoint. First few: {not_loaded[:5]}"
            )

    # ------------------------------------------------------------------
    def _build_history(self, obs: np.ndarray) -> np.ndarray:
        """Re-assemble the 225-dim per-term history stack from the 45-dim frame ring (oldest->newest).
        Identical to MoEPolicy._build_history so the two arms see the exact same encoder input."""
        if self._hist is None:
            self._hist = deque([obs.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
        else:
            self._hist.append(obs.copy())
        frames = list(self._hist)  # oldest -> newest
        blocks = [frame[a:b] for (a, b) in TERM_SLICES for frame in frames]  # term-major
        return np.concatenate(blocks).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        assert obs.shape[0] == OBS_DIM, f"MlpPolicy expects a {OBS_DIM}-dim obs, got {obs.shape[0]}"

        hist = self._build_history(obs)
        obs_td = {"obs_history": torch.from_numpy(hist).unsqueeze(0).to(self._device)}  # (1, 225)
        with torch.no_grad():
            # normalizers are Identity (normalization off); actor consumes the 225-dim history directly
            actor_obs = self._model.actor_obs_normalizer(self._model.get_actor_obs(obs_td))
            action = self._model.actor(actor_obs)  # deterministic mean action (1, 12)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        self._hist = None
        self.current_expert = -1
