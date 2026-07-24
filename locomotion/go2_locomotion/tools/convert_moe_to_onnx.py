"""Export an rsl_rl ActorCriticMoE / ActorCriticVel (.pt) to ONNX for the torch-free Go2 runtime.

Why a special exporter (vs convert_pt_to_onnx.py, which only does plain MLP / recurrent):
the MoE actor is two-stage and its ROUTING is stateful (hysteresis + min-dwell across control
steps), which cannot live inside a single stateless ONNX graph. So we export a STATELESS graph
that returns, for the current observation, the action of EVERY expert plus the gate logits:

    (obs_history[1,225], obs_policy[1,45])  ->  expert_actions[1,K,12], gate_logits[1,K]

The robot-side wrapper (:class:`OnnxMoEPolicy`) then does argmax + persistence in numpy and picks
``expert_actions[chosen]`` -- so the exact deployment routing (incl. anti-chatter persistence and
the live expert index) is preserved with no torch on the robot.

A Hist-MLP checkpoint (ActorCriticVel, no ``actor.moe.*``) is exported as the simpler
``obs_history[1,225] -> action[1,12]``.

Run on a machine WITH torch + rsl_rl:
    python3 tools/convert_moe_to_onnx.py --checkpoint model_8700.pt --out policy_moe.onnx \
        [--rsl-rl-path /home/chung/workspace/rsl_rl]
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 45
HISTORY_LENGTH = 5
NUM_ACTIONS = 12


def _outs(sd, prefix):
    ws = sorted((k for k in sd if k.startswith(prefix) and k.endswith(".weight")),
                key=lambda k: int(k[len(prefix):].split(".")[0]))
    return [sd[k].shape[0] for k in ws]


def _infer_moe_arch(sd):
    n_exp = 1 + max((int(k.split(".")[3]) for k in sd if k.startswith("actor.moe.experts.")), default=0)
    gate_hidden = _outs(sd, "actor.moe.gating.")[:-1]
    expert_outs = _outs(sd, "actor.moe.experts.0.")
    trunk_outs = _outs(sd, "actor.shared_trunk.")
    assert any(k.startswith("actor.action_head.") for k in sd), "only encoder-stage MoE is supported for ONNX"
    arch = dict(actor_hidden_dims=gate_hidden, critic_hidden_dims=gate_hidden, activation="elu",
                expert_num=n_exp, moe_stage="encoder", latent_dim=expert_outs[-1],
                encoder_obs_groups=["obs_history"])
    if trunk_outs:
        arch.update(shared_trunk_dim=trunk_outs[-1], encoder_hidden_dims=trunk_outs[:-1],
                    expert_hidden_dims=expert_outs[:-1])
    else:
        arch.update(shared_trunk_dim=0, encoder_hidden_dims=expert_outs[:-1], expert_hidden_dims=expert_outs[:-1])
    v = _outs(sd, "vel_head.")
    if v:
        arch.update(vel_head_hidden_dim=v[0], vel_dim=v[-1])
    return arch


class MoEExport(nn.Module):
    """Stateless MoE forward: obs -> (per-expert action, gate logits). Routing is done off-graph."""

    def __init__(self, actor):
        super().__init__()
        self.shared_trunk = actor.shared_trunk        # may be None
        self.gating = actor.moe.gating
        self.experts = actor.moe.experts
        self.action_head = actor.action_head
        self.K = len(self.experts)

    def forward(self, obs_history, obs_policy):
        h = self.shared_trunk(obs_history) if self.shared_trunk is not None else obs_history
        logits = self.gating(h)                                       # (B, K)
        acts = []
        for k in range(self.K):
            latent = F.normalize(self.experts[k](h), p=2.0, dim=-1)   # (B, latent)
            acts.append(self.action_head(torch.cat([latent, obs_policy], dim=-1)))  # (B, 12)
        return torch.stack(acts, dim=1), logits                      # (B, K, 12), (B, K)


class MlpExport(nn.Module):
    """Stateless Hist-MLP forward: obs_history -> action."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs_history):
        return self.actor(obs_history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--rsl-rl-path", default="/home/chung/workspace/rsl_rl")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    if args.rsl_rl_path and args.rsl_rl_path not in sys.path:
        sys.path.insert(0, args.rsl_rl_path)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    obs = {"obs_history": torch.zeros(1, OBS_DIM * HISTORY_LENGTH),
           "policy": torch.zeros(1, OBS_DIM), "critic": torch.zeros(1, OBS_DIM)}
    is_moe = any(k.startswith("actor.moe.") for k in sd)

    if is_moe:
        from rsl_rl.modules.actor_critic_moe import ActorCriticMoE
        arch = _infer_moe_arch(sd)
        groups = {"policy": ["policy"], "critic": ["critic", "obs_history"]}
        model = ActorCriticMoE(obs, groups, NUM_ACTIONS, actor_obs_normalization=False,
                               critic_obs_normalization=False, **arch)
        _load(model, sd)
        model.eval()
        wrap = MoEExport(model.actor).eval()
        hist, pol = torch.zeros(1, OBS_DIM * HISTORY_LENGTH), torch.zeros(1, OBS_DIM)
        torch.onnx.export(wrap, (hist, pol), args.out, opset_version=args.opset, export_params=True,
                          input_names=["obs_history", "obs_policy"],
                          output_names=["expert_actions", "gate_logits"], dynamo=False)
        print(f"[export] MoE (K={arch['expert_num']}, trunk={arch['shared_trunk_dim']}) -> {args.out}")
        _verify_moe(model, wrap, args.out, arch["expert_num"])
    else:
        from rsl_rl.modules.actor_critic import ActorCriticVel
        a_outs = _outs(sd, "actor.")
        varch = dict(actor_hidden_dims=a_outs[:-1], critic_hidden_dims=a_outs[:-1], activation="elu")
        v = _outs(sd, "vel_head.")
        if v:
            varch.update(vel_head_hidden_dim=v[0], vel_dim=v[-1])
        obs_mlp = {"obs_history": torch.zeros(1, OBS_DIM * HISTORY_LENGTH), "critic": torch.zeros(1, OBS_DIM)}
        model = ActorCriticVel(obs_mlp, {"policy": ["obs_history"], "critic": ["critic", "obs_history"]},
                               NUM_ACTIONS, actor_obs_normalization=False, critic_obs_normalization=False, **varch)
        _load(model, sd)
        model.eval()
        wrap = MlpExport(model.actor).eval()
        hist = torch.zeros(1, OBS_DIM * HISTORY_LENGTH)
        torch.onnx.export(wrap, (hist,), args.out, opset_version=args.opset, export_params=True,
                          input_names=["obs_history"], output_names=["action"], dynamo=False)
        print(f"[export] Hist-MLP -> {args.out}")
        _verify_mlp(wrap, args.out)


def _load(model, sd):
    own = model.state_dict()
    filt = {k: v for k, v in sd.items() if k in own and hasattr(v, "shape") and own[k].shape == v.shape}
    model.load_state_dict(filt, strict=False)
    missing = [k for k in own if (k.startswith("actor.") or k.startswith("vel_head."))
               and not k.endswith("_extra_state") and k not in filt]
    assert not missing, f"actor tensors did not load: {missing[:5]}"


def _verify_moe(model, wrap, path, K):
    import numpy as np, onnxruntime as ort
    hist, pol = torch.randn(1, OBS_DIM * HISTORY_LENGTH), torch.randn(1, OBS_DIM)
    with torch.no_grad():
        t_act, t_log = wrap(hist, pol)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o_act, o_log = sess.run(None, {"obs_history": hist.numpy(), "obs_policy": pol.numpy()})
    da, dl = np.abs(t_act.numpy() - o_act).max(), np.abs(t_log.numpy() - o_log).max()
    print(f"[verify] MoE onnx-vs-torch max|d action|={da:.2e}  max|d logits|={dl:.2e}")
    assert da < 1e-4 and dl < 1e-4, "ONNX numerics diverged from torch"
    print("[verify] OK")


def _verify_mlp(wrap, path):
    import numpy as np, onnxruntime as ort
    hist = torch.randn(1, OBS_DIM * HISTORY_LENGTH)
    with torch.no_grad():
        t = wrap(hist)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"obs_history": hist.numpy()})[0]
    d = np.abs(t.numpy() - o).max()
    print(f"[verify] MLP onnx-vs-torch max|d action|={d:.2e}")
    assert d < 1e-4
    print("[verify] OK")


if __name__ == "__main__":
    main()
