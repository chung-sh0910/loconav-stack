#!/usr/bin/env python3
"""
RSL-RL ActorCritic .pt → ONNX 변환기.

MLP(non-recurrent)와 recurrent(GRU/LSTM, ActorCriticRecurrent) 체크포인트를 모두 지원.
recurrent 체크포인트는 memory_a.rnn.* 키 존재 여부로 자동 감지되며, rnn_type/hidden_size/
num_layers는 체크포인트 텐서 shape에서 자동으로 추론됨 (별도 플래그 불필요).

recurrent 출력 ONNX I/O:
  GRU  : inputs  obs, h_in     → outputs actions, h_out
  LSTM : inputs  obs, h_in, c_in → outputs actions, h_out, c_out

Usage (MLP와 recurrent 모두 동일한 커맨드 형태):
    python3 tools/convert_pt_to_onnx.py \
        --checkpoint /path/to/model_9000.pt \
        --obs-dim 45 \
        --action-dim 12 \
        --hidden-dims 512 256 128 \
        --output /path/to/policy.onnx
"""

import argparse
import torch
import torch.nn as nn


def build_actor(obs_dim: int, action_dim: int, hidden_dims) -> nn.Sequential:
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, action_dim))
    return nn.Sequential(*layers)


def load_actor_weights(actor: nn.Sequential, state_dict: dict) -> None:
    """model_state_dict (old rsl-rl < 5.0) 또는 actor_state_dict (new >= 5.0) 자동 감지."""
    if any(k.startswith("actor.") for k in state_dict):
        # old format: actor.0.weight, actor.2.weight, ...
        actor_sd = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
    elif any(k.startswith("mlp.") for k in state_dict):
        # new format: mlp.0.weight, ...
        actor_sd = {k[len("mlp."):]: v for k, v in state_dict.items() if k.startswith("mlp.")}
    else:
        raise ValueError(f"알 수 없는 state_dict 형식. 키 목록: {list(state_dict.keys())[:10]}")
    actor.load_state_dict(actor_sd)


def infer_rnn_arch(state_dict: dict) -> dict:
    """memory_a.rnn.* 텐서 shape에서 rnn_type/hidden_size/num_layers/input_size를 추론."""
    if "memory_a.rnn.weight_ih_l0" not in state_dict:
        raise ValueError("state_dict에 memory_a.rnn.weight_ih_l0 키가 없음 (recurrent 체크포인트 아님)")

    num_layers = 0
    while f"memory_a.rnn.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1

    weight_ih_l0 = state_dict["memory_a.rnn.weight_ih_l0"]
    weight_hh_l0 = state_dict["memory_a.rnn.weight_hh_l0"]
    hidden_size = weight_hh_l0.shape[1]
    input_size = weight_ih_l0.shape[1]

    gate_rows = weight_ih_l0.shape[0]
    if gate_rows == 3 * hidden_size:
        rnn_type = "gru"
    elif gate_rows == 4 * hidden_size:
        rnn_type = "lstm"
    else:
        raise ValueError(
            f"알 수 없는 RNN 게이트 비율: weight_ih_l0.shape[0]={gate_rows}, hidden_size={hidden_size} "
            f"(3x=GRU, 4x=LSTM 이어야 함)"
        )

    return {
        "rnn_type": rnn_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_size": input_size,
    }


class RecurrentActor(nn.Module):
    """memory_a.rnn (GRU/LSTM) + actor MLP head. ONNX export용 단일 스텝 forward."""

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, num_layers: int,
                 action_dim: int, actor_hidden_dims) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.actor = build_actor(hidden_size, action_dim, actor_hidden_dims)

    def forward_gru(self, x, h_in):
        out, h_out = self.rnn(x.unsqueeze(0), h_in)
        return self.actor(out.squeeze(0)), h_out

    def forward_lstm(self, x, h_in, c_in):
        out, (h_out, c_out) = self.rnn(x.unsqueeze(0), (h_in, c_in))
        return self.actor(out.squeeze(0)), h_out, c_out


def load_recurrent_weights(model: "RecurrentActor", state_dict: dict) -> None:
    rnn_sd = {k[len("memory_a.rnn."):]: v for k, v in state_dict.items() if k.startswith("memory_a.rnn.")}
    model.rnn.load_state_dict(rnn_sd)
    load_actor_weights(model.actor, state_dict)


def _convert_recurrent(raw_sd: dict, obs_dim: int, action_dim: int,
                        hidden_dims, output_path: str) -> None:
    arch = infer_rnn_arch(raw_sd)
    if arch["input_size"] != obs_dim:
        print(
            f"경고: --obs-dim={obs_dim} 이지만 체크포인트의 memory_a.rnn 입력 크기는 "
            f"{arch['input_size']} 입니다. 체크포인트 값을 사용합니다."
        )
    model = RecurrentActor(
        rnn_type=arch["rnn_type"],
        input_size=arch["input_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        action_dim=action_dim,
        actor_hidden_dims=hidden_dims,
    )
    load_recurrent_weights(model, raw_sd)
    model.eval()

    dummy_obs = torch.zeros(1, arch["input_size"])
    h_in = torch.zeros(arch["num_layers"], 1, arch["hidden_size"])

    if arch["rnn_type"] == "lstm":
        c_in = torch.zeros(arch["num_layers"], 1, arch["hidden_size"])
        model.forward = model.forward_lstm
        torch.onnx.export(
            model, (dummy_obs, h_in, c_in), output_path,
            export_params=True, opset_version=18,
            input_names=["obs", "h_in", "c_in"],
            output_names=["actions", "h_out", "c_out"],
            dynamic_axes={}, verbose=False,
        )
    else:
        model.forward = model.forward_gru
        torch.onnx.export(
            model, (dummy_obs, h_in), output_path,
            export_params=True, opset_version=18,
            input_names=["obs", "h_in"],
            output_names=["actions", "h_out"],
            dynamic_axes={}, verbose=False,
        )
    print(
        f"저장 완료 (recurrent, {arch['rnn_type']}, hidden={arch['hidden_size']}, "
        f"layers={arch['num_layers']}): {output_path}"
    )

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    feed = {
        "obs": np.zeros((1, arch["input_size"]), dtype=np.float32),
        "h_in": np.zeros((arch["num_layers"], 1, arch["hidden_size"]), dtype=np.float32),
    }
    if arch["rnn_type"] == "lstm":
        feed["c_in"] = np.zeros((arch["num_layers"], 1, arch["hidden_size"]), dtype=np.float32)
    out = sess.run(None, feed)
    print(f"추론 확인 — actions shape: {out[0].shape}, 샘플: {out[0][0, :4]}")


def convert(checkpoint_path: str, obs_dim: int, action_dim: int,
            hidden_dims, output_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 구 포맷 / 신 포맷 자동 구분
    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif "actor_state_dict" in ckpt:
        raw_sd = ckpt["actor_state_dict"]
    else:
        raise ValueError(f"checkpoint에서 state dict를 찾을 수 없음. 키: {list(ckpt.keys())}")

    is_recurrent = any(k.startswith("memory_a.rnn.") for k in raw_sd)
    if is_recurrent:
        _convert_recurrent(raw_sd, obs_dim, action_dim, hidden_dims, output_path)
        return

    actor = build_actor(obs_dim, action_dim, hidden_dims)
    load_actor_weights(actor, raw_sd)
    actor.eval()

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        actor,
        dummy,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        verbose=False,
    )
    print(f"저장 완료: {output_path}")

    # 빠른 sanity check
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": np.zeros((1, obs_dim), dtype=np.float32)})
    print(f"추론 확인 — actions shape: {out[0].shape}, 샘플: {out[0][0, :4]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs-dim", type=int, default=45)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert(args.checkpoint, args.obs_dim, args.action_dim, args.hidden_dims, args.output)
