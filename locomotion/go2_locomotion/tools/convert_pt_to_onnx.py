#!/usr/bin/env python3
"""
RSL-RL ActorCritic .pt → ONNX 변환기 (MLP non-recurrent 전용).

python3 convert_pt_to_onnx.py \
        --checkpoint /home/unitree/ros2_ws/src/loconav-stack/locomotion/go2_locomotion/model/model_9300.pt \
        --obs-dim 45 \
        --action-dim 12 \
        --hidden-dims 512 256 128 \
        --output policy_first.onnx

Usage:
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
