#!/usr/bin/env python3
"""
Torch.compile vs ONNX Runtime 추론 속도 비교 벤치마크 — 실제 로봇(온보드 컴퓨터)에서 실행.

recurrent(.pt) 체크포인트를 세 가지 방식으로 벤치마크한다:
  1. PyTorch eager       (컴파일 없음, 기준선)
  2. PyTorch torch.compile
  3. ONNX Runtime         (변환된 .onnx, go2_locomotion/policy/onnx_policy.py와 동일한
                            SessionOptions로 실행 — 실제 배포 기준선)

각 스텝은 실제 컨트롤 루프와 동일하게 hidden state를 이어서 넘기며 obs batch=1로 실행한다.
torch.compile 컴파일 트리거는 warmup 구간에서 소모하고 타이밍에서 제외한다.

Usage:
    python3 tools/benchmark_inference.py \
        --checkpoint /path/to/model_7400.pt \
        --onnx /path/to/policy.onnx \
        --obs-dim 45 \
        --num-steps 2000 \
        --warmup-steps 50 \
        --threads 1

50Hz 컨트롤 루프 예산은 스텝당 20ms. mean/median/p95/p99 모두 그 안에 들어오는지 확인할 것.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_pt_to_onnx import infer_rnn_arch, RecurrentActor, load_recurrent_weights  # noqa: E402


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    raise ValueError(f"checkpoint에서 state dict를 찾을 수 없음. 키: {list(ckpt.keys())}")


def _stats(times_ms) -> dict:
    arr = np.array(times_ms)
    return {
        "mean": arr.mean(),
        "median": np.median(arr),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "min": arr.min(),
        "max": arr.max(),
    }


def _print_stats(label: str, times_ms) -> None:
    s = _stats(times_ms)
    print(
        f"{label:24s} mean={s['mean']:7.3f}ms  median={s['median']:7.3f}ms  "
        f"p95={s['p95']:7.3f}ms  p99={s['p99']:7.3f}ms  min={s['min']:7.3f}ms  max={s['max']:7.3f}ms"
    )


def benchmark_torch(model, arch, device, num_steps: int, warmup_steps: int, compiled: bool):
    hidden = torch.zeros(arch["num_layers"], 1, arch["hidden_size"], device=device)
    cell = (
        torch.zeros(arch["num_layers"], 1, arch["hidden_size"], device=device)
        if arch["rnn_type"] == "lstm" else None
    )
    obs = torch.zeros(1, arch["input_size"], device=device)

    def step(obs, h, c):
        if arch["rnn_type"] == "lstm":
            actions, h, c = model.forward_lstm(obs, h, c)
        else:
            actions, h = model.forward_gru(obs, h)
        return actions, h, c

    if compiled:
        step = torch.compile(step)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _, hidden, cell = step(obs, hidden, cell)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            _, hidden, cell = step(obs, hidden, cell)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms


def benchmark_onnx(onnx_path: str, arch, num_steps: int, warmup_steps: int):
    import onnxruntime as ort

    # go2_locomotion/policy/onnx_policy.py와 동일한 설정 (실제 배포 기준선)
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    obs = np.zeros((1, arch["input_size"]), dtype=np.float32)
    h_in = np.zeros((arch["num_layers"], 1, arch["hidden_size"]), dtype=np.float32)
    c_in = (
        np.zeros((arch["num_layers"], 1, arch["hidden_size"]), dtype=np.float32)
        if arch["rnn_type"] == "lstm" else None
    )

    def step(h_in, c_in):
        feed = {"obs": obs, "h_in": h_in}
        if arch["rnn_type"] == "lstm":
            feed["c_in"] = c_in
            out = sess.run(None, feed)
            return out[0], out[1], out[2]
        out = sess.run(None, feed)
        return out[0], out[1], None

    for _ in range(warmup_steps):
        _, h_in, c_in = step(h_in, c_in)

    times_ms = []
    for _ in range(num_steps):
        t0 = time.perf_counter()
        _, h_in, c_in = step(h_in, c_in)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="rsl_rl recurrent .pt 체크포인트")
    parser.add_argument("--onnx", required=True, help="convert_pt_to_onnx.py로 변환한 .onnx (ONNX 기준선용)")
    parser.add_argument("--obs-dim", type=int, default=45)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1, help="torch 스레드 수 (ONNX는 항상 1로 고정)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                         help="torch eager/compile을 실행할 device. ONNX 기준선은 항상 CPU(실제 배포와 동일).")
    parser.add_argument("--skip-compile", action="store_true",
                         help="torch.compile 단계 건너뛰기 (컴파일 툴체인 없는 기기에서 eager/ONNX만 비교)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 지정했지만 이 기기에서 CUDA를 사용할 수 없음")

    print(f"torch.set_num_threads({args.threads}), device={device}, CUDA available: {torch.cuda.is_available()}")

    raw_sd = _load_checkpoint_state_dict(args.checkpoint)
    arch = infer_rnn_arch(raw_sd)
    print(f"체크포인트 구조: {arch}")

    model = RecurrentActor(
        rnn_type=arch["rnn_type"], input_size=arch["input_size"],
        hidden_size=arch["hidden_size"], num_layers=arch["num_layers"],
        action_dim=args.action_dim, actor_hidden_dims=args.hidden_dims,
    )
    load_recurrent_weights(model, raw_sd)
    model.eval()
    model.to(device)

    print(f"\n--- {args.num_steps} steps (warmup {args.warmup_steps}) — 50Hz 컨트롤 루프 예산: 20ms/step ---\n")

    eager_times = benchmark_torch(model, arch, device, args.num_steps, args.warmup_steps, compiled=False)
    _print_stats("PyTorch eager", eager_times)

    if args.skip_compile:
        print(f"{'PyTorch torch.compile':24s} 건너뜀 (--skip-compile)")
    else:
        try:
            compiled_times = benchmark_torch(model, arch, device, args.num_steps, args.warmup_steps, compiled=True)
            _print_stats("PyTorch torch.compile", compiled_times)
        except Exception as e:
            print(f"{'PyTorch torch.compile':24s} 실패: {e!r}")
            print("  (이 기기에 컴파일 툴체인이 없을 수 있음 — 실패해도 eager/ONNX 결과는 유효함)")

    onnx_times = benchmark_onnx(args.onnx, arch, args.num_steps, args.warmup_steps)
    _print_stats("ONNX Runtime (CPU)", onnx_times)


if __name__ == "__main__":
    main()
