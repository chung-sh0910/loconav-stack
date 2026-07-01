# Recurrent (GRU/LSTM) policy ONNX export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tools/convert_pt_to_onnx.py` able to convert a recurrent (`rsl_rl`
`ActorCriticRecurrent`/`ActorCriticRecurrentVel`) `.pt` checkpoint to ONNX, auto-detecting
GRU vs. LSTM and all RNN dimensions from the checkpoint's tensor shapes, with zero new required
CLI flags.

**Architecture:** Add a `RecurrentActor` `nn.Module` (RNN + reused `build_actor()` MLP head) and
an architecture-inference helper that reads `num_layers`/`hidden_size`/`rnn_type` straight from
`memory_a.rnn.*` tensor shapes in the checkpoint. `convert()` detects recurrent checkpoints by the
presence of `memory_a.rnn.` keys and dispatches to a recurrent export path that produces
`obs,h_in[,c_in]` → `actions,h_out[,c_out]` ONNX I/O, matching what
`go2_locomotion/policy/onnx_policy.py` already expects. Non-recurrent checkpoints keep using the
existing MLP path untouched.

**Tech Stack:** Python 3.8, PyTorch 2.4 (CPU), onnxruntime 1.16, pytest.

## Global Constraints

- No new required CLI flags — `rnn_type`/`hidden_size`/`num_layers` are always inferred from the
  checkpoint's tensor shapes (per spec, user-approved: no override flags).
- ONNX input/output names for recurrent export must exactly match
  `go2_locomotion/policy/onnx_policy.py`'s expectations: GRU → inputs `obs,h_in`, outputs
  `actions,h_out`; LSTM → inputs `obs,h_in,c_in`, outputs `actions,h_out,c_out`.
- `opset_version=18`, `dynamic_axes={}` (unchanged from existing script).
- Non-recurrent checkpoints must continue to convert exactly as before — no regression.
- Real checkpoint used for validation: `/home/csh/go2_logs3/unitree_go2_velocity/2026-06-30_13-04-56/model_7400.pt`
  (confirmed: single-layer GRU, `hidden_size=256`, `input_size=45`, actor head hidden dims
  `[512, 256, 128]`, `action_dim=12`).

---

## File Structure

- Modify: `tools/convert_pt_to_onnx.py` — add recurrent detection, architecture inference,
  `RecurrentActor` module, and recurrent export path. Existing `build_actor()` /
  `load_actor_weights()` / non-recurrent `convert()` logic stays and is reused.
- Create: `test/test_convert_pt_to_onnx.py` — unit tests for the new inference/build logic using
  small synthetic checkpoints (no dependency on the real trained file, so tests run anywhere).

## Task 1: Architecture inference from checkpoint tensor shapes

**Files:**
- Modify: `tools/convert_pt_to_onnx.py`
- Test: `test/test_convert_pt_to_onnx.py`

**Interfaces:**
- Produces: `infer_rnn_arch(state_dict: dict) -> dict` with keys `rnn_type` (`"gru"` or
  `"lstm"`), `hidden_size` (int), `num_layers` (int), `input_size` (int). Raises `ValueError` if
  `memory_a.rnn.weight_ih_l0` is missing, or if the ih/hh row-count ratio isn't 3 or 4.

- [ ] **Step 1: Write the failing tests**

Create `test/test_convert_pt_to_onnx.py`:

```python
import sys
import os
import importlib.util

import torch
import pytest

_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
_MODULE_PATH = os.path.join(_TOOLS_DIR, "convert_pt_to_onnx.py")
_spec = importlib.util.spec_from_file_location("convert_pt_to_onnx", _MODULE_PATH)
convert_pt_to_onnx = importlib.util.module_from_spec(_spec)
sys.modules["convert_pt_to_onnx"] = convert_pt_to_onnx
_spec.loader.exec_module(convert_pt_to_onnx)


def _gru_state_dict(input_size=45, hidden_size=256, num_layers=1):
    sd = {}
    in_dim = input_size
    for layer in range(num_layers):
        sd[f"memory_a.rnn.weight_ih_l{layer}"] = torch.zeros(3 * hidden_size, in_dim)
        sd[f"memory_a.rnn.weight_hh_l{layer}"] = torch.zeros(3 * hidden_size, hidden_size)
        sd[f"memory_a.rnn.bias_ih_l{layer}"] = torch.zeros(3 * hidden_size)
        sd[f"memory_a.rnn.bias_hh_l{layer}"] = torch.zeros(3 * hidden_size)
        in_dim = hidden_size
    return sd


def _lstm_state_dict(input_size=45, hidden_size=128, num_layers=2):
    sd = {}
    in_dim = input_size
    for layer in range(num_layers):
        sd[f"memory_a.rnn.weight_ih_l{layer}"] = torch.zeros(4 * hidden_size, in_dim)
        sd[f"memory_a.rnn.weight_hh_l{layer}"] = torch.zeros(4 * hidden_size, hidden_size)
        sd[f"memory_a.rnn.bias_ih_l{layer}"] = torch.zeros(4 * hidden_size)
        sd[f"memory_a.rnn.bias_hh_l{layer}"] = torch.zeros(4 * hidden_size)
        in_dim = hidden_size
    return sd


def test_infer_gru_single_layer():
    sd = _gru_state_dict(input_size=45, hidden_size=256, num_layers=1)
    arch = convert_pt_to_onnx.infer_rnn_arch(sd)
    assert arch == {"rnn_type": "gru", "hidden_size": 256, "num_layers": 1, "input_size": 45}


def test_infer_lstm_two_layers():
    sd = _lstm_state_dict(input_size=45, hidden_size=128, num_layers=2)
    arch = convert_pt_to_onnx.infer_rnn_arch(sd)
    assert arch == {"rnn_type": "lstm", "hidden_size": 128, "num_layers": 2, "input_size": 45}


def test_infer_rnn_arch_missing_key_raises():
    with pytest.raises(ValueError):
        convert_pt_to_onnx.infer_rnn_arch({"actor.0.weight": torch.zeros(1, 1)})


def test_infer_rnn_arch_bad_gate_ratio_raises():
    sd = {
        "memory_a.rnn.weight_ih_l0": torch.zeros(500, 45),  # 500/256 is not 3 or 4
        "memory_a.rnn.weight_hh_l0": torch.zeros(500, 256),
    }
    with pytest.raises(ValueError):
        convert_pt_to_onnx.infer_rnn_arch(sd)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: FAIL with `AttributeError: module 'convert_pt_to_onnx' has no attribute 'infer_rnn_arch'`

- [ ] **Step 3: Implement `infer_rnn_arch`**

In `tools/convert_pt_to_onnx.py`, add after `load_actor_weights`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add tools/convert_pt_to_onnx.py test/test_convert_pt_to_onnx.py
git commit -m "Add RNN architecture inference for recurrent checkpoint export"
```

## Task 2: `RecurrentActor` module + weight loading

**Files:**
- Modify: `tools/convert_pt_to_onnx.py`
- Test: `test/test_convert_pt_to_onnx.py`

**Interfaces:**
- Consumes: `infer_rnn_arch` (Task 1), `build_actor(obs_dim, action_dim, hidden_dims) ->
  nn.Sequential` (existing).
- Produces: `RecurrentActor(nn.Module)` with constructor
  `RecurrentActor(rnn_type: str, input_size: int, hidden_size: int, num_layers: int,
  action_dim: int, actor_hidden_dims)`, holding `self.rnn` and `self.actor`, and methods
  `forward_gru(self, x, h_in)` and `forward_lstm(self, x, h_in, c_in)`.
- Produces: `load_recurrent_weights(model: RecurrentActor, state_dict: dict) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_convert_pt_to_onnx.py`:

```python
def test_recurrent_actor_gru_forward_shapes():
    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="gru", input_size=45, hidden_size=256, num_layers=1,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    model.eval()
    obs = torch.zeros(1, 45)
    h_in = torch.zeros(1, 1, 256)
    actions, h_out = model.forward_gru(obs, h_in)
    assert actions.shape == (1, 12)
    assert h_out.shape == (1, 1, 256)


def test_recurrent_actor_lstm_forward_shapes():
    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="lstm", input_size=45, hidden_size=128, num_layers=2,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    model.eval()
    obs = torch.zeros(1, 45)
    h_in = torch.zeros(2, 1, 128)
    c_in = torch.zeros(2, 1, 128)
    actions, h_out, c_out = model.forward_lstm(obs, h_in, c_in)
    assert actions.shape == (1, 12)
    assert h_out.shape == (2, 1, 128)
    assert c_out.shape == (2, 1, 128)


def test_load_recurrent_weights_matches_manual_rnn_step():
    torch.manual_seed(0)
    sd = _gru_state_dict(input_size=45, hidden_size=256, num_layers=1)
    # give actor head real weights too (matches build_actor([512,256,128]) for hidden_size=256 in, 12 out)
    sd["actor.0.weight"] = torch.randn(512, 256)
    sd["actor.0.bias"] = torch.randn(512)
    sd["actor.2.weight"] = torch.randn(256, 512)
    sd["actor.2.bias"] = torch.randn(256)
    sd["actor.4.weight"] = torch.randn(128, 256)
    sd["actor.4.bias"] = torch.randn(128)
    sd["actor.6.weight"] = torch.randn(12, 128)
    sd["actor.6.bias"] = torch.randn(12)
    for k in list(sd.keys()):
        if k.startswith("memory_a.rnn."):
            sd[k] = torch.randn_like(sd[k])

    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="gru", input_size=45, hidden_size=256, num_layers=1,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    convert_pt_to_onnx.load_recurrent_weights(model, sd)
    model.eval()

    obs = torch.randn(1, 45)
    h_in = torch.zeros(1, 1, 256)
    actions, h_out = model.forward_gru(obs, h_in)

    # reference: run torch.nn.GRU directly with the same weights
    ref_rnn = torch.nn.GRU(input_size=45, hidden_size=256, num_layers=1)
    ref_rnn.load_state_dict(
        {k[len("memory_a.rnn."):]: v for k, v in sd.items() if k.startswith("memory_a.rnn.")}
    )
    ref_rnn.eval()
    ref_out, ref_h = ref_rnn(obs.unsqueeze(0), h_in)
    assert torch.allclose(h_out, ref_h, atol=1e-6)
    assert actions.shape == (1, 12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: FAIL — `AttributeError: module 'convert_pt_to_onnx' has no attribute 'RecurrentActor'`

- [ ] **Step 3: Implement `RecurrentActor` and `load_recurrent_weights`**

In `tools/convert_pt_to_onnx.py`, add after `infer_rnn_arch`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add tools/convert_pt_to_onnx.py test/test_convert_pt_to_onnx.py
git commit -m "Add RecurrentActor module and recurrent weight loading"
```

## Task 3: Wire recurrent export into `convert()` + update docstring

**Files:**
- Modify: `tools/convert_pt_to_onnx.py`
- Test: `test/test_convert_pt_to_onnx.py`

**Interfaces:**
- Consumes: `infer_rnn_arch`, `RecurrentActor`, `load_recurrent_weights` (Tasks 1-2).
- Modifies: `convert(checkpoint_path, obs_dim, action_dim, hidden_dims, output_path)` — same
  signature, now branches on recurrence internally. No caller-facing changes.

- [ ] **Step 1: Write the failing test**

Append to `test/test_convert_pt_to_onnx.py`:

```python
def test_convert_recurrent_checkpoint_exports_onnx_with_expected_io(tmp_path):
    import onnxruntime as ort
    import numpy as np

    torch.manual_seed(1)
    sd = _gru_state_dict(input_size=45, hidden_size=256, num_layers=1)
    sd["actor.0.weight"] = torch.randn(512, 256)
    sd["actor.0.bias"] = torch.randn(512)
    sd["actor.2.weight"] = torch.randn(256, 512)
    sd["actor.2.bias"] = torch.randn(256)
    sd["actor.4.weight"] = torch.randn(128, 256)
    sd["actor.4.bias"] = torch.randn(128)
    sd["actor.6.weight"] = torch.randn(12, 128)
    sd["actor.6.bias"] = torch.randn(12)
    for k in list(sd.keys()):
        if k.startswith("memory_a.rnn."):
            sd[k] = torch.randn_like(sd[k])
    # non-actor keys that must be ignored
    sd["critic.0.weight"] = torch.randn(512, 256)
    sd["vel_head.0.weight"] = torch.randn(256, 256)

    ckpt_path = tmp_path / "model_recurrent.pt"
    torch.save({"model_state_dict": sd}, ckpt_path)
    onnx_path = tmp_path / "policy.onnx"

    convert_pt_to_onnx.convert(
        str(ckpt_path), obs_dim=45, action_dim=12, hidden_dims=[512, 256, 128],
        output_path=str(onnx_path),
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    output_names = {o.name for o in sess.get_outputs()}
    assert input_names == {"obs", "h_in"}
    assert output_names == {"actions", "h_out"}

    out = sess.run(None, {
        "obs": np.zeros((1, 45), dtype=np.float32),
        "h_in": np.zeros((1, 1, 256), dtype=np.float32),
    })
    actions = out[0]
    assert actions.shape == (1, 12)


def test_convert_non_recurrent_checkpoint_still_works(tmp_path):
    import onnxruntime as ort
    import numpy as np

    torch.manual_seed(2)
    sd = {
        "actor.0.weight": torch.randn(512, 45), "actor.0.bias": torch.randn(512),
        "actor.2.weight": torch.randn(256, 512), "actor.2.bias": torch.randn(256),
        "actor.4.weight": torch.randn(128, 256), "actor.4.bias": torch.randn(128),
        "actor.6.weight": torch.randn(12, 128), "actor.6.bias": torch.randn(12),
    }
    ckpt_path = tmp_path / "model_mlp.pt"
    torch.save({"model_state_dict": sd}, ckpt_path)
    onnx_path = tmp_path / "policy_mlp.onnx"

    convert_pt_to_onnx.convert(
        str(ckpt_path), obs_dim=45, action_dim=12, hidden_dims=[512, 256, 128],
        output_path=str(onnx_path),
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    assert input_names == {"obs"}
    out = sess.run(None, {"obs": np.zeros((1, 45), dtype=np.float32)})
    assert out[0].shape == (1, 12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: `test_convert_recurrent_checkpoint_exports_onnx_with_expected_io` FAILs (raises inside
`load_actor_weights` because `actor.0.weight` shape `(512, 256)` doesn't match an actor built with
`obs_dim=45` as input — this is exactly the current MLP-only bug this task fixes). The
non-recurrent test should already PASS since that path is untouched.

- [ ] **Step 3: Update `convert()` to branch on recurrence**

Replace the body of `convert()` in `tools/convert_pt_to_onnx.py`:

```python
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
```

Also update the module docstring at the top of the file:

```python
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
```

(This replaces the old docstring, including removing the stale `policy_first.onnx` usage example
that was pointing at the old MLP-only checkpoint.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_convert_pt_to_onnx.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/ -v`
Expected: all tests pass (25 previously-passing tests + 9 new ones = 34 passed; `test_onnx_policy.py`
tests will still be skipped/failed only if their hardcoded `MODEL_PATH` file doesn't exist locally —
pre-existing condition, unrelated to this change).

- [ ] **Step 6: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add tools/convert_pt_to_onnx.py test/test_convert_pt_to_onnx.py
git commit -m "Support recurrent (GRU/LSTM) checkpoints in convert_pt_to_onnx.py"
```

## Task 4: Validate against the real trained checkpoint

**Files:**
- None modified — this is a manual validation task using the script from Task 3.

**Interfaces:**
- Consumes: `convert()` CLI from `tools/convert_pt_to_onnx.py` (Task 3).

- [ ] **Step 1: Convert the real checkpoint**

Run:
```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
python3 tools/convert_pt_to_onnx.py \
  --checkpoint /home/csh/go2_logs3/unitree_go2_velocity/2026-06-30_13-04-56/model_7400.pt \
  --obs-dim 45 \
  --action-dim 12 \
  --hidden-dims 512 256 128 \
  --output model/policy_lstm_7400.onnx
```

Expected: prints `저장 완료 (recurrent, gru, hidden=256, layers=1): model/policy_lstm_7400.onnx`
followed by `추론 확인 — actions shape: (1, 12), 샘플: [...]` with finite (non-NaN) values.

- [ ] **Step 2: Load the exported ONNX file with `OnnxPolicy` and confirm GRU auto-detection**

Run:
```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
python3 - <<'EOF'
import numpy as np
from go2_locomotion.policy.onnx_policy import OnnxPolicy

policy = OnnxPolicy("model/policy_lstm_7400.onnx")
assert policy._is_gru is True
assert policy._is_lstm is False

obs = np.zeros(45, dtype=np.float32)
for _ in range(5):
    actions = policy(obs)
    assert actions.shape == (12,)
    assert np.all(np.isfinite(actions))

policy.reset()
actions = policy(obs)
assert actions.shape == (12,)
print("OK: GRU policy loads, steps, and resets correctly")
EOF
```

Expected: prints `OK: GRU policy loads, steps, and resets correctly` with no assertion errors.

- [ ] **Step 3: Update `config/params.yaml` to point at the new model (if the user confirms this is the model to deploy)**

Ask the user whether `nn_policy.model_path` in `config/params.yaml` should be updated to
`model/policy_lstm_7400.onnx` now, or left for them to wire up later. Do not change it without
confirmation, since swapping the active deployed policy is a real-robot-affecting change.

- [ ] **Step 4: No commit in this task**

This task only validates; no repo files are modified unless the user confirms Step 3, in which
case commit `config/params.yaml` alone with message `Point nn_policy at the new recurrent policy
export`.
