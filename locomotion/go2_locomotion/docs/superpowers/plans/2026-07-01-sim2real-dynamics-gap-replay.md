# Sim2Real Dynamics-Gap Replay Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log the real robot's rollout, replay the same actions in Isaac Lab, and compare joint/torque/base trajectories to quantify the sim-to-real dynamics gap.

**Architecture:** A shared CSV+JSON log schema (one row per 50 Hz step) written by both a real-robot logger (hooked into `NNPolicyController`) and an Isaac Lab open-loop replay script. A standalone comparison script loads both logs and produces overlay plots + an RMSE gap table. Replay feeds the recorded policy actions into `env.step()` (no policy in sim), isolating the dynamics gap.

**Tech Stack:** Python 3.8, numpy (logger, dependency-light), pandas + matplotlib (comparison script only), ROS2 Foxy (real side), Isaac Lab / rsl_rl (replay side).

## Global Constraints

- Measured joint columns (`target_q`, `q`, `dq`, `tau`) are in **SDK order**
  (`JOINT_NAMES`: FR,FL,RR,RL × hip,thigh,calf). `raw_action` columns are in **policy/Isaac
  order** (fed straight to `env.step()`, no remap). JSON records
  `raw_action_order: "policy"`, `measured_order: "sdk"`.
- `dt = 0.02` (50 Hz). `action_scale = 0.25`, `action_clip = 6.0`, RL gains Kp=25/Kd=0.5.
- `rollout_log.py` must stay dependency-light: stdlib `csv`/`json` + numpy only (it is
  imported by the ROS controller). pandas/matplotlib live only in the comparison script.
- Real `base_height`/`base_vx`/`base_vy`/`base_vyaw` columns are always `NaN` (not measured).
  Sim fills them from ground truth.
- Isaac Lab replay: `num_envs=1`, domain randomization + push events + observation noise
  disabled, action clip set to ±6 to match real.
- The Isaac Lab replay script cannot be executed in the dev environment (no GPU/Isaac). Only
  its pure-Python joint-order helper and schema round-trip are unit-tested here.

## File Structure

- Create: `go2_locomotion/go2_locomotion/utils/rollout_log.py` — schema constants,
  `quat_to_roll_pitch`, `make_row`, `write_log`, `read_log`, `RolloutLogger`.
- Modify: `go2_locomotion/go2_locomotion/controllers/nn_policy_controller.py` — optional
  logging in `start`/`step`/`stop`.
- Modify: `go2_locomotion/go2_locomotion/locomotion_node.py` + `config/params.yaml` —
  `nn_policy.log_path` param.
- Create: `go2_locomotion/tools/sim2real_compare.py` — comparison + plots + RMSE table.
- Create: `unitree_rl_lab/scripts/rsl_rl/joint_order.py` — pure-Python SDK↔Isaac index map.
- Create: `unitree_rl_lab/scripts/rsl_rl/replay_real_log.py` — Isaac Lab replay + sim logger.
- Test: `go2_locomotion/test/test_rollout_log.py`, additions to
  `go2_locomotion/test/test_nn_controller.py`, `go2_locomotion/test/test_sim2real_compare.py`,
  `unitree_rl_lab/scripts/rsl_rl/test_joint_order.py`.

---

## Task 1: Shared log schema module (`rollout_log.py`)

**Files:**
- Create: `go2_locomotion/go2_locomotion/utils/rollout_log.py`
- Test: `go2_locomotion/test/test_rollout_log.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `COLUMNS: list[str]` — full ordered column list.
  - `quat_to_roll_pitch(w, x, y, z) -> tuple[float, float]`
  - `make_row(step, t, cmd, raw_action, target_q, q, dq, tau, quat, gyro, base_height=nan, base_vx=nan, base_vy=nan, base_vyaw=nan) -> dict` — `cmd` is `(vx,vy,vyaw)`; `raw_action/target_q/q/dq/tau` are length-12 sequences; `quat` is `(w,x,y,z)`; `gyro` is `(x,y,z)`. Returns a dict keyed by `COLUMNS`.
  - `write_log(base_path, rows: list[dict], metadata: dict) -> None` — writes `base_path + ".csv"` and `base_path + ".json"`.
  - `read_log(base_path) -> tuple[dict, dict[str, np.ndarray]]` — returns `(metadata, columns)` where `columns[name]` is a float array.
  - `class RolloutLogger` with `__init__(self, base_path: str, metadata: dict)`, `append(self, row: dict)`, `flush(self)`.

- [ ] **Step 1: Write the failing tests**

Create `go2_locomotion/test/test_rollout_log.py`:

```python
import math
import numpy as np
from go2_locomotion.utils import rollout_log as rl


def test_columns_shape():
    # 5 scalars + 12*5 joint cols + 13 base cols
    assert rl.COLUMNS[:5] == ["step", "t", "cmd_vx", "cmd_vy", "cmd_vyaw"]
    assert "raw_action_11" in rl.COLUMNS
    assert "target_q_0" in rl.COLUMNS and "tau_11" in rl.COLUMNS
    assert rl.COLUMNS[-4:] == ["base_height", "base_vx", "base_vy", "base_vyaw"]
    assert len(rl.COLUMNS) == 5 + 12 * 5 + 13


def test_quat_to_roll_pitch_level():
    roll, pitch = rl.quat_to_roll_pitch(1.0, 0.0, 0.0, 0.0)
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6


def test_quat_to_roll_pitch_known_roll():
    # 90 deg roll about x: q = (cos45, sin45, 0, 0)
    c = math.cos(math.pi / 4)
    roll, pitch = rl.quat_to_roll_pitch(c, c, 0.0, 0.0)
    assert abs(roll - math.pi / 2) < 1e-6


def test_make_row_keys_and_values():
    row = rl.make_row(
        step=3, t=0.06, cmd=(0.5, 0.0, 0.1),
        raw_action=list(range(12)), target_q=[0.1] * 12, q=[0.2] * 12,
        dq=[0.3] * 12, tau=[0.4] * 12, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.01, 0.02, 0.03),
    )
    assert set(row.keys()) == set(rl.COLUMNS)
    assert row["step"] == 3 and row["cmd_vx"] == 0.5
    assert row["raw_action_5"] == 5 and row["q_0"] == 0.2 and row["tau_11"] == 0.4
    assert math.isnan(row["base_vx"]) and math.isnan(row["base_height"])


def test_write_read_roundtrip(tmp_path):
    rows = [
        rl.make_row(step=i, t=i * 0.02, cmd=(0.5, 0.0, 0.0),
                    raw_action=[i + 0.1 * j for j in range(12)],
                    target_q=[0.1] * 12, q=[0.2] * 12, dq=[0.3] * 12, tau=[0.4] * 12,
                    quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
        for i in range(4)
    ]
    meta = {"source": "real", "version": rl.SCHEMA_VERSION, "dt": 0.02,
            "joint_sdk_names": ["j%d" % i for i in range(12)]}
    base = str(tmp_path / "log")
    rl.write_log(base, rows, meta)

    meta2, cols = rl.read_log(base)
    assert meta2["source"] == "real" and meta2["version"] == rl.SCHEMA_VERSION
    assert cols["step"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(cols["raw_action_5"], [0.5, 1.5, 2.5, 3.5])


def test_rollout_logger_append_flush(tmp_path):
    base = str(tmp_path / "roll")
    logger = rl.RolloutLogger(base, {"source": "real", "version": rl.SCHEMA_VERSION})
    for i in range(3):
        logger.append(rl.make_row(step=i, t=i * 0.02, cmd=(0.0, 0.0, 0.0),
                                   raw_action=[0.0] * 12, target_q=[0.0] * 12, q=[0.0] * 12,
                                   dq=[0.0] * 12, tau=[0.0] * 12, quat=(1.0, 0.0, 0.0, 0.0),
                                   gyro=(0.0, 0.0, 0.0)))
    logger.set_meta("initial_q", [0.1] * 12)
    logger.flush()
    meta, cols = rl.read_log(base)
    assert len(cols["step"]) == 3
    assert meta["n_steps"] == 3
    assert meta["initial_q"] == [0.1] * 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_rollout_log.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError` (module and symbols don't exist yet).

- [ ] **Step 3: Implement `rollout_log.py`**

Create `go2_locomotion/go2_locomotion/utils/rollout_log.py`:

```python
"""Shared rollout log schema for sim2real comparison (dependency-light: csv/json + numpy)."""
import csv
import json
import math

import numpy as np

SCHEMA_VERSION = 1

_JOINT_GROUPS = ["raw_action", "target_q", "q", "dq", "tau"]
COLUMNS = (
    ["step", "t", "cmd_vx", "cmd_vy", "cmd_vyaw"]
    + [f"{g}_{i}" for g in _JOINT_GROUPS for i in range(12)]
    + ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
       "base_roll", "base_pitch",
       "base_gyro_x", "base_gyro_y", "base_gyro_z",
       "base_height", "base_vx", "base_vy", "base_vyaw"]
)


def quat_to_roll_pitch(w, x, y, z):
    """Roll (about x) and pitch (about y) from a (w,x,y,z) quaternion, radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


def make_row(step, t, cmd, raw_action, target_q, q, dq, tau, quat, gyro,
             base_height=float("nan"), base_vx=float("nan"),
             base_vy=float("nan"), base_vyaw=float("nan")):
    roll, pitch = quat_to_roll_pitch(quat[0], quat[1], quat[2], quat[3])
    row = {
        "step": step, "t": t,
        "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_vyaw": cmd[2],
        "base_quat_w": quat[0], "base_quat_x": quat[1],
        "base_quat_y": quat[2], "base_quat_z": quat[3],
        "base_roll": roll, "base_pitch": pitch,
        "base_gyro_x": gyro[0], "base_gyro_y": gyro[1], "base_gyro_z": gyro[2],
        "base_height": base_height, "base_vx": base_vx,
        "base_vy": base_vy, "base_vyaw": base_vyaw,
    }
    for name, arr in zip(_JOINT_GROUPS, [raw_action, target_q, q, dq, tau]):
        for i in range(12):
            row[f"{name}_{i}"] = float(arr[i])
    return row


def write_log(base_path, rows, metadata):
    with open(base_path + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    meta = dict(metadata)
    meta.setdefault("version", SCHEMA_VERSION)
    meta["n_steps"] = len(rows)
    with open(base_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def read_log(base_path):
    with open(base_path + ".json") as f:
        metadata = json.load(f)
    data = {name: [] for name in COLUMNS}
    with open(base_path + ".csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in COLUMNS:
                data[name].append(float(row[name]))
    cols = {name: np.array(vals, dtype=float) for name, vals in data.items()}
    return metadata, cols


class RolloutLogger:
    """In-memory buffer; O(1) append, single write on flush (keeps the 50 Hz loop clean)."""

    def __init__(self, base_path, metadata):
        self._base_path = base_path
        self._metadata = dict(metadata)
        self._rows = []

    def append(self, row):
        self._rows.append(row)

    def set_meta(self, key, value):
        self._metadata[key] = value

    def flush(self):
        write_log(self._base_path, self._rows, self._metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_rollout_log.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add go2_locomotion/go2_locomotion/utils/rollout_log.py go2_locomotion/test/test_rollout_log.py
git commit -m "Add shared rollout log schema for sim2real comparison"
```

---

## Task 2: Real-robot logging in `NNPolicyController`

**Files:**
- Modify: `go2_locomotion/go2_locomotion/controllers/nn_policy_controller.py`
- Modify: `go2_locomotion/go2_locomotion/locomotion_node.py`
- Modify: `go2_locomotion/config/params.yaml`
- Test: `go2_locomotion/test/test_nn_controller.py`

**Interfaces:**
- Consumes: `rollout_log.RolloutLogger`, `rollout_log.make_row` (Task 1); existing
  `JOINT_NAMES`, `JOINT_IDS_MAP`, `NUM_JOINTS`.
- Produces: `NNPolicyController.__init__(..., log_path: str = None)`; when set, a CSV+JSON
  log is written on `stop()`.

- [ ] **Step 1: Write the failing test**

Add to `go2_locomotion/test/test_nn_controller.py` (imports `os`, `rollout_log` at top as needed):

```python
class TestRolloutLogging:

    def test_logging_records_and_flushes(self, tmp_path):
        import os
        from go2_locomotion.utils import rollout_log as rl
        from go2_locomotion.utils.go2_constants import JOINT_IDS_MAP

        base = str(tmp_path / "real_log")
        raw = np.arange(NUM_JOINTS, dtype=np.float32)   # policy order 0..11
        policy = MagicMock(return_value=raw)
        ctrl = make_ctrl(policy=policy, action_scale=0.25)
        ctrl._log_path = base            # enable logging directly
        ctrl._start_logging()
        # deterministic lowstate: q=i, dq=10+i in SDK slots via helper params;
        # tau_est isn't set by make_fake_lowstate, so set it via subscript (stable per-slot mock)
        ls = make_fake_lowstate(
            joint_q=[float(i) for i in range(NUM_JOINTS)],
            joint_dq=[float(10 + i) for i in range(NUM_JOINTS)],
        )
        for i in range(NUM_JOINTS):
            ls.motor_state[i].tau_est = float(20 + i)
        ctrl._lowstate = ls

        with patch.object(ctrl, '_send_low_cmd'):
            ctrl.step()
        ctrl.stop()

        assert os.path.exists(base + ".csv") and os.path.exists(base + ".json")
        meta, cols = rl.read_log(base)
        assert meta["measured_order"] == "sdk"
        assert meta["raw_action_order"] == "policy"
        assert len(cols["step"]) == 1
        # raw_action logged in policy order (unremapped)
        np.testing.assert_allclose(cols["raw_action_5"], [5.0])
        # q logged in SDK order straight from motor_state slots
        np.testing.assert_allclose(cols["q_7"], [7.0])
        np.testing.assert_allclose(cols["tau_7"], [27.0])
        # initial state captured on first step
        assert meta["initial_q"][7] == 7.0
        assert meta["initial_base_quat"][0] == 1.0
```

Note: `make_fake_lowstate`'s `motor_state.__getitem__.side_effect` returns per-slot mocks;
the test sets `.q/.dq/.tau_est` on `ls.motor_state(i)` which returns the same mock the
controller reads via `motor_state[i]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_nn_controller.py::TestRolloutLogging -v`
Expected: FAIL — `AttributeError: 'NNPolicyController' object has no attribute '_start_logging'`.

- [ ] **Step 3: Implement logging in the controller**

In `nn_policy_controller.py`, add the import near the other util imports:

```python
from go2_locomotion.utils.rollout_log import RolloutLogger, make_row
```

Add `log_path` to `__init__` signature (after `policy_kd`) and store it:

```python
        policy_kp: list = None,
        policy_kd: list = None,
        log_path: str = None,
    ):
```
and in the body (after gains are set):
```python
        self._log_path = log_path if log_path else None
        self._logger = None
        self._log_step = 0
        self._log_t0 = None
```

Add a `_start_logging` helper (place after `_reset_policy`):

```python
    def _start_logging(self) -> None:
        if not self._log_path:
            return
        metadata = {
            "source": "real",
            "dt": 0.02,
            "raw_action_order": "policy",
            "measured_order": "sdk",
            "joint_sdk_names": list(JOINT_NAMES),
            "action_scale": self._action_scale,
            "action_clip": self._action_clip,
            "policy_kp": list(self._policy_kp),
            "policy_kd": list(self._policy_kd),
        }
        self._logger = RolloutLogger(self._log_path, metadata)
        self._log_step = 0
        self._log_t0 = time.time()
```

Import `JOINT_NAMES` in the existing `go2_constants` import block (add it to the list).

Call `self._start_logging()` at the end of `start()` (after `self._reset_policy()`).

In `step()`, after `self._send_low_cmd(...)`, append a log row:

```python
        if self._logger is not None:
            q   = np.array([lowstate.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
            dq  = np.array([lowstate.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
            tau = np.array([lowstate.motor_state[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float32)
            with self._cmd_lock:
                cmd = (self._vx, self._vy, self._vyaw)
            quat = tuple(lowstate.imu_state.quaternion[j] for j in range(4))
            gyro = tuple(lowstate.imu_state.gyroscope[j] for j in range(3))
            if self._log_step == 0:
                self._logger.set_meta("initial_q", q.tolist())
                self._logger.set_meta("initial_base_quat", list(quat))
            self._logger.append(make_row(
                step=self._log_step, t=time.time() - self._log_t0, cmd=cmd,
                raw_action=raw_action, target_q=target_q, q=q, dq=dq, tau=tau,
                quat=quat, gyro=gyro,
            ))
            self._log_step += 1
```

In `stop()`, flush before/after sending the default pose:

```python
    def stop(self) -> None:
        if self._logger is not None:
            self._logger.flush()
            self._logger = None
        if self._cmd_pub is not None:
            self._send_low_cmd(self._default_pos)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_nn_controller.py::TestRolloutLogging -v`
Expected: PASS.

- [ ] **Step 5: Wire the ROS param**

In `locomotion_node.py`, declare the param near the other nn_policy params:
```python
        self.declare_parameter('nn_policy.log_path', '')
```
read it in `_build_nn_controller`:
```python
        log_path     = self.get_parameter('nn_policy.log_path').value
```
and pass it:
```python
            policy_kp=list(policy_kp),
            policy_kd=list(policy_kd),
            log_path=log_path,
        )
```

In `config/params.yaml`, add under `nn_policy:` (after `policy_kd`):
```yaml
      # Sim2real rollout log output base path (no extension). Empty = logging off.
      # When set, writes <log_path>.csv/.json on stop() with the real rollout.
      log_path: ""
```

- [ ] **Step 6: Run the full controller test file**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_nn_controller.py -v`
Expected: all pass (existing + new `TestRolloutLogging`).

- [ ] **Step 7: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add go2_locomotion/go2_locomotion/controllers/nn_policy_controller.py \
        go2_locomotion/go2_locomotion/locomotion_node.py config/params.yaml \
        go2_locomotion/test/test_nn_controller.py
git commit -m "Log real rollout from NNPolicyController when nn_policy.log_path set"
```

---

## Task 3: Comparison / plotting script (`sim2real_compare.py`)

**Files:**
- Create: `go2_locomotion/tools/sim2real_compare.py`
- Test: `go2_locomotion/test/test_sim2real_compare.py`

**Interfaces:**
- Consumes: `rollout_log.read_log`, `rollout_log.COLUMNS` (Task 1).
- Produces:
  - `align_lengths(real_cols, sim_cols) -> int` — returns the common step count (min), never
    raises on mismatch.
  - `rmse(a: np.ndarray, b: np.ndarray) -> float` — NaN-aware; returns `nan` if no finite
    pairs.
  - `gap_table(real_cols, sim_cols, n_common, steady_start=20) -> list[dict]` — per-signal
    rows with `signal`, `rmse_full`, `rmse_steady`.
  - `main()` — CLI: `--real`, `--sim`, `--out`, `--steady-start` (default 20).

- [ ] **Step 1: Write the failing tests**

Create `go2_locomotion/test/test_sim2real_compare.py`:

```python
import importlib.util, os, sys
import numpy as np

_MOD = os.path.join(os.path.dirname(__file__), "..", "tools", "sim2real_compare.py")
_spec = importlib.util.spec_from_file_location("sim2real_compare", _MOD)
s2r = importlib.util.module_from_spec(_spec)
sys.modules["sim2real_compare"] = s2r
_spec.loader.exec_module(s2r)


def test_rmse_basic():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert abs(s2r.rmse(a, b) - np.sqrt((9 + 16 + 0) / 3)) < 1e-9


def test_rmse_ignores_nan():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    assert s2r.rmse(a, b) == 0.0   # only finite pairs (indices 0,2) count


def test_rmse_all_nan_returns_nan():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    assert np.isnan(s2r.rmse(a, b))


def test_align_lengths_truncates_to_min():
    real = {"step": np.arange(10.0)}
    sim = {"step": np.arange(7.0)}
    assert s2r.align_lengths(real, sim) == 7


def test_gap_table_offset_case():
    n = 50
    real = {f"q_{i}": np.zeros(n) for i in range(12)}
    sim = {f"q_{i}": np.zeros(n) for i in range(12)}
    real["tau_0"] = np.zeros(n); sim["tau_0"] = np.zeros(n)
    real["base_roll"] = np.zeros(n); sim["base_roll"] = np.zeros(n)
    real["base_pitch"] = np.zeros(n); sim["base_pitch"] = np.zeros(n)
    # constant 0.1 offset on q_3
    sim["q_3"] = np.full(n, 0.1)
    rows = s2r.gap_table(real, sim, n_common=n, steady_start=20)
    by_sig = {r["signal"]: r for r in rows}
    assert abs(by_sig["q_3"]["rmse_full"] - 0.1) < 1e-9
    assert abs(by_sig["q_0"]["rmse_full"]) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_sim2real_compare.py -v`
Expected: FAIL — `AttributeError` (functions not defined).

- [ ] **Step 3: Implement `sim2real_compare.py`**

Create `go2_locomotion/tools/sim2real_compare.py`:

```python
#!/usr/bin/env python3
"""Compare a real rollout log against an Isaac Lab replay log (dynamics-gap analysis).

Usage:
    python3 tools/sim2real_compare.py --real real_log --sim sim_log --out out_dir
(paths are the base name without .csv/.json)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from go2_locomotion.utils import rollout_log as rl


def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def align_lengths(real_cols, sim_cols):
    return int(min(len(real_cols["step"]), len(sim_cols["step"])))


def _signals():
    joints = [f"q_{i}" for i in range(12)] + [f"tau_{i}" for i in range(12)]
    return joints + ["base_roll", "base_pitch"]


def gap_table(real_cols, sim_cols, n_common, steady_start=20):
    rows = []
    for sig in _signals():
        r = real_cols[sig][:n_common]
        s = sim_cols[sig][:n_common]
        rows.append({
            "signal": sig,
            "rmse_full": rmse(r, s),
            "rmse_steady": rmse(r[steady_start:], s[steady_start:]),
        })
    return rows


def _plot_group(names, titles, real_cols, sim_cols, n, out_path, extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 3
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    t = np.arange(n)
    for idx, (name, title) in enumerate(zip(names, titles)):
        ax = axes[idx // ncol][idx % ncol]
        if extra and name in extra:
            ax.plot(t, extra[name][:n], "k--", lw=0.8, label=extra["_label"])
        ax.plot(t, real_cols[name][:n], "b", lw=0.9, label="real")
        ax.plot(t, sim_cols[name][:n], "r", lw=0.9, label="sim")
        ax.set_title(title); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steady-start", type=int, default=20)
    args = p.parse_args()

    _, real_cols = rl.read_log(args.real)
    _, sim_cols = rl.read_log(args.sim)
    n = align_lengths(real_cols, sim_cols)
    if len(real_cols["step"]) != len(sim_cols["step"]):
        print(f"경고: 길이 불일치 real={len(real_cols['step'])} sim={len(sim_cols['step'])} → {n}로 truncate")

    os.makedirs(args.out, exist_ok=True)

    jn = [f"q_{i}" for i in range(12)]
    tgt = {f"q_{i}": real_cols[f"target_q_{i}"] for i in range(12)}
    tgt["_label"] = "target"
    _plot_group(jn, jn, real_cols, sim_cols, n, os.path.join(args.out, "joints.png"), extra=tgt)

    tn = [f"tau_{i}" for i in range(12)]
    _plot_group(tn, tn, real_cols, sim_cols, n, os.path.join(args.out, "torque.png"))

    bn = ["base_roll", "base_pitch", "base_gyro_x", "base_gyro_y", "base_gyro_z"]
    _plot_group(bn, bn, real_cols, sim_cols, n, os.path.join(args.out, "base.png"))

    rows = gap_table(real_cols, sim_cols, n, args.steady_start)
    print(f"{'signal':<14}{'rmse_full':>12}{'rmse_steady':>14}")
    with open(os.path.join(args.out, "gap_summary.csv"), "w") as f:
        f.write("signal,rmse_full,rmse_steady\n")
        for r in rows:
            print(f"{r['signal']:<14}{r['rmse_full']:>12.5f}{r['rmse_steady']:>14.5f}")
            f.write(f"{r['signal']},{r['rmse_full']},{r['rmse_steady']}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/test_sim2real_compare.py -v`
Expected: 5 passed.

- [ ] **Step 5: Smoke-test the CLI on synthetic logs**

Run:
```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
python3 - <<'EOF'
import numpy as np
from go2_locomotion.utils import rollout_log as rl
def mk(base, jitter):
    rows=[]
    for i in range(30):
        rows.append(rl.make_row(step=i,t=i*0.02,cmd=(0.5,0,0),
            raw_action=[0.0]*12, target_q=[0.1]*12,
            q=[0.1+jitter]*12, dq=[0.0]*12, tau=[1.0+jitter]*12,
            quat=(1.0,0.0,0.0,0.0), gyro=(0.0,0.0,0.0)))
    rl.write_log(base, rows, {"source":"real","joint_sdk_names":["j%d"%i for i in range(12)]})
mk("/tmp/real_demo", 0.0); mk("/tmp/sim_demo", 0.02)
EOF
python3 tools/sim2real_compare.py --real /tmp/real_demo --sim /tmp/sim_demo --out /tmp/s2r_out
ls /tmp/s2r_out
```
Expected: prints a gap table (q_* rmse ≈ 0.02, tau_* rmse ≈ 0.02) and `/tmp/s2r_out` contains
`joints.png`, `torque.png`, `base.png`, `gap_summary.csv`.

- [ ] **Step 6: Commit**

```bash
cd /home/csh/smll_project/locomotion/go2_locomotion
git add go2_locomotion/tools/sim2real_compare.py go2_locomotion/test/test_sim2real_compare.py
git commit -m "Add sim2real comparison script (overlay plots + RMSE gap table)"
```

---

## Task 4: Isaac Lab replay script + joint-order helper

**Files:**
- Create: `unitree_rl_lab/scripts/rsl_rl/joint_order.py`
- Create: `unitree_rl_lab/scripts/rsl_rl/test_joint_order.py`
- Create: `unitree_rl_lab/scripts/rsl_rl/replay_real_log.py`

**Interfaces:**
- Consumes: the shared schema (a copy of `rollout_log.py` semantics — reads via the JSON
  `joint_sdk_names` and the same CSV columns).
- Produces: `sdk_to_isaac_indices(isaac_joint_names, joint_sdk_names) -> np.ndarray` and its
  inverse `isaac_to_sdk_indices(...)`.

- [ ] **Step 1: Write the failing test (joint-order helper)**

Create `unitree_rl_lab/scripts/rsl_rl/test_joint_order.py`:

```python
import importlib.util, os, sys
import numpy as np

_MOD = os.path.join(os.path.dirname(__file__), "joint_order.py")
_spec = importlib.util.spec_from_file_location("joint_order", _MOD)
jo = importlib.util.module_from_spec(_spec)
sys.modules["joint_order"] = jo
_spec.loader.exec_module(jo)

SDK = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
       "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
# a plausible Isaac order: all hips, then all thighs, then all calves
ISAAC = ["FR_hip", "FL_hip", "RR_hip", "RL_hip",
         "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
         "FR_calf", "FL_calf", "RR_calf", "RL_calf"]


def test_sdk_to_isaac_is_permutation():
    idx = jo.sdk_to_isaac_indices(ISAAC, SDK)
    assert sorted(idx.tolist()) == list(range(12))


def test_roundtrip_reorders_correctly():
    idx = jo.sdk_to_isaac_indices(ISAAC, SDK)
    inv = jo.isaac_to_sdk_indices(ISAAC, SDK)
    # a value in SDK order, moved to isaac order and back, is unchanged
    v_sdk = np.arange(12.0)
    v_isaac = v_sdk[idx]
    np.testing.assert_array_equal(v_isaac[inv], v_sdk)
    # isaac vector index for SDK "FR_calf" (sdk idx 2) sits where ISAAC has FR_calf (idx 8)
    assert idx[2] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/csh/smll_project/unitree_rl_lab && python3 -m pytest scripts/rsl_rl/test_joint_order.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 3: Implement `joint_order.py`**

Create `unitree_rl_lab/scripts/rsl_rl/joint_order.py`:

```python
"""Pure-Python SDK<->Isaac joint-order index maps (no Isaac import, unit-testable)."""
import numpy as np


def sdk_to_isaac_indices(isaac_joint_names, joint_sdk_names):
    """Return idx s.t. v_sdk[idx] reorders an SDK-ordered vector into Isaac order.

    idx[k] = position in the SDK vector of the joint that Isaac slot k holds.
    """
    sdk_pos = {name: i for i, name in enumerate(joint_sdk_names)}
    return np.array([sdk_pos[name] for name in isaac_joint_names], dtype=np.intp)


def isaac_to_sdk_indices(isaac_joint_names, joint_sdk_names):
    """Return idx s.t. v_isaac[idx] reorders an Isaac-ordered vector into SDK order."""
    isaac_pos = {name: i for i, name in enumerate(isaac_joint_names)}
    return np.array([isaac_pos[name] for name in joint_sdk_names], dtype=np.intp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/csh/smll_project/unitree_rl_lab && python3 -m pytest scripts/rsl_rl/test_joint_order.py -v`
Expected: 2 passed.

- [ ] **Step 5: Implement `replay_real_log.py` (not executable in dev env)**

Create `unitree_rl_lab/scripts/rsl_rl/replay_real_log.py`. This mirrors `play.py`'s setup;
it is validated on the training machine, not here.

```python
#!/usr/bin/env python3
"""Open-loop replay of a real rollout log in Isaac Lab; writes a sim log in the same schema.

Feeds the recorded policy actions into env.step() (no policy), disables randomization, and
matches the real action clip (±6). Run on the training machine:

    python3 scripts/rsl_rl/replay_real_log.py --task <GO2_VELOCITY_TASK> \
        --real_log /path/real_log --out /path/sim_log --num_envs 1 --headless
"""
import argparse
import csv
import json

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--real_log", required=True, help="base path (no ext) of the real log")
parser.add_argument("--out", required=True, help="base path (no ext) for the sim log")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from joint_order import isaac_to_sdk_indices, sdk_to_isaac_indices  # noqa: E402

_JOINT_GROUPS = ["raw_action", "target_q", "q", "dq", "tau"]
COLUMNS = (
    ["step", "t", "cmd_vx", "cmd_vy", "cmd_vyaw"]
    + [f"{g}_{i}" for g in _JOINT_GROUPS for i in range(12)]
    + ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
       "base_roll", "base_pitch", "base_gyro_x", "base_gyro_y", "base_gyro_z",
       "base_height", "base_vx", "base_vy", "base_vyaw"]
)


def _read_real(base):
    with open(base + ".json") as f:
        meta = json.load(f)
    raw_actions, cmds = [], []
    with open(base + ".csv") as f:
        for row in csv.DictReader(f):
            raw_actions.append([float(row[f"raw_action_{i}"]) for i in range(12)])
            cmds.append((float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_vyaw"])))
    return meta, np.array(raw_actions, dtype=np.float32), cmds


def _roll_pitch(w, x, y, z):
    import math
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def main():
    meta, raw_actions, cmds = _read_real(args.real_log)
    sdk_names = meta["joint_sdk_names"]

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # nominal dynamics: disable randomization / push / obs noise, match real action clip
    if hasattr(env_cfg, "events"):
        for attr in list(vars(env_cfg.events)):
            if any(k in attr for k in ("push", "random", "material", "mass")):
                setattr(env_cfg.events, attr, None)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.actions.JointPositionAction.clip = {".*": (-6.0, 6.0)}

    env = gym.make(args.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    isaac_names = robot.data.joint_names
    isaac2sdk = isaac_to_sdk_indices(isaac_names, sdk_names)  # v_isaac[idx] -> sdk order
    sdk2isaac = sdk_to_isaac_indices(isaac_names, sdk_names)

    obs, _ = env.reset()
    # reset joints to the real initial pose (SDK order -> Isaac order)
    q0_sdk = np.array(meta["initial_q"], dtype=np.float32) if "initial_q" in meta else None
    if q0_sdk is not None:
        q0_isaac = torch.tensor(q0_sdk[sdk2isaac], device=env.unwrapped.device).unsqueeze(0)
        robot.write_joint_state_to_sim(q0_isaac, torch.zeros_like(q0_isaac))

    rows = []
    dt = float(meta.get("dt", 0.02))
    for step in range(len(raw_actions)):
        act = torch.tensor(raw_actions[step], device=env.unwrapped.device).unsqueeze(0)
        obs, _, _, _, _ = env.step(act)
        d = robot.data
        q = d.joint_pos[0].cpu().numpy()[isaac2sdk]
        dq = d.joint_vel[0].cpu().numpy()[isaac2sdk]
        tau = d.applied_torque[0].cpu().numpy()[isaac2sdk]
        target_q = (d.joint_pos_target[0].cpu().numpy()[isaac2sdk]
                    if hasattr(d, "joint_pos_target") else q)
        quat = d.root_quat_w[0].cpu().numpy()  # (w,x,y,z)
        gyro = d.root_ang_vel_b[0].cpu().numpy()
        lin = d.root_lin_vel_b[0].cpu().numpy()
        roll, pitch = _roll_pitch(*quat)
        row = {"step": step, "t": step * dt,
               "cmd_vx": cmds[step][0], "cmd_vy": cmds[step][1], "cmd_vyaw": cmds[step][2],
               "base_quat_w": quat[0], "base_quat_x": quat[1],
               "base_quat_y": quat[2], "base_quat_z": quat[3],
               "base_roll": roll, "base_pitch": pitch,
               "base_gyro_x": gyro[0], "base_gyro_y": gyro[1], "base_gyro_z": gyro[2],
               "base_height": float(d.root_pos_w[0, 2].cpu()),
               "base_vx": lin[0], "base_vy": lin[1], "base_vyaw": gyro[2]}
        for name, arr in zip(_JOINT_GROUPS, [raw_actions[step], target_q, q, dq, tau]):
            for i in range(12):
                row[f"{name}_{i}"] = float(arr[i])
        rows.append(row)

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out + ".json", "w") as f:
        json.dump({"source": "sim", "version": meta.get("version", 1), "dt": dt,
                   "raw_action_order": "policy", "measured_order": "sdk",
                   "joint_sdk_names": sdk_names, "n_steps": len(rows)}, f, indent=2)
    print(f"sim log written: {args.out}.csv/.json ({len(rows)} steps)")
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
```

Note for the implementer: exact Isaac Lab attribute names (`joint_pos_target`,
`applied_torque`, event term names, action clip cfg path) can vary by Isaac Lab version —
verify against the installed version on the training machine and adjust. The joint-order
helper (Step 3) and the CSV schema are the parts pinned by tests.

- [ ] **Step 6: Verify the schema constants match `rollout_log.py`**

Run:
```bash
cd /home/csh/smll_project
python3 - <<'EOF'
import importlib.util, sys
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
rl = load("locomotion/go2_locomotion/go2_locomotion/utils/rollout_log.py", "rl")
# rebuild COLUMNS the way replay_real_log.py does and compare
groups = ["raw_action","target_q","q","dq","tau"]
cols = (["step","t","cmd_vx","cmd_vy","cmd_vyaw"]
        + [f"{g}_{i}" for g in groups for i in range(12)]
        + ["base_quat_w","base_quat_x","base_quat_y","base_quat_z","base_roll","base_pitch",
           "base_gyro_x","base_gyro_y","base_gyro_z","base_height","base_vx","base_vy","base_vyaw"])
assert cols == rl.COLUMNS, "replay COLUMNS drifted from rollout_log.COLUMNS"
print("schema match OK")
EOF
```
Expected: prints `schema match OK`.

- [ ] **Step 7: Commit**

```bash
cd /home/csh/smll_project/unitree_rl_lab
git add scripts/rsl_rl/joint_order.py scripts/rsl_rl/test_joint_order.py scripts/rsl_rl/replay_real_log.py
git commit -m "Add Isaac Lab open-loop replay script + joint-order helper for sim2real"
```

(Note: `unitree_rl_lab` is a separate git repo; commit there. The go2_locomotion pieces were
committed in Tasks 1-3.)

---

## Final verification

- [ ] Run the full go2_locomotion suite:
  `cd /home/csh/smll_project/locomotion/go2_locomotion && python3 -m pytest test/ --ignore=test/test_locomotion_node.py -v`
  Expected: all pass (existing + `test_rollout_log.py` + `TestRolloutLogging` + `test_sim2real_compare.py`).
- [ ] Run the unitree_rl_lab helper test:
  `cd /home/csh/smll_project/unitree_rl_lab && python3 -m pytest scripts/rsl_rl/test_joint_order.py -v`
  Expected: pass.
