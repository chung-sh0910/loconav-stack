# Sim2Real Dynamics-Gap Open-Loop Replay Comparison — Design

## Context

The Go2 GRU locomotion policy now walks on hardware after fixing the PD gains
(RL walking at Kp=25/Kd=0.5). To close the remaining sim-to-real gap we want a
signal-level comparison, not just video: how do the real robot's joint/torque/base
trajectories differ from the training simulator (Isaac Lab) under **identical
actions**?

Decisions made during brainstorming:
- **Sim reference = Isaac Lab** (the training simulator), not MuJoCo. MuJoCo would add
  its own sim2sim gap.
- **Method = open-loop action replay.** Log the real robot's action sequence, replay the
  exact same actions in Isaac Lab, and compare the resulting trajectories. This isolates
  the *dynamics* gap (actuator + contact + inertia + latency) from the *policy* gap.
- **This measures the dynamics gap only.** The policy-inference gap (ONNX vs training
  policy) was already verified separately (ONNX ≡ PyTorch to 4.6e-6 over 200 steps).
  Because replay feeds recorded actions instead of running the policy in sim, the GRU
  hidden state is irrelevant to the sim rollout.
- **Signals = joints + base + torque.**
- **Real base linear velocity and height are NOT logged** (Go2 LowState doesn't measure
  them; FAST_LIO integration is out of scope for now). Real base comparison uses only
  IMU-derived orientation (roll/pitch) and gyro. Sim logs base velocity/height as
  ground truth for its own reference but they are not compared against real.

## Architecture

Four components bound by one shared log schema, spanning two repos:

```
real robot (go2_locomotion, ROS)                Isaac Lab (unitree_rl_lab, GPU)
  NNPolicyController.step() ──► RolloutLogger      replay_real_log.py
        │                          │ writes             │ reads real_log, feeds
        │                          ▼                     ▼ recorded raw_action
        │                    real_log.csv/json  ──scp──► env.step() (no policy)
        │                                                │ writes
        ▼                                                ▼
                                                   sim_log.csv/json
                            both logs ──► sim2real_compare.py (go2_locomotion/tools)
                                              └─► overlay plots + RMSE gap table (PNG + stdout)
```

The **log schema is the contract**; each component is independently testable against it.

## Component 1: Shared log schema

One CSV (one row per 50 Hz control step) + one sidecar JSON of metadata. All **measured**
joint-indexed columns (`target_q`, `q`, `dq`, `tau`) are in **SDK order**
(FR, FL, RR, RL × hip, thigh, calf) so real and sim align. Isaac Lab uses a different
internal joint order and MUST remap to SDK order using the robot cfg's `joint_sdk_names`.

**Exception — `raw_action` is in POLICY order, not SDK order.** The policy (ONNX actor)
emits its 12 actions in the training/Isaac joint order, and the real controller remaps them
to SDK order via `JOINT_IDS_MAP` only when building `target_q`. Logging `raw_action` in its
native policy order means the Isaac replay can feed it straight into `env.step()` with **no
remap on the action path** (the correctness-critical path). The measured signals stay SDK
order for real↔sim alignment. This split is recorded in the JSON metadata
(`raw_action_order: "policy"`, `measured_order: "sdk"`) so no consumer can guess wrong.

**CSV columns (per step):**
- `step` (int), `t` (float, seconds since start)
- `cmd_vx`, `cmd_vy`, `cmd_vyaw`
- `raw_action_0..11` — policy output before scale/offset, **in policy/Isaac joint order**
  (used to drive replay; see the ordering note above)
- `target_q_0..11` — commanded joint position sent to motors (SDK order)
- `q_0..11` — measured joint position (SDK order)
- `dq_0..11` — measured joint velocity (SDK order)
- `tau_0..11` — real: `motor_state.tau_est`; sim: applied joint torque (SDK order)
- `base_quat_w`, `base_quat_x`, `base_quat_y`, `base_quat_z`
- `base_roll`, `base_pitch` — derived from quat for convenience
- `base_gyro_x`, `base_gyro_y`, `base_gyro_z`
- `base_height`, `base_vx`, `base_vy`, `base_vyaw` — sim: ground truth; real: `NaN`

**JSON sidecar (metadata):**
- `source`: `"real"` | `"sim"`
- `version`: schema version int (bumped if columns change)
- `dt`: 0.02
- `raw_action_order`: `"policy"`, `measured_order`: `"sdk"`
- `joint_sdk_names`: the 12-name SDK order (single source of truth for column ordering)
- `action_scale`: 0.25, `action_clip`: 6.0
- `policy_kp`, `policy_kd`: the RL walking gains used
- `initial_base_quat`, `initial_q` (SDK order): for replay init matching
- `n_steps`

A tiny shared helper module defines the column list and read/write functions so real and
sim producers cannot drift. Location: `go2_locomotion/go2_locomotion/utils/rollout_log.py`
(pure Python + numpy, no ROS). The Isaac Lab side imports the same column definitions by
copying this module into `unitree_rl_lab` (the two repos deploy separately; the JSON
`joint_sdk_names` + a schema `version` field guard against drift).

## Component 2: Real-robot rollout logger

`go2_locomotion/go2_locomotion/utils/rollout_log.py` provides `RolloutLogger`:
- `RolloutLogger(path, metadata)` — holds an in-memory list of rows.
- `append(row: dict)` — O(1), no disk I/O (keeps the 50 Hz loop clean).
- `flush()` — writes CSV + JSON once, at stop.

`NNPolicyController`:
- New optional constructor arg `log_path: str = None` (wired from a new
  `nn_policy.log_path` ROS param, default `""` = disabled).
- When enabled, `start()` creates the `RolloutLogger` with metadata (dt, joint_sdk_names,
  gains, action_scale/clip, initial q/quat captured at first step).
- `step()` appends one row: it already computes `obs`, `raw_action`, `target_q`, and reads
  `lowstate` (q, dq, `tau_est`, imu quat/gyro). Base velocity/height columns = `NaN`.
- `stop()` calls `logger.flush()`.

Joint remap: the controller already maps policy order ↔ SDK order via `JOINT_IDS_MAP`;
`q`, `dq`, `tau`, `target_q` are logged in SDK order (as read from `motor_state[sdk_i]`).

## Component 3: Isaac Lab replay script

`unitree_rl_lab/scripts/rsl_rl/replay_real_log.py`, based on `play.py`:
- Loads `real_log.csv/json`.
- Builds the go2 velocity env with **num_envs=1**, and disables stochastic dynamics for a
  clean nominal comparison: no domain randomization, no push/velocity events, no observation
  noise. (Concretely: in the env cfg, remove/disable `events` randomization terms and set
  `observations` corruption off.)
- Sets the action term's clip to ±6 to match the real controller (the env default is
  ±100; replay must match real so applied joint targets are identical).
- Resets the robot to the log's `initial_q` (remapped SDK→Isaac order) and
  `initial_base_quat`; base height to the env default standing height.
- Rollout loop: for each real step, call `env.step(raw_action)` using the **recorded**
  `raw_action` instead of `policy(obs)`. `raw_action` is already in policy/Isaac order, so
  it is fed directly with no remap (see schema ordering note).
- Logs each step in the shared schema (remapping Isaac→SDK order), with base
  height/velocity from ground truth (`root_pos_w[2]`, `root_lin_vel_b`, `root_ang_vel_b`),
  torque from `robot.data.applied_torque`.
- Writes `sim_log.csv/json`.

**Known limitations (documented, not fixed here):** perfect initial-state match is
impossible (real base pose is not fully observed); the first ~10–20 steps diverge more, so
the comparison reports both full-trace and steady-state (drop first N) metrics. Real
actuation/sensing latency is not modeled in sim — it is part of the measured gap.

## Component 4: Comparison / plotting script

`go2_locomotion/tools/sim2real_compare.py` (numpy + pandas + matplotlib; no ROS/Isaac):
- `sim2real_compare.py --real real_log.csv --sim sim_log.csv --out out_dir/`
- Loads both, validates schema `version` and matching `joint_sdk_names`.
- Aligns by `step`; if lengths differ, truncate to the shorter and print a warning.
- Produces:
  - **Joint positions**: 12 subplots, each overlaying `target_q`, real `q`, sim `q`.
  - **Torque**: 12 subplots overlaying real `tau` vs sim `tau`.
  - **Base**: roll, pitch, gyro_x/y/z overlays (real vs sim). Base height/velocity plotted
    sim-only (real is NaN) and skipped if all-NaN.
  - **Gap summary table** (stdout + saved `gap_summary.csv`): per-joint position RMSE,
    per-joint torque RMSE, base roll/pitch RMSE — computed over the full trace and over the
    steady-state window (steps ≥ N, default N=20).
- Saves all figures as PNG to `out_dir/`.

## Testing

Runnable in this environment (no GPU/Isaac):
- `rollout_log.py`: round-trip test (write rows → read back → identical), schema column
  count/order test.
- `RolloutLogger` via `NNPolicyController`: with a mock lowstate and logging enabled, one
  `step()` records a row with the correct keys and SDK-ordered values; `stop()` writes a
  valid CSV+JSON.
- `sim2real_compare.py`: two synthetic logs → correct RMSE (including a known-offset case),
  length-mismatch truncation + warning, all-NaN base columns skipped without error.
- Joint remap helper (SDK↔Isaac order) used by the replay script: unit test the mapping is
  a correct permutation/inverse.

NOT runnable here (requires the training machine): `replay_real_log.py` end-to-end. It gets
a schema round-trip test only; real execution and the actual gap analysis happen on the
GPU machine.

## Out of scope
- FAST_LIO / real base velocity & height estimation (columns left NaN).
- Closed-loop matched-command comparison and observation-distribution analysis.
- Automatically tuning sim params to close the gap (this design measures the gap; acting on
  it is follow-up work).
- MuJoCo as a sim reference.
