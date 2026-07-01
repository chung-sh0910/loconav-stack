# Recurrent (GRU/LSTM) policy support in convert_pt_to_onnx.py

## Context

Training now produces `rsl_rl` `ActorCriticRecurrent`/`ActorCriticRecurrentVel` checkpoints
(recurrent memory + MLP head) instead of plain MLP `ActorCritic` checkpoints. The runtime
inference side, `go2_locomotion/policy/onnx_policy.py`, already auto-detects GRU vs. LSTM from
the ONNX graph's input names (`h_in`/`c_in`) and needs no changes.

The only gap is `tools/convert_pt_to_onnx.py`, which currently only knows how to reconstruct and
export the non-recurrent MLP actor. It cannot convert a recurrent `.pt` checkpoint (e.g.
`/home/csh/go2_logs3/unitree_go2_velocity/2026-06-30_13-04-56/model_7400.pt`, confirmed via
inspection to be a single-layer GRU: `memory_a.rnn.weight_hh_l0` shape `(768, 256)` → hidden=256,
768/256=3 gates = GRU) to ONNX.

## Design

Extend `tools/convert_pt_to_onnx.py` to detect and handle recurrent checkpoints automatically,
with no new required CLI flags:

1. **Detection**: after loading the raw state dict, check for any key starting with
   `memory_a.rnn.`. If present, the checkpoint is recurrent; otherwise fall back to the existing
   MLP-only path unchanged.

2. **Architecture inference from tensor shapes** (no new flags):
   - `num_layers`: count of `memory_a.rnn.weight_ih_l{i}` keys present.
   - `hidden_size`: `memory_a.rnn.weight_hh_l0.shape[1]`.
   - `input_size`: `memory_a.rnn.weight_ih_l0.shape[1]` (cross-checked against `--obs-dim`; warn
     on mismatch rather than silently proceeding).
   - `rnn_type`: `weight_ih_l0.shape[0] / hidden_size` → 3 gates = GRU, 4 gates = LSTM. Any other
     ratio raises a clear error.

3. **Model construction**: a small wrapper module holding `self.rnn` (`nn.GRU`/`nn.LSTM` built
   from the inferred params) and `self.actor` (the existing `build_actor()` MLP, but with
   `in_dim=hidden_size` instead of `obs_dim`, using the existing `--hidden-dims`/`--action-dim`
   flags for the head).
   - `forward_lstm(obs, h_in, c_in)` / `forward_gru(obs, h_in)`: run the RNN for one step
     (`unsqueeze(0)`/`squeeze(0)` around the single-step sequence dim, mirroring Isaac Lab's own
     `_OnnxPolicyExporter`), then feed the RNN output through `self.actor`.

4. **Weight loading**: load `memory_a.rnn.*` into `self.rnn` and `actor.*` into `self.actor`
   (reusing the existing prefix-based `actor.`/`mlp.` auto-detection). Keys under `critic.*`,
   `memory_c.*`, and `vel_head.*` are present in the checkpoint but irrelevant to inference and
   are ignored.

5. **ONNX export**: input/output names must exactly match what `onnx_policy.py` expects:
   - GRU: inputs `obs, h_in` → outputs `actions, h_out`
   - LSTM: inputs `obs, h_in, c_in` → outputs `actions, h_out, c_out`
   - `dynamic_axes={}`, `opset_version=18` (unchanged from current script).

6. **Sanity check**: after export, run one `onnxruntime` inference step with zeroed hidden
   state(s) (and zeroed cell state for LSTM) and print the output shape/sample, same as the
   existing non-recurrent sanity check.

7. **Docstring/usage**: update the module docstring to drop the "(MLP non-recurrent 전용)"
   caveat and document the auto-detected recurrent path.

## Out of scope

- `onnx_policy.py`, `base_policy.py`, `nn_policy_controller.py` — already correct, no changes.
- `rsl_rl` / `unitree_rl_lab` — no changes; they're the training-side source of truth, not touched.
- CLI overrides for `rnn_type`/`hidden_size`/`num_layers` — explicitly rejected in favor of full
  auto-detection (user-approved).
- Any change to observation building or joint mapping in `nn_policy_controller.py`.

## Testing

- Convert the real checkpoint
  (`/home/csh/go2_logs3/unitree_go2_velocity/2026-06-30_13-04-56/model_7400.pt`) to ONNX with the
  updated script and confirm the sanity-check inference step runs without error.
- Load the resulting ONNX file with `OnnxPolicy` directly (outside pytest) and confirm
  `_is_gru` is `True` and repeated `__call__` + `reset()` don't crash and produce finite actions.
