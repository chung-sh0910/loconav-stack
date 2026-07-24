"""Torch-free runtime for the MoE / Hist-MLP ONNX exports (convert_moe_to_onnx.py).

Runs on the Go2 with only numpy + onnxruntime (no torch). Builds the 225-d history exactly like
:class:`MoEPolicy`, runs the STATELESS ONNX graph, and -- for the MoE export -- applies argmax +
routing persistence (hysteresis + min-dwell) in numpy to pick the expert, so the deployed routing
matches the torch model bit-for-bit (incl. the live ``current_expert``). The MLP export is a plain
``obs_history -> action`` and needs no routing.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import onnxruntime as ort

from .base_policy import BasePolicy

# Self-contained (no import of moe_policy, which pulls torch-adjacent code) so the Go2 runtime is
# numpy + onnxruntime only -- works on the robot's Python 3.8. These MUST match the training obs.
OBS_DIM = 45
HISTORY_LENGTH = 5
# single-frame term boundaries (== NNPolicyController._build_observation / the trained PolicyCfg order)
TERM_SLICES = [(0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45)]


class OnnxMoEPolicy(BasePolicy):
    """ONNX runtime for the MoE (routing + live expert) or Hist-MLP export.

    ``route_hysteresis`` / ``route_min_dwell`` MUST match what you want at deployment (defaults are
    the demo values). ``force_expert`` (>=0) pins one expert, like the torch demo's --force-expert.
    """

    def __init__(self, model_path: str, route_hysteresis: float = 1.0, route_min_dwell: int = 10,
                 force_expert: int | None = None, **kwargs):
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        in_names = [i.name for i in self._sess.get_inputs()]
        # the MoE export has an obs_policy input + gate_logits output; the MLP export is history-only
        self._is_moe = "obs_policy" in in_names
        self._hyst = float(route_hysteresis)
        self._dwell_max = int(route_min_dwell)
        self._force = force_expert if (force_expert is not None and force_expert >= 0) else None
        self._hist: deque | None = None
        self._held = -1
        self._dwell = 0
        self.current_expert = -1
        print(f"[OnnxMoEPolicy] {model_path}  backend={'MoE' if self._is_moe else 'Hist-MLP'}"
              + (f"  FORCED->expert {self._force}" if self._force is not None else "")
              + (f"  hysteresis={self._hyst} min_dwell={self._dwell_max}" if self._is_moe else ""))

    def _build_history(self, obs: np.ndarray) -> np.ndarray:
        if self._hist is None:
            self._hist = deque([obs.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
        else:
            self._hist.append(obs.copy())
        frames = list(self._hist)  # oldest -> newest
        blocks = [frame[a:b] for (a, b) in TERM_SLICES for frame in frames]
        return np.concatenate(blocks).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        assert obs.shape[0] == OBS_DIM, f"OnnxMoEPolicy expects {OBS_DIM}-dim obs, got {obs.shape[0]}"
        hist = self._build_history(obs)[None, :]  # (1, 225)
        if not self._is_moe:
            action = self._sess.run(None, {"obs_history": hist})[0]
            self.current_expert = -1
            return action.reshape(-1).astype(np.float32)
        acts, logits = self._sess.run(None, {"obs_history": hist, "obs_policy": obs[None, :]})
        mode = self._select(logits[0])            # logits: (K,)
        self.current_expert = mode
        return acts[0, mode].astype(np.float32)   # acts: (K, 12)

    def _select(self, logits: np.ndarray) -> int:
        """argmax + persistence, replicating MoE._route_with_persistence (deployment path, no floor)."""
        if self._force is not None:
            return self._force
        proposal = int(np.argmax(logits))
        if self._held < 0:                        # first step after reset: adopt the proposal
            self._held, self._dwell = proposal, 0
            return proposal
        held = self._held
        if self._hyst > 0.0 or self._dwell_max > 0:
            beats = logits[proposal] > logits[held] + self._hyst
            switch = (proposal != held) and (self._dwell >= self._dwell_max) and bool(beats)
            mode = proposal if switch else held
        else:
            switch = proposal != held
            mode = proposal
        self._dwell = 0 if switch else self._dwell + 1
        self._held = mode
        return mode

    def reset(self) -> None:
        self._hist = None
        self._held = -1
        self._dwell = 0
        self.current_expert = -1
