import numpy as np
import onnxruntime as ort

from .base_policy import BasePolicy


class OnnxPolicy(BasePolicy):
    """
    Isaac Lab RSL-RL policy loaded from ONNX export.

    Isaac Lab exporter (exporter.py) 기준 input/output 이름:

    Non-recurrent MLP:
      input  : "obs"           (1, obs_dim)
      output : "actions"       (1, 12)

    LSTM:
      inputs : "obs", "h_in", "c_in"
      outputs: "actions", "h_out", "c_out"

    GRU:
      inputs : "obs", "h_in"
      outputs: "actions", "h_out"

    Normalizer는 ONNX 안에 포함되어 있으므로 별도 처리 불필요.
    """

    def __init__(self, model_path: str, **kwargs):
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        input_names = [i.name for i in self._session.get_inputs()]
        self._is_lstm    = "c_in" in input_names
        self._is_gru     = "h_in" in input_names and not self._is_lstm
        self._is_recurrent = self._is_lstm or self._is_gru

        self._hidden_state = None
        self._cell_state   = None

        if self._is_recurrent:
            # hidden state 크기를 ONNX 입력 shape에서 읽음
            for inp in self._session.get_inputs():
                if inp.name == "h_in":
                    shape = inp.shape   # (num_layers, 1, hidden_size)
                    self._hidden_state = np.zeros(shape, dtype=np.float32)
                if inp.name == "c_in":
                    shape = inp.shape
                    self._cell_state   = np.zeros(shape, dtype=np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.astype(np.float32).reshape(1, -1)

        if self._is_lstm:
            outputs = self._session.run(
                None,
                {"obs": obs, "h_in": self._hidden_state, "c_in": self._cell_state},
            )
            actions, self._hidden_state, self._cell_state = outputs[0], outputs[1], outputs[2]

        elif self._is_gru:
            outputs = self._session.run(
                None,
                {"obs": obs, "h_in": self._hidden_state},
            )
            actions, self._hidden_state = outputs[0], outputs[1]

        else:
            outputs = self._session.run(None, {"obs": obs})
            actions = outputs[0]

        return actions.squeeze(0).astype(np.float32)   # (12,)

    def reset(self) -> None:
        if self._hidden_state is not None:
            self._hidden_state[:] = 0.0
        if self._cell_state is not None:
            self._cell_state[:] = 0.0
