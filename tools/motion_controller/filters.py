import numpy as np

DEFAULT_CUTOFF_FREQ_HZ: int = 10
DEFAULT_TIME_DIFF_SEC: float = 0.02
SPEED_THRESHOLD: float = 1e-3
DEFAULT_FILTER_PRECISION: int = 4


class LowPassFilter:

    def __init__(
        self,
        cutoff_frequency: int = DEFAULT_CUTOFF_FREQ_HZ,
        time_diff: float = DEFAULT_TIME_DIFF_SEC,
    ):
        self.time_diff = time_diff
        self.cutoff_frequency = cutoff_frequency
        self.tau = self._compute_tau()
        self.prev_data: float = 0.0

    def filter(self, data: float, cmd: float, precision: int = DEFAULT_FILTER_PRECISION) -> float:
        if np.abs(cmd) > SPEED_THRESHOLD:
            val = (self.time_diff * data + self.tau * self.prev_data) / (self.tau + self.time_diff)
            self.prev_data = val
            return round(val, precision)
        self.prev_data = 0.0
        return data

    def _compute_tau(self) -> float:
        return 1.0 / (2.0 * np.pi * self.cutoff_frequency)
