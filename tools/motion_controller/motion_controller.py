from __future__ import annotations

import logging
import time

import numpy as np

from filters import LowPassFilter

logger = logging.getLogger(__name__)

REMOTE_TIMEOUT_SEC: float = 3.0
CMD_TIMEOUT_SEC: float = 1.0
DEFAULT_PRECISION: int = 8
CURVE_DA_LIMIT: np.ndarray = np.array([20.0, 0.0, 50.0])
VELOCITY_AXES: tuple = ('x', 'y', 'th')


class MotionController:

    # Number of elements in cmd and timestamp arguments expected by process()
    process_args: int = 3

    def __init__(self, control_type: str, service_condition: dict | None):
        self.control_type = control_type
        self.service_condition = service_condition
        self.remote_condition: dict | None = None

        self._is_remote: bool = False
        self._active_constraints: dict = {}
        self._acceleration: list[float] = [0.0, 0.0, 0.0]
        self.prev_cmd_ctr: list[float] = [0.0, 0.0, 0.0]
        self._velocity_filters: list[LowPassFilter] = [LowPassFilter() for _ in range(3)]

        # Set externally by the ROS2 adapter before first process() call
        self.control_t: float = 0.02

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, cmd: list, timestamp: list) -> tuple[list, str]:
        cmd, info = self._select_command(cmd, timestamp)
        self._active_constraints = (
            self.remote_condition if self._is_remote else self.service_condition
        )
        output = self.control(cmd, self.prev_cmd_ctr)
        output = self._clip_velocity(output)
        self._update_acceleration(output, self.prev_cmd_ctr)
        self.prev_cmd_ctr = output
        return output, info

    def control(self, cmd: list, current_vel: list) -> list:
        if self.control_type == 'immediate':
            return list(cmd)
        elif self.control_type == 'control':
            return self._control_with_accel(cmd, current_vel)
        elif self.control_type == 'curve':
            return self._control_curve(cmd, current_vel)
        else:
            raise ValueError(f"Unknown control_type: {self.control_type!r}")

    # ------------------------------------------------------------------
    # Command selection
    # ------------------------------------------------------------------

    def _select_command(self, cmd: list, timestamp: list) -> tuple[list, str]:
        robot_vel, cmd_nav, cmd_rmt = cmd
        _ts_vel, timestamp_nav, timestamp_rmt = timestamp
        self._is_remote = False

        now = time.time()
        if now - timestamp_nav > CMD_TIMEOUT_SEC and now - timestamp_rmt > CMD_TIMEOUT_SEC:
            return [0, 0, 0], 'No command in. publish 0,0'

        if timestamp_rmt + REMOTE_TIMEOUT_SEC > now:
            self._is_remote = True
            return cmd_rmt, 'Remote'

        return cmd_nav, 'Navigation'

    # ------------------------------------------------------------------
    # Velocity clipping and acceleration tracking
    # ------------------------------------------------------------------

    def _clip_velocity(self, cmd: list) -> list:
        mins = np.array([self._active_constraints[f'min_velocity_{a}'] for a in VELOCITY_AXES])
        maxs = np.array([self._active_constraints[f'max_velocity_{a}'] for a in VELOCITY_AXES])
        return np.clip(cmd, mins, maxs).tolist()

    def _update_acceleration(self, cmd: list, prev: list) -> None:
        t = round(self.control_t, DEFAULT_PRECISION)
        self._acceleration = ((np.array(cmd) - np.array(prev)) / t).tolist()

    # ------------------------------------------------------------------
    # Control strategies
    # ------------------------------------------------------------------

    def _is_speeding_up(self, target: float, current: float) -> bool:
        return (target > current and current >= 0) or (target < current and current <= 0)

    def _control_with_accel(self, cmd: list, current_vel: list) -> list:
        """Shared acceleration-based control for control modes."""
        t = round(self.control_t, DEFAULT_PRECISION)
        for i, axis in enumerate(VELOCITY_AXES):
            limit_key = 'max_speed_up_%s' if self._is_speeding_up(cmd[i], current_vel[i]) else 'max_speed_down_%s'
            acc_limit = self._active_constraints[limit_key % axis]
            self._acceleration[i] = min(acc_limit, abs((cmd[i] - current_vel[i]) / t))
            if cmd[i] < current_vel[i]:
                self._acceleration[i] *= -1
        return self._integrate_velocity(current_vel, t, cmd)

    def _control_curve(self, cmd: list, current_vel: list) -> list:
        """Acceleration control with jerk limiting (curve_da_limit)."""
        t = round(self.control_t, DEFAULT_PRECISION)
        a_prev = list(self._acceleration)
        self._control_with_accel(cmd, current_vel)
        for i in range(3):
            a_min = a_prev[i] - CURVE_DA_LIMIT[i] * t
            a_max = a_prev[i] + CURVE_DA_LIMIT[i] * t
            clamped = float(np.clip(self._acceleration[i], a_min, a_max))
            if clamped != self._acceleration[i]:
                logger.debug(
                    'curve jerk clamp axis=%d cmd=%.3f a_raw=%.3f -> a_clamped=%.3f',
                    i, cmd[i], self._acceleration[i], clamped,
                )
            self._acceleration[i] = clamped
        return self._integrate_velocity(current_vel, t, cmd)

    def _integrate_velocity(self, current_vel: list, t: float, cmd: list) -> list:
        """Integrate acceleration to produce output velocity [vx, vy, w]."""
        result = []
        for i, (lpf, axis) in enumerate(zip(self._velocity_filters, VELOCITY_AXES)):
            v = round(current_vel[i] + self._acceleration[i] * t, DEFAULT_PRECISION)
            if axis != 'x':  # vx filter is intentionally unused (no lateral smoothing needed)
                v = lpf.filter(v, cmd[i])
            result.append(v)
        return result
