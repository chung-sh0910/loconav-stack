import threading

from .base_controller import BaseController
from go2_locomotion.utils.go2_constants import MAX_VX, MIN_VX, MAX_VY, MAX_VYAW


class SDKVelocityController(BaseController):
    """
    Mode 1: High-level velocity control via unitree_sdk2py SportClient.

    /cmd_vel → update_cmd_vel() → _vx/_vy/_vyaw
    50Hz timer → step() → SportClient.Move(_vx, _vy, _vyaw)
    """

    name = "sdk_velocity"

    def __init__(self):
        self._lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._client = None

    def start(self) -> None:
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        self._client = SportClient()
        self._client.SetTimeout(10.0)
        self._client.Init()

    def update_cmd_vel(self, vx: float, vy: float, vyaw: float) -> None:
        with self._lock:
            self._vx   = max(MIN_VX,    min(MAX_VX,    vx))
            self._vy   = max(-MAX_VY,   min(MAX_VY,    vy))
            self._vyaw = max(-MAX_VYAW, min(MAX_VYAW, vyaw))

    def step(self) -> None:
        with self._lock:
            vx, vy, vyaw = self._vx, self._vy, self._vyaw
        if self._client is not None:
            self._client.Move(vx, vy, vyaw)

    def stop(self) -> None:
        if self._client is not None:
            self._client.Move(0.0, 0.0, 0.0)
            self._client.StopMove()

    def emergency_stop(self) -> None:
        # Damp(): kp=0, kd 유지 — 모터 힘 빠지고 중력에 접힘 (sport 컨트롤러 내장 Passive)
        if self._client is not None:
            self._client.Damp()

    def recover(self) -> None:
        # sport mode는 RecoveryStand()로 바로 일어설 수 있음
        if self._client is not None:
            self._client.RecoveryStand()
