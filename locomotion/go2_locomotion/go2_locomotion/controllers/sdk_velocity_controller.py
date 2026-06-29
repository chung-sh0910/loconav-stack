import threading
import time
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
        # sport 모드가 항상 켜져 있어 네이티브 명령으로 바로 처리.
        # StandDown(): 네 다리 모두 접어 천천히 엎드림 → Damp(): 힘 빼기(수동)
        if self._client is not None:
            self._client.StandDown()    # 네 발 다 접고 엎드림 — 천천히 내려앉음
            time.sleep(2.0)             # 엎드리기 완료 대기
            self._client.Damp()         # kp=0, 모터 힘 빠짐

    def recover(self) -> None:
        # sport mode는 RecoveryStand()로 바로 일어설 수 있음
        if self._client is not None:
            self._client.RecoveryStand()
