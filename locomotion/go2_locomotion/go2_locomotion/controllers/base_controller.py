from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Abstract base for Go2 locomotion controllers.

    Lifecycle:
      1. __init__()         — construction
      2. start()            — activate (init SDK / load model)
      3. update_cmd_vel()   — called from /cmd_vel subscriber (may be any thread)
      4. step()             — called by 50Hz ROS2 timer
      5. stop()             — deactivate (send zeros, release resources)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def update_cmd_vel(self, vx: float, vy: float, vyaw: float) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately cut motor torque to passive/damp state."""
        ...

    @abstractmethod
    def recover(self) -> None:
        """비상정지 후 복구. 로봇을 일으켜 세우고 정상 제어 상태로 복귀."""
        ...
