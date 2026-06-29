from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """
    Abstract neural network policy plug-in interface.

    Observation layout (42-dim default for Isaac Lab Go2):
      [0:3]   projected_gravity  (3)  — gravity vector in body frame
      [3:6]   vel_commands       (3)  — [vx, vy, vyaw]
      [6:18]  joint_pos_rel      (12) — current pos minus default standing pos
      [18:30] joint_vel          (12) — joint angular velocities
      [30:42] prev_actions       (12) — previous network output

    Output (12): joint position offsets (added to default standing pos)

    To plug in a model:
      Create go2_locomotion/policy/onnx_policy.py implementing BasePolicy,
      then set nn_policy.model_path in params.yaml.
    """

    @abstractmethod
    def __init__(self, model_path: str, **kwargs):
        ...

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """
        Args:
            observation: float32 array of shape (obs_dim,)
        Returns:
            action: float32 array of shape (12,) — joint position offsets
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recurrent state (if any)."""
        ...
