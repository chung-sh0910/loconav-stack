import numpy as np


def quat_to_projected_gravity(quaternion: list) -> np.ndarray:
    """
    Convert IMU quaternion to gravity vector in body frame.

    Args:
        quaternion: [w, x, y, z]
    Returns:
        np.ndarray of shape (3,) — gravity direction in body frame
    """
    qw, qx, qy, qz = quaternion
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity
