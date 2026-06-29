import numpy as np
import pytest
from go2_locomotion.utils.imu_utils import quat_to_projected_gravity


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def normalize(v):
    return np.array(v, dtype=np.float64) / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 기본 케이스
# ---------------------------------------------------------------------------

def test_identity_points_down():
    """identity quaternion → 로봇이 수평 → gravity는 body -Z 방향 [0, 0, -1]."""
    g = quat_to_projected_gravity([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)


def test_output_shape_and_dtype():
    g = quat_to_projected_gravity([1.0, 0.0, 0.0, 0.0])
    assert g.shape == (3,)
    assert g.dtype == np.float32


def test_unit_length():
    """gravity 벡터 크기는 항상 1이어야 한다 (단위 quaternion 입력 기준)."""
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)
    for quat in [
        [1.0, 0.0, 0.0, 0.0],
        [c, s, 0.0, 0.0],          # 정확한 √2/2
        [0.5, 0.5, 0.5, 0.5],      # 이미 단위 quaternion (0.5²×4=1)
    ]:
        g = quat_to_projected_gravity(quat)
        np.testing.assert_allclose(np.linalg.norm(g), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 90° 회전 케이스
# ---------------------------------------------------------------------------

def test_pitch_forward_90():
    """로봇이 앞으로 90° 숙임 → gravity가 body +X 방향.

    R_y(+90°): body X(전방)가 world -Z(아래)를 가리킴.
    따라서 world gravity [0,0,-1]은 body +X 방향으로 투영됨.
    quaternion for Y-rotation +90°: [cos(45°), 0, sin(45°), 0]
    """
    angle = np.pi / 2
    quat = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    g = quat_to_projected_gravity(quat)
    np.testing.assert_allclose(g, [1.0, 0.0, 0.0], atol=1e-5)


def test_roll_right_90():
    """로봇이 오른쪽으로 90° 기울음 → gravity가 body -Y 방향."""
    # roll +90° (right): rot around X by +90° → [cos(45°), sin(45°), 0, 0]
    angle = np.pi / 2
    quat = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
    g = quat_to_projected_gravity(quat)
    np.testing.assert_allclose(g, [0.0, -1.0, 0.0], atol=1e-5)


def test_upside_down():
    """로봇이 뒤집힘 → gravity가 body +Z 방향."""
    # 180° roll around X
    angle = np.pi
    quat = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
    g = quat_to_projected_gravity(quat)
    np.testing.assert_allclose(g, [0.0, 0.0, 1.0], atol=1e-5)
