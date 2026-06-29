import pytest
from go2_locomotion.utils.go2_constants import MAX_VX, MIN_VX, MAX_VY, MAX_VYAW


def _clamp(value, low, high):
    """update_cmd_vel()에서 쓰는 clamp 로직과 동일."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 경계값 테스트
# ---------------------------------------------------------------------------

class TestVxClamp:
    def test_within_range(self):
        assert _clamp(0.5, MIN_VX, MAX_VX) == 0.5

    def test_above_max(self):
        assert _clamp(MAX_VX + 1.0, MIN_VX, MAX_VX) == MAX_VX

    def test_below_min(self):
        assert _clamp(MIN_VX - 1.0, MIN_VX, MAX_VX) == MIN_VX

    def test_exact_max(self):
        assert _clamp(MAX_VX, MIN_VX, MAX_VX) == MAX_VX

    def test_exact_min(self):
        assert _clamp(MIN_VX, MIN_VX, MAX_VX) == MIN_VX

    def test_zero(self):
        assert _clamp(0.0, MIN_VX, MAX_VX) == 0.0

    def test_backward_allowed(self):
        """MIN_VX는 음수여야 한다 — 후진 가능."""
        assert MIN_VX < 0.0


class TestVyClamp:
    def test_within_range(self):
        assert _clamp(0.2, -MAX_VY, MAX_VY) == 0.2

    def test_above_max(self):
        assert _clamp(MAX_VY + 1.0, -MAX_VY, MAX_VY) == MAX_VY

    def test_below_min(self):
        assert _clamp(-MAX_VY - 1.0, -MAX_VY, MAX_VY) == -MAX_VY

    def test_symmetric(self):
        """좌우 대칭 — 양방향 한계가 동일해야 한다."""
        pos = _clamp(MAX_VY + 1.0, -MAX_VY, MAX_VY)
        neg = _clamp(-MAX_VY - 1.0, -MAX_VY, MAX_VY)
        assert pos == -neg


class TestVyawClamp:
    def test_within_range(self):
        assert _clamp(0.5, -MAX_VYAW, MAX_VYAW) == 0.5

    def test_above_max(self):
        assert _clamp(MAX_VYAW + 1.0, -MAX_VYAW, MAX_VYAW) == MAX_VYAW

    def test_below_min(self):
        assert _clamp(-MAX_VYAW - 1.0, -MAX_VYAW, MAX_VYAW) == -MAX_VYAW


# ---------------------------------------------------------------------------
# 상수 범위 sanity check
# ---------------------------------------------------------------------------

def test_velocity_limits_positive():
    assert MAX_VX > 0
    assert MAX_VY > 0
    assert MAX_VYAW > 0

def test_min_vx_negative():
    assert MIN_VX < 0, "후진을 위해 MIN_VX는 음수여야 한다"
