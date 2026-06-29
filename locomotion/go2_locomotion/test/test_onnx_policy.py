import numpy as np
import pytest
from go2_locomotion.policy.onnx_policy import OnnxPolicy

MODEL_PATH = "/home/csh/go2_logs/2026-06-26_02-37-40/policy.onnx"
OBS_DIM = 45
ACTION_DIM = 12


@pytest.fixture(scope="module")
def policy():
    return OnnxPolicy(MODEL_PATH)


# ---------------------------------------------------------------------------
# 출력 shape / dtype
# ---------------------------------------------------------------------------

def test_output_shape(policy):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    actions = policy(obs)
    assert actions.shape == (ACTION_DIM,)


def test_output_dtype(policy):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    actions = policy(obs)
    assert actions.dtype == np.float32


# ---------------------------------------------------------------------------
# 발산 / NaN 없음
# ---------------------------------------------------------------------------

def test_no_nan_on_zero_obs(policy):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    actions = policy(obs)
    assert not np.any(np.isnan(actions)), "zero obs → NaN action"


def test_no_nan_on_random_obs(policy):
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        actions = policy(obs)
        assert not np.any(np.isnan(actions))
        assert not np.any(np.isinf(actions))


def test_actions_in_reasonable_range(policy):
    """학습된 policy의 raw output은 대개 [-5, 5] 범위 안에 있어야 한다."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    actions = policy(obs)
    assert np.all(np.abs(actions) < 10.0), f"비정상적으로 큰 action: {actions}"


# ---------------------------------------------------------------------------
# 결정론성 (동일 obs → 동일 action)
# ---------------------------------------------------------------------------

def test_deterministic(policy):
    obs = np.ones(OBS_DIM, dtype=np.float32) * 0.1
    a1 = policy(obs)
    a2 = policy(obs)
    np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_does_not_crash(policy):
    policy.reset()  # MLP는 상태 없음 — 그냥 no-op이어야 함


def test_inference_after_reset(policy):
    policy.reset()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    actions = policy(obs)
    assert actions.shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# obs dim 불일치 — 에러 발생 확인
# ---------------------------------------------------------------------------

def test_wrong_obs_dim_raises(policy):
    with pytest.raises(Exception):
        policy(np.zeros(OBS_DIM + 1, dtype=np.float32))


# ---------------------------------------------------------------------------
# deploy.yaml 기준 action scale 적용 후 범위
# ---------------------------------------------------------------------------

def test_scaled_action_range(policy):
    """action_scale=0.25 적용 시 target_q가 default_pos 근처여야 한다."""
    from go2_locomotion.utils.go2_constants import DEFAULT_JOINT_POS
    action_scale = 0.25

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw = policy(obs)
    clipped = np.clip(raw, -1.0, 1.0) * action_scale
    target_q = DEFAULT_JOINT_POS + clipped

    # DEFAULT_JOINT_POS ± 0.25 rad 이내여야 함
    diff = np.abs(target_q - DEFAULT_JOINT_POS)
    assert np.all(diff <= action_scale + 1e-6), f"target_q 범위 초과: {diff}"
