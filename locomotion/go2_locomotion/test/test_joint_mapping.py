"""
joint_ids_map 적용 검증 테스트.

policy 출력이 올바른 SDK 관절에 도달하는지,
observation도 올바른 SDK 관절에서 읽히는지 확인한다.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.utils.go2_constants import (
    DEFAULT_JOINT_POS, NUM_JOINTS, JOINT_IDS_MAP,
)
from test.test_nn_controller import make_fake_lowstate, make_ctrl


# ---------------------------------------------------------------------------
# JOINT_IDS_MAP 상수 sanity check
# ---------------------------------------------------------------------------

def test_joint_ids_map_length():
    assert len(JOINT_IDS_MAP) == NUM_JOINTS

def test_joint_ids_map_is_permutation():
    """맵이 0~11의 완전한 순열이어야 한다 (중복·누락 없음)."""
    assert sorted(JOINT_IDS_MAP) == list(range(NUM_JOINTS))

def test_joint_ids_map_not_identity():
    """맵이 [0,1,2,...] 그대로이면 버그를 잡지 못한다."""
    assert list(JOINT_IDS_MAP) != list(range(NUM_JOINTS))


# ---------------------------------------------------------------------------
# action → SDK 관절 매핑 검증
# ---------------------------------------------------------------------------

def test_step_routes_action_to_correct_sdk_joint():
    """
    policy[policy_i] = 1.0 (나머지 0) 일 때
    target_q[sdk_i] 만 DEFAULT_JOINT_POS[sdk_i] + scale 이어야 한다.
    """
    action_scale = 0.25

    for policy_i in range(NUM_JOINTS):
        sdk_i = JOINT_IDS_MAP[policy_i]

        raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        raw_action[policy_i] = 1.0
        policy = MagicMock(return_value=raw_action)

        ctrl = make_ctrl(policy=policy, action_scale=action_scale)
        ctrl._lowstate = make_fake_lowstate()

        captured = {}
        def capture(q, _i=sdk_i):
            captured['q'] = q.copy()

        with patch.object(ctrl, '_send_low_cmd', side_effect=capture):
            ctrl.step()

        q = captured['q']
        expected_delta = action_scale  # clip(1.0,-1,1)*0.25

        # sdk_i 관절만 이동해야 한다
        assert abs(q[sdk_i] - (DEFAULT_JOINT_POS[sdk_i] + expected_delta)) < 1e-5, \
            f"policy[{policy_i}] → sdk[{sdk_i}]: " \
            f"expected {DEFAULT_JOINT_POS[sdk_i] + expected_delta:.4f}, got {q[sdk_i]:.4f}"

        # 나머지 관절은 DEFAULT_JOINT_POS 그대로여야 한다
        for j in range(NUM_JOINTS):
            if j != sdk_i:
                assert abs(q[j] - DEFAULT_JOINT_POS[j]) < 1e-5, \
                    f"policy[{policy_i}] → sdk[{sdk_i}]: " \
                    f"sdk[{j}]이 의도치 않게 변경됨: {q[j]:.4f} != {DEFAULT_JOINT_POS[j]:.4f}"


# ---------------------------------------------------------------------------
# observation → policy 관절 순서 검증
# ---------------------------------------------------------------------------

def test_observation_reads_correct_sdk_joint():
    """
    sdk[sdk_i] = DEFAULT_JOINT_POS[sdk_i] + 0.5 (나머지 default) 일 때
    obs[9 + policy_i] 만 0.5 이어야 한다.
    """
    for policy_i in range(NUM_JOINTS):
        sdk_i = JOINT_IDS_MAP[policy_i]

        joint_q = list(DEFAULT_JOINT_POS)
        joint_q[sdk_i] += 0.5

        ctrl = make_ctrl()
        ls   = make_fake_lowstate(joint_q=joint_q)
        obs  = ctrl._build_observation(ls)

        assert abs(obs[9 + policy_i] - 0.5) < 1e-5, \
            f"sdk[{sdk_i}]의 변화가 obs[9+{policy_i}]에 반영 안 됨: {obs[9+policy_i]:.4f}"

        for j in range(NUM_JOINTS):
            if j != policy_i:
                assert abs(obs[9 + j]) < 1e-5, \
                    f"sdk[{sdk_i}] 변경이 obs[9+{j}](policy_j={j})에 누출됨: {obs[9+j]:.4f}"
