"""
NNPolicyController 단위 테스트 — 하드웨어/DDS 없이 동작.

conftest.py가 unitree_sdk2py 하드웨어 모듈을 mock으로 교체해 두므로
각 테스트에서 별도 패치 불필요.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.utils.go2_constants import (
    DEFAULT_JOINT_POS, NUM_JOINTS,
    MAX_VX, MIN_VX, MAX_VY, MAX_VYAW,
    KD_PASSIVE,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def make_fake_lowstate(
    gyro=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    joint_q=None,
    joint_dq=None,
):
    """unitree LowState 구조체를 흉내 낸 MagicMock.

    MagicMock.__getitem__은 모든 인덱스에 동일한 return_value를 반환하므로
    motor_state 슬롯별로 개별 MagicMock을 만들어 side_effect로 연결한다.
    """
    joint_q   = joint_q   if joint_q   is not None else [0.0] * NUM_JOINTS
    joint_dq  = joint_dq  if joint_dq  is not None else [0.0] * NUM_JOINTS

    # 슬롯별 독립 mock (최대 20개 — unitree motor 배열 크기)
    motor_mocks = [MagicMock() for _ in range(20)]
    for i in range(NUM_JOINTS):
        motor_mocks[i].q  = joint_q[i]
        motor_mocks[i].dq = joint_dq[i]

    ls = MagicMock()
    ls.imu_state.gyroscope  = list(gyro)
    ls.imu_state.quaternion = list(quat)
    ls.motor_state.__getitem__.side_effect = lambda i: motor_mocks[i]
    return ls


def make_ctrl(policy=None, obs_dim=45, action_scale=0.25):
    return NNPolicyController(policy=policy, obs_dim=obs_dim, action_scale=action_scale)


# ---------------------------------------------------------------------------
# _build_observation
# ---------------------------------------------------------------------------

class TestBuildObservation:

    def test_output_shape(self):
        ctrl = make_ctrl(obs_dim=45)
        ls   = make_fake_lowstate()
        obs  = ctrl._build_observation(ls)
        assert obs.shape == (45,)

    def test_output_dtype(self):
        ctrl = make_ctrl(obs_dim=45)
        obs  = ctrl._build_observation(make_fake_lowstate())
        assert obs.dtype == np.float32

    def test_gyro_scale(self):
        """gyro는 0.2 배율로 들어가야 한다 (deploy.yaml base_ang_vel scale)."""
        ctrl = make_ctrl()
        ls   = make_fake_lowstate(gyro=(1.0, 2.0, 3.0))
        obs  = ctrl._build_observation(ls)
        np.testing.assert_allclose(obs[0:3], [0.2, 0.4, 0.6], atol=1e-6)

    def test_identity_gravity(self):
        """identity quaternion → projected_gravity = [0, 0, -1]."""
        ctrl = make_ctrl()
        ls   = make_fake_lowstate(quat=(1.0, 0.0, 0.0, 0.0))
        obs  = ctrl._build_observation(ls)
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-5)

    def test_velocity_command_written(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(0.5, 0.2, -0.3)
        obs  = ctrl._build_observation(make_fake_lowstate())
        np.testing.assert_allclose(obs[6:9], [0.5, 0.2, -0.3], atol=1e-6)

    def test_joint_pos_relative_to_default(self):
        """joint_pos_rel = q - DEFAULT_JOINT_POS."""
        ctrl = make_ctrl()
        joint_q = list(DEFAULT_JOINT_POS + 0.1)
        ls   = make_fake_lowstate(joint_q=joint_q)
        obs  = ctrl._build_observation(ls)
        np.testing.assert_allclose(obs[9:21], [0.1] * NUM_JOINTS, atol=1e-6)

    def test_joint_vel_scale(self):
        """joint_vel_rel = dq * 0.05 (deploy.yaml joint_vel_rel scale)."""
        ctrl = make_ctrl()
        ls   = make_fake_lowstate(joint_dq=[2.0] * NUM_JOINTS)
        obs  = ctrl._build_observation(ls)
        np.testing.assert_allclose(obs[21:33], [0.1] * NUM_JOINTS, atol=1e-6)

    def test_last_action_zero_on_init(self):
        """초기화 직후 prev_actions는 모두 0."""
        ctrl = make_ctrl()
        obs  = ctrl._build_observation(make_fake_lowstate())
        np.testing.assert_array_equal(obs[33:45], np.zeros(NUM_JOINTS))

    def test_last_action_updated_after_step(self):
        """step() 후 prev_actions가 policy output으로 갱신되어야 한다."""
        raw_action = np.ones(NUM_JOINTS, dtype=np.float32) * 0.5
        policy = MagicMock(return_value=raw_action)
        ctrl   = make_ctrl(policy=policy)
        ctrl._lowstate = make_fake_lowstate()

        with patch.object(ctrl, '_send_low_cmd'):
            ctrl.step()

        obs = ctrl._build_observation(make_fake_lowstate())
        np.testing.assert_array_equal(obs[33:45], raw_action)


# ---------------------------------------------------------------------------
# update_cmd_vel 클램핑
# ---------------------------------------------------------------------------

class TestUpdateCmdVel:

    def test_clamp_vx_above_max(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(MAX_VX + 5, 0.0, 0.0)
        with ctrl._cmd_lock:
            assert ctrl._vx == MAX_VX

    def test_clamp_vx_below_min(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(MIN_VX - 5, 0.0, 0.0)
        with ctrl._cmd_lock:
            assert ctrl._vx == MIN_VX

    def test_clamp_vy_symmetric(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(0.0, MAX_VY + 5, 0.0)
        with ctrl._cmd_lock:
            assert ctrl._vy == MAX_VY
        ctrl.update_cmd_vel(0.0, -(MAX_VY + 5), 0.0)
        with ctrl._cmd_lock:
            assert ctrl._vy == -MAX_VY

    def test_clamp_vyaw(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(0.0, 0.0, MAX_VYAW + 5)
        with ctrl._cmd_lock:
            assert ctrl._vyaw == MAX_VYAW

    def test_zero_passthrough(self):
        ctrl = make_ctrl()
        ctrl.update_cmd_vel(0.0, 0.0, 0.0)
        with ctrl._cmd_lock:
            assert ctrl._vx == 0.0
            assert ctrl._vy == 0.0
            assert ctrl._vyaw == 0.0


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_skipped_when_policy_none(self):
        ctrl = make_ctrl(policy=None)
        ctrl._lowstate = make_fake_lowstate()
        with patch.object(ctrl, '_send_low_cmd') as mock_send:
            ctrl.step()
            mock_send.assert_not_called()

    def test_step_skipped_when_no_lowstate(self):
        policy = MagicMock(return_value=np.zeros(NUM_JOINTS, dtype=np.float32))
        ctrl   = make_ctrl(policy=policy)
        # _lowstate는 None (기본값)
        with patch.object(ctrl, '_send_low_cmd') as mock_send:
            ctrl.step()
            mock_send.assert_not_called()

    def test_step_calls_policy_with_obs(self):
        raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        policy = MagicMock(return_value=raw_action)
        ctrl   = make_ctrl(policy=policy)
        ctrl._lowstate = make_fake_lowstate()

        with patch.object(ctrl, '_send_low_cmd'):
            ctrl.step()

        policy.assert_called_once()
        obs_arg = policy.call_args[0][0]
        assert obs_arg.shape == (45,)
        assert obs_arg.dtype == np.float32

    def test_step_calls_send_low_cmd(self):
        policy = MagicMock(return_value=np.zeros(NUM_JOINTS, dtype=np.float32))
        ctrl   = make_ctrl(policy=policy)
        ctrl._lowstate = make_fake_lowstate()

        with patch.object(ctrl, '_send_low_cmd') as mock_send:
            ctrl.step()
            mock_send.assert_called_once()

    def test_step_action_scale_applied(self):
        """target_q = DEFAULT_JOINT_POS + clip(raw, -1, 1) * action_scale."""
        raw_action = np.ones(NUM_JOINTS, dtype=np.float32)   # clip 후 1.0
        policy = MagicMock(return_value=raw_action)
        ctrl   = make_ctrl(policy=policy, action_scale=0.25)
        ctrl._lowstate = make_fake_lowstate()

        captured = {}
        def capture(target_q):
            captured['q'] = target_q.copy()

        with patch.object(ctrl, '_send_low_cmd', side_effect=capture):
            ctrl.step()

        expected = DEFAULT_JOINT_POS + 1.0 * 0.25
        np.testing.assert_allclose(captured['q'], expected, atol=1e-6)

    def test_step_clips_raw_action(self):
        """raw action > 1 이면 clip 후 1.0 으로 제한."""
        raw_action = np.full(NUM_JOINTS, 5.0, dtype=np.float32)
        policy = MagicMock(return_value=raw_action)
        ctrl   = make_ctrl(policy=policy, action_scale=0.25)
        ctrl._lowstate = make_fake_lowstate()

        captured = {}
        def capture(target_q):
            captured['q'] = target_q.copy()

        with patch.object(ctrl, '_send_low_cmd', side_effect=capture):
            ctrl.step()

        expected = DEFAULT_JOINT_POS + 1.0 * 0.25   # clip(5, -1, 1) = 1
        np.testing.assert_allclose(captured['q'], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# stop() / emergency_stop()
# ---------------------------------------------------------------------------

class TestStopAndEmergency:

    def test_stop_sends_default_pos(self):
        ctrl = make_ctrl()
        ctrl._cmd_pub = MagicMock()   # start() 없이 stop()을 테스트하려면 pub을 설정해야 함
        captured = {}
        def capture(q):
            captured['q'] = q.copy()

        with patch.object(ctrl, '_send_low_cmd', side_effect=capture):
            ctrl.stop()

        np.testing.assert_allclose(captured['q'], DEFAULT_JOINT_POS, atol=1e-6)

    def test_stop_no_error_when_pub_none(self):
        """_cmd_pub가 None이면 stop()은 아무것도 안 해야 한다."""
        ctrl = make_ctrl()
        ctrl._cmd_pub = None
        ctrl.stop()   # 예외 없어야 함

    def test_emergency_stop_no_error_when_pub_none(self):
        ctrl = make_ctrl()
        ctrl._cmd_pub = None
        ctrl.emergency_stop()

    def test_emergency_stop_sends_passive_gains(self):
        """emergency_stop은 kp=0, kd=KD_PASSIVE 모드를 보내야 한다."""
        ctrl = make_ctrl()
        ctrl._cmd_pub = MagicMock()

        sent_msgs = []
        ctrl._cmd_pub.Write.side_effect = lambda m: sent_msgs.append(m)

        ctrl.emergency_stop()

        assert len(sent_msgs) == 1
        msg = sent_msgs[0]
        for i in range(NUM_JOINTS):
            assert msg.motor_cmd[i].mode == 0x00, f"joint {i}: mode should be 0x00 (Damp)"
            assert msg.motor_cmd[i].kp == 0.0,   f"joint {i}: kp should be 0"
            assert msg.motor_cmd[i].kd == KD_PASSIVE[i % NUM_JOINTS], f"joint {i}: kd mismatch"
