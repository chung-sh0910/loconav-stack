"""
LocomotionNode ROS2 통합 테스트 — 실제 하드웨어/DDS 없이 동작.

conftest.py가 unitree_sdk2py SDK mock을 삽입하고,
각 테스트는 _build_sdk_controller / _build_nn_controller를 MagicMock으로 교체해
ROS2 노드 자체의 로직(watchdog, mode-switch, e-stop, param 변경 등)만 검증한다.
"""
import time
import threading
from unittest.mock import MagicMock, patch, call

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool, Trigger

from go2_locomotion.locomotion_node import LocomotionNode
from go2_locomotion.controllers.base_controller import BaseController


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """모듈 전체에서 rclpy context 공유."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_mock_controller(name="mock_ctrl"):
    ctrl = MagicMock(spec=BaseController)
    ctrl.name = name
    return ctrl


@pytest.fixture
def node_and_ctrls():
    """
    LocomotionNode + 두 mock controller를 생성하고,
    테스트 종료 시 destroy_node를 자동 호출한다.
    """
    sdk_ctrl = _make_mock_controller("sdk_velocity")
    nn_ctrl  = _make_mock_controller("nn_policy")

    with patch.object(LocomotionNode, '_build_sdk_controller', return_value=sdk_ctrl), \
         patch.object(LocomotionNode, '_build_nn_controller',  return_value=nn_ctrl):
        node = LocomotionNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    yield node, sdk_ctrl, nn_ctrl, executor

    node.destroy_node()


def _spin_until(executor, condition, timeout=2.0, step=0.02):
    """condition()이 True가 될 때까지 최대 timeout초 spin."""
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        executor.spin_once(timeout_sec=step)
    return condition()


def _call_service(executor, node, srv_type, srv_name, request):
    client = node.create_client(srv_type, srv_name)
    future = client.call_async(request)
    assert _spin_until(executor, lambda: future.done()), \
        f"{srv_name} 서비스 응답 timeout"
    return future.result()


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------

class TestInit:

    def test_initial_mode_is_sdk_velocity(self, node_and_ctrls):
        node, sdk_ctrl, nn_ctrl, _ = node_and_ctrls
        assert node._control_mode_name == 'sdk_velocity'

    def test_sdk_controller_started_on_init(self, node_and_ctrls):
        _, sdk_ctrl, _, _ = node_and_ctrls
        sdk_ctrl.start.assert_called_once()

    def test_nn_controller_not_started_on_init(self, node_and_ctrls):
        _, _, nn_ctrl, _ = node_and_ctrls
        nn_ctrl.start.assert_not_called()

    def test_estop_inactive_on_init(self, node_and_ctrls):
        node, *_ = node_and_ctrls
        assert not node._estop_active

    def test_active_controller_is_sdk(self, node_and_ctrls):
        node, sdk_ctrl, _, _ = node_and_ctrls
        assert node._active_controller is sdk_ctrl


# ---------------------------------------------------------------------------
# /set_control_mode 서비스
# ---------------------------------------------------------------------------

class TestModeSwitch:

    def test_switch_to_nn_policy(self, node_and_ctrls):
        node, sdk_ctrl, nn_ctrl, executor = node_and_ctrls
        req = SetBool.Request()
        req.data = True   # True → nn_policy
        resp = _call_service(executor, node, SetBool, '/set_control_mode', req)
        assert resp.success
        assert node._control_mode_name == 'nn_policy'

    def test_switch_to_nn_starts_nn_controller(self, node_and_ctrls):
        node, _, nn_ctrl, executor = node_and_ctrls
        req = SetBool.Request()
        req.data = True
        _call_service(executor, node, SetBool, '/set_control_mode', req)
        nn_ctrl.start.assert_called()

    def test_switch_to_sdk_velocity(self, node_and_ctrls):
        node, _, _, executor = node_and_ctrls
        # 먼저 nn으로 전환 후 sdk로 복귀
        for data in [True, False]:
            req = SetBool.Request()
            req.data = data
            _call_service(executor, node, SetBool, '/set_control_mode', req)
        assert node._control_mode_name == 'sdk_velocity'

    def test_switch_blocked_during_estop(self, node_and_ctrls):
        node, _, _, executor = node_and_ctrls
        node._estop_active = True
        req = SetBool.Request()
        req.data = True
        resp = _call_service(executor, node, SetBool, '/set_control_mode', req)
        assert not resp.success
        node._estop_active = False   # 다음 테스트를 위해 복원


# ---------------------------------------------------------------------------
# /emergency_stop 서비스
# ---------------------------------------------------------------------------

class TestEmergencyStop:

    def test_estop_sets_flag(self, node_and_ctrls):
        node, _, _, executor = node_and_ctrls
        node._estop_active = False
        req = Trigger.Request()
        resp = _call_service(executor, node, Trigger, '/emergency_stop', req)
        assert resp.success
        assert node._estop_active

    def test_estop_calls_emergency_stop_on_controller(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        # sdk_velocity 모드로 확인
        req = Trigger.Request()
        _call_service(executor, node, Trigger, '/emergency_stop', req)
        node._active_controller.emergency_stop.assert_called()

    def test_estop_blocks_step(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = True
        sdk_ctrl.step.reset_mock()
        # 컨트롤 타이머 콜백을 직접 호출
        node._control_timer_callback()
        sdk_ctrl.step.assert_not_called()
        node._estop_active = False


# ---------------------------------------------------------------------------
# /recover 서비스
# ---------------------------------------------------------------------------

class TestRecover:

    def test_recover_requires_active_estop(self, node_and_ctrls):
        node, _, _, executor = node_and_ctrls
        node._estop_active = False
        req = Trigger.Request()
        resp = _call_service(executor, node, Trigger, '/recover', req)
        assert not resp.success

    def test_recover_clears_estop_flag(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = True
        sdk_ctrl.recover.return_value = None   # 예외 없이 성공
        req = Trigger.Request()
        resp = _call_service(executor, node, Trigger, '/recover', req)
        assert resp.success
        assert not node._estop_active

    def test_recover_calls_controller_recover(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = True
        sdk_ctrl.recover.reset_mock()
        req = Trigger.Request()
        _call_service(executor, node, Trigger, '/recover', req)
        sdk_ctrl.recover.assert_called_once()


# ---------------------------------------------------------------------------
# /cmd_ctr 구독 → update_cmd_vel
# ---------------------------------------------------------------------------

class TestCmdCtr:

    def _pub_cmd(self, node, executor, vx, vy, vyaw):
        pub = node.create_publisher(TwistStamped, '/cmd_ctr', 10)
        msg = TwistStamped()
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.angular.z = vyaw
        pub.publish(msg)
        executor.spin_once(timeout_sec=0.1)

    def test_cmd_ctr_updates_controller(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        node._watchdog_triggered = False
        sdk_ctrl.update_cmd_vel.reset_mock()
        self._pub_cmd(node, executor, 0.3, 0.1, -0.2)
        sdk_ctrl.update_cmd_vel.assert_called_once_with(0.3, 0.1, -0.2)

    def test_cmd_ctr_resets_watchdog(self, node_and_ctrls):
        node, _, _, executor = node_and_ctrls
        node._watchdog_triggered = True
        self._pub_cmd(node, executor, 0.0, 0.0, 0.0)
        with node._watchdog_lock:
            assert not node._watchdog_triggered


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class TestWatchdog:

    def test_watchdog_triggers_after_timeout(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        node._watchdog_triggered = False
        # 마지막 수신 시각을 오래 전으로 조작
        node._last_cmd_time = node.get_clock().now() - \
            rclpy.duration.Duration(seconds=10)
        node._watchdog_callback()
        with node._watchdog_lock:
            assert node._watchdog_triggered

    def test_watchdog_sends_zero_cmd(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._watchdog_triggered = False
        sdk_ctrl.update_cmd_vel.reset_mock()
        node._last_cmd_time = node.get_clock().now() - \
            rclpy.duration.Duration(seconds=10)
        node._watchdog_callback()
        sdk_ctrl.update_cmd_vel.assert_called_with(0.0, 0.0, 0.0)

    def test_watchdog_blocks_step(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        node._watchdog_triggered = True
        sdk_ctrl.step.reset_mock()
        node._control_timer_callback()
        sdk_ctrl.step.assert_not_called()
        node._watchdog_triggered = False


# ---------------------------------------------------------------------------
# 컨트롤 루프 (step)
# ---------------------------------------------------------------------------

class TestControlLoop:

    def test_step_called_when_active(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        node._watchdog_triggered = False
        node._transition_active = False
        sdk_ctrl.step.reset_mock()
        node._control_timer_callback()
        sdk_ctrl.step.assert_called_once()

    def test_step_skipped_during_transition(self, node_and_ctrls):
        node, sdk_ctrl, _, executor = node_and_ctrls
        node._estop_active = False
        node._watchdog_triggered = False
        node._transition_active = True
        sdk_ctrl.step.reset_mock()
        node._control_timer_callback()
        sdk_ctrl.step.assert_not_called()
        node._transition_active = False


# ---------------------------------------------------------------------------
# destroy_node
# ---------------------------------------------------------------------------

class TestDestroyNode:

    def test_destroy_calls_stop(self):
        sdk_ctrl = _make_mock_controller("sdk_velocity")
        nn_ctrl  = _make_mock_controller("nn_policy")

        with patch.object(LocomotionNode, '_build_sdk_controller', return_value=sdk_ctrl), \
             patch.object(LocomotionNode, '_build_nn_controller',  return_value=nn_ctrl):
            node = LocomotionNode()

        sdk_ctrl.stop.reset_mock()
        node.destroy_node()
        sdk_ctrl.stop.assert_called()
