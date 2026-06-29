#!/usr/bin/env python3
import threading

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import SetParametersResult

from go2_locomotion.controllers.sdk_velocity_controller import SDKVelocityController
from go2_locomotion.controllers.nn_policy_controller import NNPolicyController

# 내일할일 조이스틱기반으로 실제 제어되는거 확인해보기? 토크값 읽기? 학습해보고 명령을 주었을때 실제 잘움직이는지 확인하기
class LocomotionNode(Node):
    """
    ROS2 node bridging /cmd_ctr to Unitree Go2 via two interchangeable modes.

    Mode 1 (sdk_velocity): SportClient high-level velocity API
    Mode 2 (nn_policy):    Low-level motor position control via NN policy

    Mode switching:
      ros2 service call /set_control_mode std_srvs/srv/SetBool "data: true"
        true  → nn_policy
        false → sdk_velocity
      ros2 param set /locomotion_node control_mode nn_policy
      ros2 launch ... control_mode:=nn_policy
    """

    def __init__(self):
        super().__init__('locomotion_node')

        self.declare_parameter('control_mode',          'sdk_velocity')
        self.declare_parameter('network_interface',     'eth0')
        self.declare_parameter('control_frequency',     50.0)
        self.declare_parameter('watchdog_timeout',      0.5)
        self.declare_parameter('nn_policy.model_path',  '')
        self.declare_parameter('nn_policy.obs_dim',     42)
        self.declare_parameter('nn_policy.action_scale', 0.25)
        self.declare_parameter('nn_policy.kp', [60.0, 80.0, 80.0, 60.0, 80.0, 80.0,
                                                 60.0, 80.0, 80.0, 60.0, 80.0, 80.0])
        self.declare_parameter('nn_policy.kd', [5.0, 4.0, 4.0, 5.0, 4.0, 4.0,
                                                 5.0, 4.0, 4.0, 5.0, 4.0, 4.0])

        network_interface = self.get_parameter('network_interface').value
        freq              = self.get_parameter('control_frequency').value
        self._watchdog_timeout = self.get_parameter('watchdog_timeout').value
        initial_mode      = self.get_parameter('control_mode').value

        # ChannelFactory is a singleton — init once here, not inside each controller
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, network_interface)

        self._controllers = {
            'sdk_velocity': self._build_sdk_controller(),
            'nn_policy':    self._build_nn_controller(),
        }

        self._active_controller = None
        self._switch_lock = threading.Lock()
        self._control_mode_name = ''

        self._watchdog_lock = threading.Lock()
        self._last_cmd_time = self.get_clock().now()
        self._watchdog_triggered = False

        self._estop_active = False
        self._transition_active = False

        self._activate_controller(initial_mode)

        # Callback groups separate the fast real-time loop from slow,
        # blocking lifecycle services (start/recover have sleeps).
        #
        #   cb_realtime: control timer + watchdog + cmd_ctr + emergency_stop
        #     → all mutually exclusive, run on one thread in series.
        #       step() and emergency_stop() can never interleave, so no
        #       concurrent DDS writes / no race on _estop_active.
        #   cb_slow: set_control_mode + recover
        #     → isolated thread; their time.sleep() can't stall control.
        self._cb_realtime = MutuallyExclusiveCallbackGroup()
        self._cb_slow = MutuallyExclusiveCallbackGroup()

        self._cmd_ctr_sub = self.create_subscription(
            TwistStamped, '/cmd_ctr', self._cmd_ctr_callback, 10,
            callback_group=self._cb_realtime,
        )
        self._mode_srv = self.create_service(
            SetBool, '/set_control_mode', self._mode_service_callback,
            callback_group=self._cb_slow,
        )
        self._estop_srv = self.create_service(
            Trigger, '/emergency_stop', self._estop_callback,
            callback_group=self._cb_realtime,
        )
        self._recover_srv = self.create_service(
            Trigger, '/recover', self._recover_callback,
            callback_group=self._cb_slow,
        )
        self._control_timer = self.create_timer(
            1.0 / freq, self._control_timer_callback,
            callback_group=self._cb_realtime,
        )
        self._watchdog_timer = self.create_timer(
            self._watchdog_timeout / 5.0, self._watchdog_callback,
            callback_group=self._cb_realtime,
        )
        self.add_on_set_parameters_callback(self._on_parameter_change)

        self.get_logger().info(
            f'LocomotionNode ready — mode={initial_mode}, '
            f'interface={network_interface}, freq={freq}Hz'
        )

    # ------------------------------------------------------------------
    # Controller construction
    # ------------------------------------------------------------------

    def _build_sdk_controller(self) -> SDKVelocityController:
        return SDKVelocityController()

    def _build_nn_controller(self) -> NNPolicyController:
        model_path   = self.get_parameter('nn_policy.model_path').value
        obs_dim      = self.get_parameter('nn_policy.obs_dim').value
        action_scale = self.get_parameter('nn_policy.action_scale').value
        kp           = self.get_parameter('nn_policy.kp').value
        kd           = self.get_parameter('nn_policy.kd').value

        policy = None
        if model_path:
            try:
                from go2_locomotion.policy.onnx_policy import OnnxPolicy
                policy = OnnxPolicy(model_path)
                self.get_logger().info(f'Loaded NN policy from {model_path}')
            except Exception as e:
                self.get_logger().warning(
                    f'Could not load NN policy from "{model_path}": {e}. '
                    'nn_policy mode will not send commands until a model is loaded.'
                )

        return NNPolicyController(
            policy=policy,
            obs_dim=obs_dim,
            action_scale=action_scale,
            kp=list(kp),
            kd=list(kd),
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _activate_controller(self, mode_name: str) -> bool:
        with self._switch_lock:
            if mode_name not in self._controllers:
                self.get_logger().error(f'Unknown control mode: "{mode_name}"')
                return False
            # Block step() while we stop the old controller and start the new
            # one (start() sends LowCmds + sleeps). Without this the real-time
            # thread could write to DDS concurrently with start().
            self._transition_active = True
            try:
                if self._active_controller is not None:
                    try:
                        self._active_controller.stop()
                    except Exception as e:
                        self.get_logger().error(
                            f'Error stopping {self._control_mode_name}: {e}'
                        )
                new_ctrl = self._controllers[mode_name]
                new_ctrl.start()
                self._active_controller = new_ctrl
                self._control_mode_name = mode_name
                self.get_logger().info(f'Control mode → {mode_name}')
                return True
            except Exception as e:
                self.get_logger().error(f'Failed to start mode "{mode_name}": {e}')
                return False
            finally:
                self._transition_active = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cmd_ctr_callback(self, msg: TwistStamped) -> None:
        with self._watchdog_lock:
            self._last_cmd_time = self.get_clock().now()
            self._watchdog_triggered = False
        if self._active_controller is not None:
            self._active_controller.update_cmd_vel(
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
            )

    def _control_timer_callback(self) -> None:
        if self._active_controller is None:
            return
        if self._estop_active:
            return
        if self._transition_active:
            return
        with self._watchdog_lock:
            triggered = self._watchdog_triggered
        if triggered:
            return
        try:
            self._active_controller.step()
        except Exception as e:
            self.get_logger().error(f'step() error: {e}')

    def _watchdog_callback(self) -> None:
        with self._watchdog_lock:
            elapsed = (
                self.get_clock().now() - self._last_cmd_time
            ).nanoseconds * 1e-9
            if elapsed > self._watchdog_timeout and not self._watchdog_triggered:
                self._watchdog_triggered = True
                self.get_logger().warning(
                    f'Watchdog: no /cmd_ctr for {elapsed:.2f}s — stopping robot'
                )
                if self._active_controller is not None:
                    self._active_controller.update_cmd_vel(0.0, 0.0, 0.0)
                    self._active_controller.stop()

    def _mode_service_callback(self, request, response):
        if self._estop_active:
            response.success = False
            response.message = 'Emergency stop is active — restart node to resume'
            return response
        target = 'nn_policy' if request.data else 'sdk_velocity'
        success = self._activate_controller(target)
        response.success = success
        response.message = f'Switched to {target}' if success else 'Switch failed'
        return response

    def _estop_callback(self, request, response):
        self._estop_active = True
        self.get_logger().fatal('EMERGENCY STOP triggered')
        if self._active_controller is not None:
            try:
                self._active_controller.emergency_stop()
            except Exception as e:
                self.get_logger().error(f'emergency_stop() error: {e}')
        response.success = True
        response.message = 'Emergency stop executed — call /recover to resume'
        return response

    def _recover_callback(self, request, response):
        if not self._estop_active:
            response.success = False
            response.message = 'No emergency stop is active'
            return response
        self.get_logger().warn('Recovery requested — attempting to stand up')
        try:
            self._active_controller.recover()
            self._estop_active = False
            self.get_logger().info('Recovery complete — control resumed')
            response.success = True
            response.message = 'Recovery complete'
        except Exception as e:
            self.get_logger().error(f'recover() error: {e}')
            response.success = False
            response.message = f'Recovery failed: {e}'
        return response

    def _on_parameter_change(self, params):
        for p in params:
            if p.name == 'control_mode' and p.type_ == Parameter.Type.STRING:
                self._activate_controller(p.value)
        return SetParametersResult(successful=True)

    def destroy_node(self):
        if self._active_controller is not None:
            self._active_controller.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocomotionNode()
    # MultiThreadedExecutor so cb_slow (recover/mode-switch with sleeps) runs
    # on a separate thread from cb_realtime (control loop / emergency_stop).
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()