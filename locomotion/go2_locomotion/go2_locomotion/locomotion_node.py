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

        self.declare_parameter('control_mode',      "nn_policy")#'sdk_velocity')
        self.declare_parameter('network_interface',     'eth0')
        self.declare_parameter('control_frequency',     50.0)
        self.declare_parameter('watchdog_timeout',      0.5)
        self.declare_parameter('nn_policy.model_path',  '/home/unitree/ros2_ws/src/loconav-stack/locomotion/go2_locomotion/tools/policy_first.onnx')
        self.declare_parameter('nn_policy.obs_dim',     45)
        self.declare_parameter('nn_policy.action_scale', 0.25)
        self.declare_parameter('nn_policy.action_clip',  6.0)
        # 일어서기/hold(start·recover·stop) 게인 — FixStand 계열
        self.declare_parameter('nn_policy.kp', [60.0, 80.0, 80.0, 60.0, 80.0, 80.0,
                                                 60.0, 80.0, 80.0, 60.0, 80.0, 80.0])
        self.declare_parameter('nn_policy.kd', [5.0, 4.0, 4.0, 5.0, 4.0, 4.0,
                                                 5.0, 4.0, 4.0, 5.0, 4.0, 4.0])
        # RL 보행 게인 — 학습값(25/0.5), step()에서만 사용
        self.declare_parameter('nn_policy.policy_kp', [25.0] * 12)
        self.declare_parameter('nn_policy.policy_kd', [0.5] * 12)
        self.declare_parameter('nn_policy.log_path', '')
        # Policy backend. "auto" picks by the model file: *.onnx -> OnnxPolicy; *.pt -> the rsl_rl
        # torch policy (MoE if the checkpoint has actor.moe.*, else the Hist-MLP). Set explicitly to
        # "onnx"/"moe"/"mlp" to override. Torch backends need torch + rsl_rl on the robot compute.
        self.declare_parameter('nn_policy.policy_type', 'auto')
        self.declare_parameter('nn_policy.rsl_rl_path', '/home/unitree/ros2_ws/src/rsl_rl')
        # MoE only: pin every step to one expert (>=0), or -1 for normal gate routing. Lets you drive
        # a single terrain expert on the real robot to see its gait in isolation (like the demo).
        self.declare_parameter('nn_policy.force_expert', -1)
        # MoE routing persistence at deployment (anti mode-chatter); applies to torch MoE and onnx_moe.
        self.declare_parameter('nn_policy.route_hysteresis', 1.0)
        self.declare_parameter('nn_policy.route_min_dwell', 10)

        network_interface = self.get_parameter('network_interface').value
        freq              = self.get_parameter('control_frequency').value
        self._watchdog_timeout = self.get_parameter('watchdog_timeout').value
        initial_mode      = self.get_parameter('control_mode').value

        # ChannelFactory is a singleton — init once here, not inside each controller
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, "eth0")#network_interface)

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

    def _build_policy(self, model_path: str):
        """Select the policy backend by ``nn_policy.policy_type`` (or auto-detect from the file).

        onnx     : OnnxPolicy -- legacy single-frame (45-d) MLP/recurrent export. No torch.
        onnx_moe : OnnxMoEPolicy -- the MoE/Hist-MLP export from convert_moe_to_onnx.py (225-d
                   history + numpy routing/persistence). No torch -- THIS is the Go2 runtime.
        moe/mlp  : rsl_rl ActorCriticMoE / ActorCriticVel loaded via torch (dev/PC only).
        auto     : *.onnx -> onnx_moe if it has an 'obs_history' input else onnx; *.pt -> moe if the
                   checkpoint has actor.moe.* else mlp.
        """
        ptype = self.get_parameter('nn_policy.policy_type').value
        rsl_rl_path = self.get_parameter('nn_policy.rsl_rl_path').value
        force_expert = self.get_parameter('nn_policy.force_expert').value
        hyst = self.get_parameter('nn_policy.route_hysteresis').value
        dwell = self.get_parameter('nn_policy.route_min_dwell').value
        force_kw = {'force_expert': force_expert} if force_expert is not None and force_expert >= 0 else {}

        if ptype == 'auto':
            if model_path.endswith('.onnx'):
                import onnxruntime as ort  # our MoE/MLP exports carry an 'obs_history' input
                names = [i.name for i in ort.InferenceSession(
                    model_path, providers=['CPUExecutionProvider']).get_inputs()]
                ptype = 'onnx_moe' if 'obs_history' in names else 'onnx'
            else:
                import torch  # only needed for the torch backends
                sd = torch.load(model_path, map_location='cpu')
                sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
                ptype = 'moe' if any(k.startswith('actor.moe.') for k in sd) else 'mlp'
            self.get_logger().info(f'nn_policy backend auto-detected: {ptype}')

        if ptype == 'onnx':
            from go2_locomotion.policy.onnx_policy import OnnxPolicy
            return OnnxPolicy(model_path)
        if ptype == 'onnx_moe':
            from go2_locomotion.policy.onnx_moe_policy import OnnxMoEPolicy
            return OnnxMoEPolicy(model_path, route_hysteresis=hyst, route_min_dwell=dwell, **force_kw)
        if ptype == 'mlp':
            from go2_locomotion.policy.mlp_policy import MlpPolicy
            return MlpPolicy(model_path, rsl_rl_path=rsl_rl_path)
        if ptype == 'moe':
            from go2_locomotion.policy.moe_policy import MoEPolicy
            return MoEPolicy(model_path, rsl_rl_path=rsl_rl_path, **force_kw)
        raise ValueError(f"unknown nn_policy.policy_type {ptype!r} (auto/onnx/onnx_moe/moe/mlp)")

    def _build_nn_controller(self) -> NNPolicyController:
        model_path   = self.get_parameter('nn_policy.model_path').value
        obs_dim      = self.get_parameter('nn_policy.obs_dim').value
        action_scale = self.get_parameter('nn_policy.action_scale').value
        action_clip  = self.get_parameter('nn_policy.action_clip').value
        kp           = self.get_parameter('nn_policy.kp').value
        kd           = self.get_parameter('nn_policy.kd').value
        policy_kp    = self.get_parameter('nn_policy.policy_kp').value
        policy_kd    = self.get_parameter('nn_policy.policy_kd').value
        log_path     = self.get_parameter('nn_policy.log_path').value

        policy = None
        if model_path:
            try:
                policy = self._build_policy(model_path)
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
            action_clip=action_clip,
            kp=list(kp),
            kd=list(kd),
            policy_kp=list(policy_kp),
            policy_kd=list(policy_kd),
            log_path=log_path,
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
