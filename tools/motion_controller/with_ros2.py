from __future__ import annotations

import time
from functools import partial

import rclpy
import yaml
from absl import app, flags
from geometry_msgs.msg import Twist, TwistStamped
from motion_controller import MotionController
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool as rosbool

flags.DEFINE_string('profile_path', './default_profile.yaml', 'default profile path')
flags.DEFINE_enum('control_type', 'control', ['control', 'curve', 'immediate'],
                  "You should choose among ['control', 'curve', 'immediate']")
FLAGS = flags.FLAGS

FPS: int = 50
LOG_COUNT_THRESHOLD: int = 10


def _build_condition(profile: dict) -> dict:
    """Convert a YAML profile section into a MotionController condition dict."""
    return {
        'max_velocity_x':    profile['vx_max'],
        'min_velocity_x':    profile['vx_min'],
        'max_velocity_y':    profile['vy_max'],
        'min_velocity_y':   -profile['vy_max'],
        'max_velocity_th':   profile['vth_max'],
        'min_velocity_th':  -profile['vth_max'],
        'max_speed_up_x':    profile['speed_x_up_max'],
        'max_speed_down_x':  profile['speed_x_down_max'],
        'max_speed_up_y':    profile['speed_y_up_max'],
        'max_speed_down_y':  profile['speed_y_down_max'],
        'max_speed_up_th':   profile['speed_th_up_max'],
        'max_speed_down_th': profile['speed_th_down_max'],
    }


class MotionControllerRos2(Node):

    def __init__(
            self,
            core: MotionController,
            # Sub topics must be defined in the order expected by MotionController.process():
            # robot_vel, cmd_nav, cmd_rmt
            sub_topics: list[dict] = [
                {'name': '/cmd_vel',        'type': TwistStamped},
                {'name': '/cmd_nav',        'type': TwistStamped},
                {'name': '/cmd_vel_remote', 'type': TwistStamped},
            ],
            pub_topic: str = '/cmd_ctr'):
        super().__init__('MotionController')
        self.core = core
        self.data = [None] * MotionController.process_args
        self.timestamps = [0.0] * MotionController.process_args

        with open(FLAGS.profile_path) as f:
            profile = yaml.safe_load(f)
        self.core.service_condition = _build_condition(profile['nav'])
        self.core.remote_condition  = _build_condition(profile['remote'])

        for idx, topic in enumerate(sub_topics):
            topicname = topic['name']
            msgtype = topic['type']
            if idx >= MotionController.process_args:
                self.get_logger().error(
                    f'sub topic {topicname}: index {idx} out of range. '
                    f'Max value is {MotionController.process_args}'
                )
                continue
            self.data[idx] = False if msgtype is rosbool else [0.0, 0.0, 0.0]
            self.create_subscription(
                msgtype, topicname,
                partial(self.sub_cb, idx=idx, topicname=topicname), 1,
            )

        self.cmd_ctr_pub = self.create_publisher(TwistStamped, pub_topic, 1)
        self.core.control_t = 1.0 / FPS
        self._log_count = 0
        self._log_time = time.time()
        self.create_timer(1.0 / FPS, self.timer_cb)

    def get_ts(self, ros2header) -> float:
        return ros2header.stamp.sec + ros2header.stamp.nanosec * 1e-9

    def sub_cb(self, msg, idx: int = 0, topicname: str = '/topicname') -> None:
        if isinstance(msg, TwistStamped):
            # Remote uses system time because its header timestamp can drift
            if topicname == '/cmd_vel_remote':
                self.timestamps[idx] = time.time()
            else:
                self.timestamps[idx] = self.get_ts(msg.header)
            self.data[idx] = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)
        elif isinstance(msg, Twist):
            self.timestamps[idx] = time.time()
            self.data[idx] = (msg.linear.x, msg.linear.y, msg.angular.z)
        elif isinstance(msg, Odometry):
            self.timestamps[idx] = time.time()
            self.data[idx] = (msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                              msg.twist.twist.angular.z)
        elif isinstance(msg, rosbool):
            self.data[idx] = msg.data
            if msg.data:
                self.timestamps[idx] = time.time()

    def timer_cb(self) -> None:
        (vx, vy, vth), info = self.core.process(self.data, self.timestamps)
        self.cmd_ctr_pub.publish(self._make_twist_stamped(vx, vy, vth))
        self._log_count += 1
        if self._log_count >= LOG_COUNT_THRESHOLD:
            elapsed = time.time() - self._log_time
            self.get_logger().info(
                f'FPS: {self._log_count / elapsed:.2f}, {info}: {vx:.2f},{vy:.2f},{vth:.2f}'
            )
            self._log_count = 0
            self._log_time = time.time()

    def _make_twist_stamped(self, vx: float, vy: float, vth: float) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.angular.z = float(vth)
        return msg


def main(args=None):
    mc = MotionController(control_type=FLAGS.control_type, service_condition=None)
    rclpy.init()
    mc_ros2 = MotionControllerRos2(mc)
    try:
        rclpy.spin(mc_ros2)
    except KeyboardInterrupt:
        pass
    mc_ros2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    app.run(main)
