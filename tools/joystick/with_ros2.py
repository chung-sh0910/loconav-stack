from typing import List, Optional

import numpy as np
import rclpy
import yaml
from absl import app, flags
from geometry_msgs.msg import Pose, PoseArray, TwistStamped
from numpy.typing import NDArray
from rclpy.node import Node
from std_srvs.srv import Trigger

from core import PS4Controller

flags.DEFINE_string('profile_path', './default_profile.yaml',
                    'default profile path')
flags.DEFINE_string('base_link', 'real_base_link', 'default base frame id')
FLAGS = flags.FLAGS


def numpy2pose(
    position: Optional[NDArray] = None,
    orientation: Optional[NDArray] = None,
) -> Pose:
    msg = Pose()
    if position is not None:
        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
    if orientation is not None:
        msg.orientation.x = orientation[0]
        msg.orientation.y = orientation[1]
        msg.orientation.z = orientation[2]
        msg.orientation.w = orientation[3]
    return msg


class RemoteController(Node):

    def __init__(self, timer_period: float = 0.02) -> None:
        name = "remote_controller_node"
        super().__init__(name)
        print("Remote controller Initializing")
        self.name = name
        self.timer_period = timer_period
        self.ps4_controller = PS4Controller()

        self.remote_condition = {}
        self.default_remote = yaml.safe_load(open(
            FLAGS.profile_path))['remote']
        self.initialize_remote_condition()

        self.run_emergency_brake = False
        self._prev_need_to_stop = False

        self._estop_client = self.create_client(Trigger, '/emergency_stop')
        self._recover_client = self.create_client(Trigger, '/recover')

        self.publish_topic = "/cmd_vel_remote"
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.publish_topic,
            1,
        )

        self.waypoint_topic = "/waypoints_local"
        self.waypoint_pub = self.create_publisher(
            PoseArray,
            self.waypoint_topic,
            1,
        )

        self.ps4_listening_timer = self.create_timer(
            self.timer_period, self._ps4_listening_callback)

        waypoint_pub_period = 0.1
        self.ps4_listening_timer = self.create_timer(
            waypoint_pub_period, self._waypoint_publishing_timer)

    def _ps4_listening_callback(self) -> None:

        self.ps4_controller.listen()
        need_to_shutdown, need_to_stop = (
            self.ps4_controller.check_emergency_condition())
        self.ps4_controller.check_pub_waypoint_command()

        if need_to_shutdown:
            self._call_service(self._estop_client, 'emergency_stop')
            self._publish_stop_msg()
            raise ValueError(
                "Emergency : System will stop immediately for safety!")

        if need_to_stop:
            self._publish_stop_msg()
            if not self._prev_need_to_stop:
                self._call_service(self._estop_client, 'emergency_stop')
            self._prev_need_to_stop = True
            return

        if self._prev_need_to_stop:
            self._call_service(self._recover_client, 'recover')
        self._prev_need_to_stop = False

        is_valid, linear_action_x, linear_action_y, angular_action_w = (
            self.ps4_controller.get_robot_action())

        if is_valid:
            linear_velocity, angular_velocity = self._action_to_velocity(
                linear_action_x, linear_action_y, angular_action_w)
            robot_command_msg = self._to_twist_stamped(linear_velocity,
                                                       angular_velocity)

            self.cmd_pub.publish(robot_command_msg)

    def _action_to_velocity(self, linear_action_x: float,
                            linear_action_y: float,
                            angular_action_w: float) -> None:

        linear_velocity_vx = self._scale_to_velocity(linear_action_x,
                                                     target_axis="x")
        linear_velocity_vy = self._scale_to_velocity(linear_action_y,
                                                     target_axis="y")
        angular_velocity_w = self._scale_to_velocity(angular_action_w,
                                                     target_axis="th")

        linear_velocity = [linear_velocity_vx, linear_velocity_vy, 0.]
        angular_velocity = [0., 0., angular_velocity_w]
        return linear_velocity, angular_velocity

    def _scale_to_velocity(self, command_action: float,
                           target_axis: str) -> float:
        available_axis = ['x', 'y', 'th']
        if target_axis not in available_axis:
            raise ValueError(f"{target_axis} should be among {available_axis}")
        if command_action >= 0:
            return self.remote_condition[
                f'max_velocity_{target_axis}'] * command_action
        else:
            return -self.remote_condition[
                f'min_velocity_{target_axis}'] * command_action

    def _call_service(self, client, name: str) -> None:
        if not client.service_is_ready():
            self.get_logger().warn(f'/{name} service not available, skipping')
            return
        client.call_async(Trigger.Request())
        self.get_logger().warn(f'/{name} called')

    def _publish_stop_msg(self) -> None:
        stop_msg = TwistStamped()
        stop_msg.header.frame_id = FLAGS.base_link
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop_msg)

    def _to_twist_stamped(self, linear_velocity: List,
                          angular_velocity: List) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = FLAGS.base_link
        msg.header.stamp = self.get_clock().now().to_msg()

        if linear_velocity is not None:
            msg.twist.linear.x = linear_velocity[0]
            msg.twist.linear.y = linear_velocity[1]
            msg.twist.linear.z = linear_velocity[2]
        if angular_velocity is not None:
            msg.twist.angular.x = angular_velocity[0]
            msg.twist.angular.y = angular_velocity[1]
            msg.twist.angular.z = angular_velocity[2]

        return msg

    def _waypoint_publishing_timer(self) -> None:
        pub_go_waypoint, pub_stop_waypoint = self.ps4_controller.check_pub_waypoint_command()
        if not pub_go_waypoint and not pub_stop_waypoint:
            return

        need_to_pub_waypoint = False
        
        if pub_stop_waypoint:
            target_waypoints = self.get_stop_waypoint_msg()
            self.ps4_controller.set_stop_waypoint_condition(need_to_pub_waypoint)
            print("Publish stop waypoint")
        elif pub_go_waypoint:
            target_waypoints = self.get_go_waypoint_msg()
            self.ps4_controller.set_go_waypoint_condition(need_to_pub_waypoint)
            print("Publish go waypoint")
            
        self.waypoint_pub.publish(target_waypoints)

    def get_stop_waypoint_msg(self):
        target_waypoints = PoseArray()
        target_waypoints.header.frame_id = FLAGS.base_link
        target_waypoints.header.stamp = self.get_clock().now().to_msg()
        origin_pose = np.array([0., 0., 0.])
        target_waypoints.poses.append(numpy2pose(origin_pose))
        return target_waypoints

    def get_go_waypoint_msg(self):
        target_waypoints_dist_x = 10.
        target_waypoints = PoseArray()
        target_waypoints.header.frame_id = FLAGS.base_link
        target_waypoints.header.stamp = self.get_clock().now().to_msg()

        origin_pose = np.array([0., 0., 0.])
        target_waypoints.poses.append(numpy2pose(origin_pose))
        target_pose = np.array([target_waypoints_dist_x, 0., 0.])
        target_waypoints.poses.append(numpy2pose(target_pose))
        return target_waypoints

    def initialize_remote_condition(self) -> None:
        self.remote_condition = {
            'max_velocity_x': self.default_remote['vx_max'],
            'min_velocity_x': self.default_remote['vx_min'],
            'max_velocity_y': self.default_remote['vy_max'],
            'min_velocity_y': -self.default_remote['vy_max'],
            'max_velocity_th': self.default_remote['vth_max'],
            'min_velocity_th': -self.default_remote['vth_max'],
            'max_speed_up_x': self.default_remote['speed_x_up_max'],
            'max_speed_down_x': self.default_remote['speed_x_down_max'],
            'max_speed_up_y': self.default_remote['speed_y_up_max'],
            'max_speed_down_y': self.default_remote['speed_y_down_max'],
            'max_speed_up_th': self.default_remote['speed_th_up_max'],
            'max_speed_down_th': self.default_remote['speed_th_down_max']
        }


def main(argv) -> None:
    del argv
    rclpy.init()
    remote_controller = RemoteController()
    try:
        rclpy.spin(remote_controller)
    finally:
        remote_controller.ps4_controller.data_parser.stop()
        remote_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    app.run(main)
