import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('go2_locomotion')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to params YAML file',
        ),
        DeclareLaunchArgument(
            'control_mode',
            default_value='sdk_velocity',
            description='Control mode: sdk_velocity | nn_policy',
        ),
        DeclareLaunchArgument(
            'network_interface',
            default_value='eth0',
            description='Network interface connected to Go2 (e.g. eth0)',
        ),
        Node(
            package='go2_locomotion',
            executable='locomotion_node',
            name='locomotion_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'control_mode':      LaunchConfiguration('control_mode'),
                    'network_interface': LaunchConfiguration('network_interface'),
                },
            ],
        ),
    ])
