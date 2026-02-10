"""
ROS2 Launch file for ApexNav exploration system.
Converted from ROS1 exploration.launch
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    exploration_manager_share = get_package_share_directory('exploration_manager')

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument(
        'map_size_x',
        default_value='80.0',
        description='Map size in X direction'
    )

    map_size_y_arg = DeclareLaunchArgument(
        'map_size_y',
        default_value='80.0',
        description='Map size in Y direction'
    )

    is_real_world_arg = DeclareLaunchArgument(
        'is_real_world',
        default_value='false',
        description='Real-world mode flag'
    )

    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/habitat/odom',
        description='Topic of odometry (VIO or LIO)'
    )

    sensor_pose_topic_arg = DeclareLaunchArgument(
        'sensor_pose_topic',
        default_value='/habitat/sensor_pose',
        description='Transform of camera frame in world frame'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/habitat/camera_depth',
        description='Depth image topic (640x480 by default)'
    )

    cx_arg = DeclareLaunchArgument('cx', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx', default_value='388.1910413097385')
    fy_arg = DeclareLaunchArgument('fy', default_value='422.0475153598262')

    # Load YAML config for simulation mode
    config_file = os.path.join(
        os.path.dirname(exploration_manager_share),
        '..', '..', '..', 'config', 'habitat_eval_hm3dv2.yaml'
    )

    # Include algorithm launch file
    algorithm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            exploration_manager_share, '/launch/algorithm.launch.py'
        ]),
        launch_arguments={
            'is_real_world_': LaunchConfiguration('is_real_world'),
            'map_size_x_': LaunchConfiguration('map_size_x'),
            'map_size_y_': LaunchConfiguration('map_size_y'),
            'odometry_topic_': LaunchConfiguration('odom_topic'),
            'sensor_pose_topic_': LaunchConfiguration('sensor_pose_topic'),
            'depth_topic_': LaunchConfiguration('depth_topic'),
            'cx_': LaunchConfiguration('cx'),
            'cy_': LaunchConfiguration('cy'),
            'fx_': LaunchConfiguration('fx'),
            'fy_': LaunchConfiguration('fy'),
        }.items()
    )

    return LaunchDescription([
        # Launch arguments
        map_size_x_arg,
        map_size_y_arg,
        is_real_world_arg,
        odom_topic_arg,
        sensor_pose_topic_arg,
        depth_topic_arg,
        cx_arg,
        cy_arg,
        fx_arg,
        fy_arg,
        # Include algorithm
        algorithm_launch,
    ])
