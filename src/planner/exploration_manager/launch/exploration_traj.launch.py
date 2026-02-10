"""
ROS2 Launch file for ApexNav exploration system (trajectory mode with MPC control).
Converted from ROS1 exploration_traj.launch
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directories
    exploration_manager_share = get_package_share_directory('exploration_manager')
    trajectory_manager_share = get_package_share_directory('trajectory_manager')

    # Control parameter YAML
    control_param_file = os.path.join(
        trajectory_manager_share, 'config', 'control_param.yaml'
    )

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument(
        'map_size_x', default_value='80.0',
        description='Map size in X direction'
    )
    map_size_y_arg = DeclareLaunchArgument(
        'map_size_y', default_value='80.0',
        description='Map size in Y direction'
    )
    is_real_world_arg = DeclareLaunchArgument(
        'is_real_world', default_value='true',
        description='Real-world mode flag'
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic', default_value='/habitat/odom',
        description='Topic of odometry (VIO or LIO)'
    )
    sensor_pose_topic_arg = DeclareLaunchArgument(
        'sensor_pose_topic', default_value='/habitat/sensor_pose',
        description='Transform of camera frame in world frame'
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/habitat/camera_depth',
        description='Depth image topic (640x480 by default)'
    )
    cx_arg = DeclareLaunchArgument('cx', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx', default_value='388.1910413097385')
    fy_arg = DeclareLaunchArgument('fy', default_value='422.0475153598262')

    # Include algorithm_traj launch file
    algorithm_traj_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            exploration_manager_share, '/launch/algorithm_traj.launch.py'
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

    # Trajectory server for real robot control
    traj_server_node = Node(
        package='trajectory_manager',
        executable='traj_server',
        name='traj_server_node',
        output='screen',
        parameters=[
            control_param_file,
            {
                'need_init': False,
                'max_correction_vel': 1.0,
                'max_correction_omega': 1.57,
            }
        ],
        remappings=[
            ('odometry', LaunchConfiguration('odom_topic')),
            ('trajectory', '/planning/trajectory'),
            ('cmd_vel', '/cmd_vel'),
        ]
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
        algorithm_traj_launch,
        # Trajectory server
        traj_server_node,
    ])
