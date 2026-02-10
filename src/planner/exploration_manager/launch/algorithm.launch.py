"""
ROS2 Launch file for ApexNav algorithm nodes.
Converted from ROS1 algorithm.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import math


def generate_launch_description():
    # Get package directories
    exploration_manager_share = get_package_share_directory('exploration_manager')
    trajectory_manager_share = get_package_share_directory('trajectory_manager')
    lkh_mtsp_solver_share = get_package_share_directory('lkh_mtsp_solver')

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument('map_size_x_', default_value='80.0')
    map_size_y_arg = DeclareLaunchArgument('map_size_y_', default_value='80.0')
    odometry_topic_arg = DeclareLaunchArgument('odometry_topic_', default_value='/habitat/odom')
    sensor_pose_topic_arg = DeclareLaunchArgument('sensor_pose_topic_', default_value='/habitat/sensor_pose')
    depth_topic_arg = DeclareLaunchArgument('depth_topic_', default_value='/habitat/camera_depth')
    cx_arg = DeclareLaunchArgument('cx_', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy_', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx_', default_value='388.1910413097385')
    fy_arg = DeclareLaunchArgument('fy_', default_value='422.0475153598262')
    is_real_world_arg = DeclareLaunchArgument('is_real_world_', default_value='false')

    # Load trajectory planning parameters
    planning_param_file = os.path.join(
        trajectory_manager_share, 'config', 'planning_param.yaml'
    )

    # Calculate perception angles
    left_angle = 30 * math.pi / 180.0
    right_angle = 30 * math.pi / 180.0

    # Main exploration node
    exploration_node = Node(
        package='exploration_manager',
        executable='exploration_node',
        name='exploration_node',
        output='screen',
        parameters=[
            planning_param_file,
            {
                # FSM mode selection
                'is_real_world': LaunchConfiguration('is_real_world_'),

                # Mapping - SDF Map (C++ uses dots)
                'sdf_map.ray_mode': 0,
                'sdf_map.resolution': 0.05,
                'sdf_map.map_size_x': LaunchConfiguration('map_size_x_'),
                'sdf_map.map_size_y': LaunchConfiguration('map_size_y_'),
                'sdf_map.obstacles_inflation': 0.18,
                'sdf_map.local_bound': 5.0,
                'sdf_map.p_hit': 0.90,
                'sdf_map.p_miss': 0.48,
                'sdf_map.p_min': 0.10,
                'sdf_map.p_max': 0.98,
                'sdf_map.p_occ': 0.80,
                'sdf_map.max_ray_length': 4.99,
                'sdf_map.optimistic': False,
                'sdf_map.signed_dist': False,

                # Map ROS
                'map_ros/cx': LaunchConfiguration('cx_'),
                'map_ros/cy': LaunchConfiguration('cy_'),
                'map_ros/fx': LaunchConfiguration('fx_'),
                'map_ros/fy': LaunchConfiguration('fy_'),
                'map_ros/depth_filter_maxdist': 4.99,
                'map_ros/depth_filter_mindist': 0.0,
                'map_ros/depth_filter_margin': 2,
                'map_ros/filter_min_height': 0.28,
                'map_ros/filter_max_height': 1.18,
                'map_ros/k_depth_scaling_factor': 65535.0,
                'map_ros/skip_pixel': 1,
                'map_ros/frame_id': 'world',
                'map_ros/virtual_ground_height': -0.34,

                # Exploration manager
                'exploration/policy': 2,  # 0: distance 1: semantic 2: hybrid 3: TSP
                'exploration/sigma_threshold': 0.015,
                'exploration/max_to_mean_threshold': 1.10,
                'exploration/max_to_mean_percentage': 0.90,
                'exploration/tsp_dir': os.path.join(lkh_mtsp_solver_share, 'resource'),

                # Frontier (C++ uses dots)
                'frontier.cluster_min': 2,
                'frontier.cluster_size_xy': 0.65,
                'frontier.min_view_finish_fraction': 0.2,
                'frontier.min_contain_unknown': 30,

                # Object (C++ uses dots)
                'object.min_observation_num': 2,
                'object.fusion_type': 1,  # 0: no_fusion 1: ours 2: max
                'object.use_observation': True,
                'object.vis_cloud': True,

                # Perception utils (C++ uses dots)
                'perception_utils.left_angle': left_angle,
                'perception_utils.right_angle': right_angle,
                'perception_utils.max_dist': 4.0,
                'perception_utils.vis_dist': 1.0,

                # A* path searching (C++ uses dots)
                'astar.lambda_heu': 1.0,
                'astar.resolution_astar': 0.1,
            }
        ],
        remappings=[
            ('/odom_world', LaunchConfiguration('odometry_topic_')),
            ('/map_ros/pose', LaunchConfiguration('sensor_pose_topic_')),
            ('/map_ros/depth', LaunchConfiguration('depth_topic_')),
        ]
    )

    # TSP solver node
    tsp_solver_node = Node(
        package='lkh_mtsp_solver',
        executable='tsp_node',
        name='tsp_solver',
        output='log',
        parameters=[{
            'exploration/tsp_dir': os.path.join(lkh_mtsp_solver_share, 'resource'),
        }]
    )

    return LaunchDescription([
        # Arguments
        map_size_x_arg,
        map_size_y_arg,
        odometry_topic_arg,
        sensor_pose_topic_arg,
        depth_topic_arg,
        cx_arg,
        cy_arg,
        fx_arg,
        fy_arg,
        is_real_world_arg,
        # Nodes
        exploration_node,
        tsp_solver_node,
    ])
