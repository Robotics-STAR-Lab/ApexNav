"""
ROS2 Launch file for RViz2 visualization.
Converted from ROS1 rviz.launch
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    exploration_manager_share = get_package_share_directory('exploration_manager')

    # RViz2 config file
    rviz_config = os.path.join(
        exploration_manager_share, 'config', 'ApexNav.rviz'
    )

    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_visualisation',
        output='log',
        arguments=['-d', rviz_config]
    )

    # Static transform publisher (world -> navigation)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_navigation',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'navigation']
    )

    return LaunchDescription([
        rviz_node,
        static_tf_node,
    ])
