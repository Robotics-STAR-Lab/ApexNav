import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, Point
from tf_transformations import quaternion_from_euler
from habitat.core.simulator import Observations
import numpy as np
from copy import deepcopy


# Sensor data QoS - use BEST_EFFORT for compatibility with RViz2 and ROS2 subscribers
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


def _quat_from_array(q):
    """Convert [x, y, z, w] array to Quaternion msg (ROS2 requires kwargs)."""
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


class ROSPublisher(Node):
    def __init__(self, node_name='habitat_ros_publisher'):
        super().__init__(node_name)
        # Create ROS2 publishers with sensor QoS for RViz2 compatibility
        self.depth_pub = self.create_publisher(Image, "/habitat/camera_depth", SENSOR_QOS)
        self.rgb_pub = self.create_publisher(Image, "/habitat/camera_rgb", SENSOR_QOS)
        self.odom_pub = self.create_publisher(Odometry, "/habitat/odom", 10)
        self.pose_pub = self.create_publisher(Odometry, "/habitat/sensor_pose", SENSOR_QOS)
        # Create cv_bridge object
        self.bridge = CvBridge()

    def publish_depth(self, ros_time, depth_image):
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world"
        self.depth_pub.publish(depth_msg)

    def publish_rgb(self, ros_time, rgb_image):
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        rgb_msg.header.stamp = ros_time
        rgb_msg.header.frame_id = "world"
        self.rgb_pub.publish(rgb_msg)

    def publish_robot_odom(self, ros_time, gps, compass):
        copy_compass = deepcopy(compass)
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(x=float(-gps[2]), y=float(-gps[0]), z=float(gps[1])),
            orientation=_quat_from_array(quaternion_from_euler(0, 0, copy_compass)),
        )
        self.odom_pub.publish(odom)

    def publish_camera_odom(self, ros_time, gps, compass, pitch):
        copy_compass = deepcopy(compass)
        copy_pitch = deepcopy(pitch)
        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time
        sensor_pose.header.frame_id = "world"
        sensor_pose.child_frame_id = "base_link"
        sensor_pose.pose.pose = Pose(
            position=Point(x=float(-gps[2]), y=float(-gps[0]), z=float(gps[1] + 0.88)),
            orientation=_quat_from_array(quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        depth_image = observations["depth"]
        rgb_image = observations["rgb"]
        gps = observations["gps"]
        compass = observations["compass"]
        camera_pitch = observations["camera_pitch"]
        ros_time = self.get_clock().now().to_msg()
        self.publish_depth(ros_time, depth_image)
        self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
        self.publish_rgb(ros_time, rgb_image)
        self.publish_robot_odom(ros_time, gps, compass)


class ROSPublisherNonNode:
    """Non-node version for use with external node"""
    def __init__(self, node):
        self.node = node
        # Create ROS2 publishers with sensor QoS for RViz2 compatibility
        self.depth_pub = node.create_publisher(Image, "/habitat/camera_depth", SENSOR_QOS)
        self.rgb_pub = node.create_publisher(Image, "/habitat/camera_rgb", SENSOR_QOS)
        self.odom_pub = node.create_publisher(Odometry, "/habitat/odom", 10)
        self.pose_pub = node.create_publisher(Odometry, "/habitat/sensor_pose", SENSOR_QOS)
        # Create cv_bridge object
        self.bridge = CvBridge()

    def publish_depth(self, ros_time, depth_image):
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world"
        self.depth_pub.publish(depth_msg)

    def publish_rgb(self, ros_time, rgb_image):
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        rgb_msg.header.stamp = ros_time
        rgb_msg.header.frame_id = "world"
        self.rgb_pub.publish(rgb_msg)

    def publish_robot_odom(self, ros_time, gps, compass):
        copy_compass = deepcopy(compass)
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(x=float(-gps[2]), y=float(-gps[0]), z=float(gps[1])),
            orientation=_quat_from_array(quaternion_from_euler(0, 0, copy_compass)),
        )
        self.odom_pub.publish(odom)

    def publish_camera_odom(self, ros_time, gps, compass, pitch):
        copy_compass = deepcopy(compass)
        copy_pitch = deepcopy(pitch)
        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time
        sensor_pose.header.frame_id = "world"
        sensor_pose.child_frame_id = "base_link"
        sensor_pose.pose.pose = Pose(
            position=Point(x=float(-gps[2]), y=float(-gps[0]), z=float(gps[1] + 0.88)),
            orientation=_quat_from_array(quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        depth_image = observations["depth"]
        rgb_image = observations["rgb"]
        gps = observations["gps"]
        compass = observations["compass"]
        camera_pitch = observations["camera_pitch"]
        ros_time = self.node.get_clock().now().to_msg()
        self.publish_depth(ros_time, depth_image)
        self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
        self.publish_rgb(ros_time, rgb_image)
        self.publish_robot_odom(ros_time, gps, compass)
