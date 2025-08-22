import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, Point, PoseStamped
from tf.transformations import quaternion_from_euler
from habitat.core.simulator import Observations
import numpy as np
from copy import deepcopy


class ROSPublisher:
    def __init__(self):
        # Create ROS publishers
        self.depth_pub = rospy.Publisher("/habitat/camera_depth", Image, queue_size=10)
        self.rgb_pub = rospy.Publisher("/habitat/camera_rgb", Image, queue_size=10)
        self.odom_pub = rospy.Publisher("/habitat/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher("/habitat/sensor_pose", Odometry, queue_size=10)
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
            position=Point(-gps[2], -gps[0], gps[1]),
            orientation=Quaternion(*quaternion_from_euler(0, 0, copy_compass)),
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
            position=Point(-gps[2], -gps[0], gps[1] + 0.88),
            orientation=Quaternion(
                *quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )
            ),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        depth_image = observations["depth"]
        rgb_image = observations["rgb"]
        gps = observations["gps"]
        compass = observations["compass"]
        camera_pitch = observations["camera_pitch"]
        ros_time = rospy.Time.now()
        self.publish_depth(ros_time, depth_image)
        self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
        self.publish_rgb(ros_time, rgb_image)
        self.publish_robot_odom(ros_time, gps, compass)
