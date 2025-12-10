#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import numpy as np
import time
from cv_bridge import CvBridge
import message_filters
import tf.transformations as tft

import hydra
from omegaconf import DictConfig

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from plan_env.msg import MultipleMasksWithConfidence

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vlm.utils.get_object_utils import get_object
from vlm.utils.get_itm_message import get_itm_message_cosine
from llm.answer_reader.answer_reader import read_answer
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)


def inverse_habitat_publisher_transform(sensor_pose_msg):
    """
    Inverse transform to recover original Habitat gps and compass from ROS sensor_pose.
    """
    pos = sensor_pose_msg.pose.pose.position
    orn = sensor_pose_msg.pose.pose.orientation

    # Invert position transform:
    gps = np.array([-pos.y, pos.z - 0.88, -pos.x], dtype=np.float32)

    # Invert orientation transform:
    euler = tft.euler_from_quaternion([orn.x, orn.y, orn.z, orn.w])
    compass_scalar = euler[2] + np.pi / 2.0
    # Habitat compass is a single-element array
    compass = np.array([compass_scalar], dtype=np.float32)

    return gps, compass


class RealWorldNode:
    def __init__(self, cfg):
        self.config = cfg

        rospy.init_node("real_world_node", anonymous=False)

        self.bridge = CvBridge()

        # Configure subscribers
        self.rgb_sub_ = message_filters.Subscriber("/habitat/camera_rgb", Image)
        self.depth_sub_ = message_filters.Subscriber("/habitat/camera_depth", Image)
        self.sensor_pose_sub_ = message_filters.Subscriber(
            "/habitat/sensor_pose", Odometry
        )

        rospy.Subscriber("/habitat/odom", Odometry, self.odom_callback, queue_size=10)

        # Configure publishers
        self.confidence_threshold_pub_ = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=10
        )
        self.itm_score_pub_ = rospy.Publisher(
            "/blip2/cosine_score", Float64, queue_size=10
        )
        self.cld_with_score_pub_ = rospy.Publisher(
            "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
        )
        self.detect_img_pub_ = rospy.Publisher(
            "/detector/detect_img", Image, queue_size=10
        )

        # Initialize detector
        # Synchronize RGB, depth and sensor_pose topics
        self.sync_detect = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5,
            slop=0.01,
        )
        self.sync_detect.registerCallback(self.sync_detect_callback)

        # Initialize value module
        # (uses synchronized RGB/depth/sensor_pose messages)
        self.sync_value = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5,
            slop=0.01,
        )
        self.sync_value.registerCallback(self.sync_value_callback)

        # Initialize odometry handling
        self.robot_odom = None
        self.T_base_camera = None
        self.odom_stamp = None
        # Processing flags: ensure we don't start a new processing run
        # until the previous one finished (rate adapts to available compute)
        self.processing_detect = False
        self.processing_value = False

        # LLM config (used when label is provided via topic)
        llm_cfg = self.config.llm
        self.llm_answer_path = llm_cfg.llm_answer_path
        self.llm_response_path = llm_cfg.llm_response_path
        self.llm_client = llm_cfg.llm_client.llm_client

        # Label will be provided via ROS topic `/detector/label` (std_msgs/String)
        # Initialize empty/defaults; actual values will be set in `label_callback`.
        self.label = None
        self.llm_answer = []
        self.room = None
        self.fusion_score = 0.0

        # Subscribe to label topic (published by `habitat_trajectory_test.py`)
        rospy.Subscriber("/detector/label", String, self.label_callback, queue_size=1)

        rospy.Timer(rospy.Duration(1.0), self.publish_confidence_threshold)

    def sync_detect_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        # If a detect run is already in progress, skip this invocation.
        if self.processing_detect:
            return
        self.processing_detect = True
        try:
            # rospy.loginfo("detect: Received synchronized RGB and depth images")
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                # If timestamps differ significantly, skip this pair
                # and allow the next synchronized callback to run.
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            transform_depth_img = depth_img.astype(np.float32)
            depth_cv = np.expand_dims(transform_depth_img, axis=-1)

            cld_with_score_msg = MultipleMasksWithConfidence()
            cld_with_score_msg.point_clouds = []
            cld_with_score_msg.confidence_scores = []
            cld_with_score_msg.label_indices = []
            rospy.loginfo("detect: label: %s", self.label)
            # rospy.loginfo("detect: room: %s", self.room)

            # If label not yet received, skip detection until available
            if self.label is None:
                rospy.logwarn_throttle(5.0, "Waiting for target label on /detector/label")
                return

            detect_img, score_list, object_masks_list, label_list = get_object(
                self.label, rgb_cv, self.config.detector, self.llm_answer
            )

            # Use inverse transform to recover original Habitat observations format
            gps, compass = inverse_habitat_publisher_transform(sensor_pose_msg)

            observations = {
                "depth": depth_cv,
                "gps": gps,
                "compass": compass,  # Already a numpy array from inverse function
            }

            obj_point_cloud_list = get_object_point_cloud(
                self.config, observations, object_masks_list
            )
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            # Publish the detection image for visualization
            self.detect_img_pub_.publish(
                self.bridge.cv2_to_imgmsg(detect_img, encoding="rgb8")
            )

            # Also publish the detected object clouds with scores so other nodes / RViz can use them
            self.cld_with_score_pub_.publish(cld_with_score_msg)
        except Exception as e:
            rospy.logerr("detect: Error in synchronized processing: %s", e)
        finally:
            # mark processing complete so next invocation can proceed
            self.processing_detect = False

    def sync_value_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        # If a value run is already in progress, skip this invocation.
        if self.processing_value:
            return
        self.processing_value = True
        try:
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                # If timestamps differ significantly, skip this pair
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            # rospy.loginfo("value: room: %s", self.room)

            cosine = get_itm_message_cosine(rgb_cv, self.label, self.room)
            rospy.loginfo("value: Computed cosine score: %.3f", cosine)
            itm_score_msg = Float64()
            itm_score_msg.data = cosine
            self.itm_score_pub_.publish(itm_score_msg)
        except Exception as e:
            rospy.logerr("value: Error in synchronized processing: %s", e)
        finally:
            self.processing_value = False

    def label_callback(self, msg):
        """Handle incoming label messages and update LLM answers if configured."""
        try:
            new_label = str(msg.data)
            if new_label == self.label:
                return
            self.label = new_label
            rospy.loginfo("Received target label: %s", self.label)
            # If LLM is configured, fetch LLM answer for the new label
            try:
                self.llm_answer, self.room, self.fusion_score = read_answer(
                    self.llm_answer_path, self.llm_response_path, self.label, self.llm_client
                )
            except Exception:
                # Non-fatal: proceed without LLM answer
                self.llm_answer = []
                self.room = None
                self.fusion_score = 0.0
        except Exception as e:
            rospy.logerr("label_callback: Error processing label message: %s", e)

    def odom_callback(self, msg):
        try:
            self.robot_odom = msg
            self.odom_stamp = msg.header.stamp
            if self.odom_stamp is not None:
                # self.publish_sensor_pose()
                self.odom_stamp = None
            # rospy.loginfo("odom: Received Odometry")
        except Exception as e:
            rospy.logerr("odom: Error processing Odometry: %s", e)

    def publish_confidence_threshold(self, event):
        confidence_threshold_msg = Float64()
        confidence_threshold_msg.data = 0.5
        self.confidence_threshold_pub_.publish(confidence_threshold_msg)

    def run(self):
        rospy.loginfo("RealWorldNode running. Waiting for sensor messages...")
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test")
def main(cfg: DictConfig):
    node = RealWorldNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
