#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from cv_bridge import CvBridge
import message_filters
from tf_transformations import euler_from_quaternion

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
    euler = euler_from_quaternion([orn.x, orn.y, orn.z, orn.w])
    compass_scalar = euler[2] + np.pi / 2.0
    # Habitat compass is a single-element array
    compass = np.array([compass_scalar], dtype=np.float32)

    return gps, compass


class RealWorldNode(Node):
    def __init__(self, cfg):
        super().__init__('real_world_node')
        self.config = cfg

        self.bridge = CvBridge()

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Configure subscribers using message_filters with BEST_EFFORT QoS
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.rgb_sub_ = message_filters.Subscriber(self, Image, "/habitat/camera_rgb", qos_profile=best_effort_qos)
        self.depth_sub_ = message_filters.Subscriber(self, Image, "/habitat/camera_depth", qos_profile=best_effort_qos)
        self.sensor_pose_sub_ = message_filters.Subscriber(self, Odometry, "/habitat/sensor_pose", qos_profile=best_effort_qos)

        self.create_subscription(Odometry, "/habitat/odom", self.odom_callback, qos)

        # Configure publishers
        self.confidence_threshold_pub_ = self.create_publisher(
            Float64, "/detector/confidence_threshold", qos)
        self.itm_score_pub_ = self.create_publisher(
            Float64, "/blip2/cosine_score", qos)
        self.cld_with_score_pub_ = self.create_publisher(
            MultipleMasksWithConfidence, "/detector/clouds_with_scores", qos)
        self.detect_img_pub_ = self.create_publisher(
            Image, "/detector/detect_img", qos)

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
        self.create_subscription(String, "/detector/label", self.label_callback, 1)

        self.create_timer(1.0, self.publish_confidence_threshold)

    def sync_detect_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        # If a detect run is already in progress, skip this invocation.
        if self.processing_detect:
            return
        self.processing_detect = True
        try:
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp.sec + stamp.nanosec * 1e-9) -
                          (sensor_pose_msg.header.stamp.sec + sensor_pose_msg.header.stamp.nanosec * 1e-9))
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
            self.get_logger().info(f"detect: label: {self.label}")

            # If label not yet received, skip detection until available
            if self.label is None:
                self.get_logger().warn("Waiting for target label on /detector/label")
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
                self.config, observations, object_masks_list, self
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
            self.get_logger().error(f"detect: Error in synchronized processing: {e}")
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
            time_diff = abs((stamp.sec + stamp.nanosec * 1e-9) -
                          (sensor_pose_msg.header.stamp.sec + sensor_pose_msg.header.stamp.nanosec * 1e-9))
            if time_diff > 0.1:
                # If timestamps differ significantly, skip this pair
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

            cosine = get_itm_message_cosine(rgb_cv, self.label, self.room)
            self.get_logger().info(f"value: Computed cosine score: {cosine:.3f}")
            itm_score_msg = Float64()
            itm_score_msg.data = float(cosine)
            self.itm_score_pub_.publish(itm_score_msg)
        except Exception as e:
            self.get_logger().error(f"value: Error in synchronized processing: {e}")
        finally:
            self.processing_value = False

    def label_callback(self, msg):
        """Handle incoming label messages and update LLM answers if configured."""
        try:
            new_label = str(msg.data)
            if new_label == self.label:
                return
            self.label = new_label
            self.get_logger().info(f"Received target label: {self.label}")
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
            self.get_logger().error(f"label_callback: Error processing label message: {e}")

    def odom_callback(self, msg):
        try:
            self.robot_odom = msg
            self.odom_stamp = msg.header.stamp
            if self.odom_stamp is not None:
                self.odom_stamp = None
        except Exception as e:
            self.get_logger().error(f"odom: Error processing Odometry: {e}")

    def publish_confidence_threshold(self):
        confidence_threshold_msg = Float64()
        confidence_threshold_msg.data = 0.5
        self.confidence_threshold_pub_.publish(confidence_threshold_msg)

    def run(self):
        self.get_logger().info("RealWorldNode running. Waiting for sensor messages...")
        rclpy.spin(self)


@hydra.main(version_base=None, config_path="config", config_name="real_world_test")
def main(cfg: DictConfig):
    rclpy.init()
    try:
        node = RealWorldNode(cfg)
        node.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
