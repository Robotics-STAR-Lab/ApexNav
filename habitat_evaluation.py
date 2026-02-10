"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros.habitat_publisher import ROSPublisherNonNode
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


class HabitatEvalNode(Node):
    def __init__(self):
        super().__init__('habitat_eval_node')

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State variables
        self.global_action = None
        self.ros_state = ROS_STATE.INIT
        self.final_state = 0
        self.expl_result = 0
        self.msg_observations = None
        self.fusion_threshold = 0.0
        self.pub_timer = None

        # Publishers
        self.obj_point_cloud_pub = self.create_publisher(
            PointCloud2, "habitat/object_point_cloud", qos)
        self.state_pub = self.create_publisher(Int32, "/habitat/state", qos)
        self.trigger_pub = self.create_publisher(PoseStamped, "/move_base_simple/goal", qos)
        self.itm_score_pub = self.create_publisher(Float64, "/blip2/cosine_score", qos)
        self.confidence_threshold_pub = self.create_publisher(
            Float64, "/detector/confidence_threshold", qos)
        self.cld_with_score_pub = self.create_publisher(
            MultipleMasksWithConfidence, "/detector/clouds_with_scores", qos)
        self.progress_pub = self.create_publisher(Int32MultiArray, "/habitat/progress", qos)
        self.record_pub = self.create_publisher(Float32MultiArray, "/habitat/record", qos)

        # ROS Publisher for habitat topics
        self.ros_pub = ROSPublisherNonNode(self)

        # Subscribers
        self.create_subscription(Int32, "/habitat/plan_action", self.ros_action_callback, qos)
        self.create_subscription(Int32, "/ros/state", self.ros_state_callback, qos)
        self.create_subscription(Int32, "/ros/expl_state", self.ros_final_state_callback, qos)
        self.create_subscription(Int32, "/ros/expl_result", self.ros_expl_result_callback, qos)

    def ros_action_callback(self, msg):
        self.global_action = msg.data

    def ros_state_callback(self, msg):
        self.ros_state = msg.data

    def ros_final_state_callback(self, msg):
        self.final_state = msg.data

    def ros_expl_result_callback(self, msg):
        self.expl_result = msg.data

    def publish_int32(self, publisher, data):
        msg = Int32()
        msg.data = data
        publisher.publish(msg)

    def publish_float64(self, publisher, data):
        msg = Float64()
        msg.data = float(data)
        publisher.publish(msg)

    def publish_int32_array(self, publisher, data_list):
        msg = Int32MultiArray()
        msg.data = data_list
        publisher.publish(msg)

    def publish_float32_array(self, publisher, data_list):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in data_list]
        publisher.publish(msg)

    def publish_observations_callback(self):
        """Timer callback to publish habitat observations and trigger messages"""
        if self.msg_observations is None:
            return
        tmp = deepcopy(self.msg_observations)
        self.ros_pub.habitat_publish_ros_topic(tmp)
        self.publish_float64(self.confidence_threshold_pub, self.fusion_threshold)
        trigger = PoseStamped()
        self.trigger_pub.publish(trigger)

    def start_observation_timer(self):
        """Start timer for publishing observations"""
        self.pub_timer = self.create_timer(0.25, self.publish_observations_callback)

    def stop_observation_timer(self):
        """Stop observation timer"""
        if self.pub_timer is not None:
            self.pub_timer.cancel()
            self.pub_timer = None


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rclpy.shutdown()
    os._exit(0)


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between each step (default: 0.0)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, args.delay, unknown


def main(cfg: DictConfig, node: HabitatEvalNode, step_delay: float = 0.0) -> None:
    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    node.final_state = 0
    node.expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        node.publish_int32_array(node.progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        node.global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        node.msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, node.fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        node.start_observation_timer()

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        node.ros_state = ROS_STATE.INIT
        while node.ros_state == ROS_STATE.INIT or node.ros_state == ROS_STATE.WAIT_TRIGGER:
            if node.ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif node.ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rclpy.spin_once(node, timeout_sec=0.1)

        # Stop timer publishing when starting action execution
        node.stop_observation_timer()

        print("Agent is ready to go!!!!")

        while rclpy.ok() and not env.episode_over:
            # Process ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.1)

            # Keep publishing observations, confidence, and trigger so FSM
            # always has fresh odom and can re-trigger after episode transitions
            if node.msg_observations is not None:
                node.ros_pub.habitat_publish_ros_topic(deepcopy(node.msg_observations))
                node.publish_float64(node.confidence_threshold_pub, node.fusion_threshold)
                trigger = PoseStamped()
                node.trigger_pub.publish(trigger)

            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if node.global_action is not None:
                if count_steps == max_episode_steps - 1:
                    node.global_action = ACTION.STOP

                if node.global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif node.global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif node.global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif node.global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif node.global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif node.global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                node.global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            node.publish_int32(node.state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)

            # Calculate ITM cosine similarity score
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            node.publish_float64(node.itm_score_pub, cosine)

            # Detect objects in the current observation
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            node.msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            node.ros_pub.habitat_publish_ros_topic(node.msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list, node
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            node.cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Optional delay between steps for visualization
            if step_delay > 0:
                time.sleep(step_delay)

            # Notify ROS system that action execution is complete
            node.publish_int32(node.state_pub, HABITAT_STATE.ACTION_FINISH)

        # Notify ROS system that current episode evaluation is complete
        node.publish_int32(node.state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Wait for FSM to process EPISODE_FINISH and reset to INIT state
        # This prevents a race condition where the next episode starts
        # before the FSM has re-initialized
        for _ in range(50):  # up to 5 seconds
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.ros_state == ROS_STATE.INIT:
                break

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                node.final_state,
                node.expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        node.publish_float32_array(node.record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        time.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init()

    # Register habitat config search path plugin before Hydra init
    from habitat.config.default_structured_configs import (
        HabitatConfigPlugin,
        register_hydra_plugin,
    )
    register_hydra_plugin(HabitatConfigPlugin)

    try:
        node = HabitatEvalNode()
        dataset, step_delay, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg, node, step_delay=step_delay)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rclpy.shutdown()
        os._exit(1)
    finally:
        rclpy.shutdown()
