"""
Manual Habitat ObjectNav Runner (HM3D/MP3D)

This manual runner lets you control the agent with keyboard in Habitat.
It aligns its config loading and CLI with habitat_evaluation.py:

Usage:
    # HM3D-v1
    python habitat_manual_control.py --dataset hm3dv1

    # HM3D-v2 (Default)
    python habitat_manual_control.py --dataset hm3dv2

    # MP3D
    python habitat_manual_control.py --dataset mp3d

    # Test specific episode
    python habitat_manual_control.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import cv2
import rospy
from omegaconf import DictConfig
from std_msgs.msg import Float64

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
    observations_to_image,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from habitat2ros import habitat_publisher
from vlm.utils.get_object_utils import get_object
from vlm.utils.get_itm_message import get_itm_message_cosine
from llm.answer_reader.answer_reader import read_answer
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from vlm.Labels import MP3D_ID_TO_NAME


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
LOOK_UP_KEY = "q"
LOOK_DOWN_KEY = "e"
FINISH = "f"


def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def publish_float64(publisher, data: float):
    """Publish a Float64 value to the given ROS publisher."""
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def print_manual_controls():
    """Print manual control key bindings for the player."""
    print("\nManual controls:")
    print(f"  {FORWARD_KEY} - Move forward")
    print(f"  {LEFT_KEY} - Turn left")
    print(f"  {RIGHT_KEY} - Turn right")
    print(f"  {LOOK_UP_KEY} - Look up")
    print(f"  {LOOK_DOWN_KEY} - Look down")
    print(f"  {FINISH} - Stop (end episode)")
    print("  Ctrl+C - Quit (graceful shutdown)")
    print("Note: Focus the 'Observations' window before pressing keys.\n")


def publish_observations(event):
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Habitat Manual Runner", add_help=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub

    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    score_list = []
    object_masks_list = []
    label_list = []
    llm_answer = []

    cfg = patch_config(cfg)
    env_count = 0 if cfg.test_epi_num == -1 else cfg.test_epi_num
    detector_cfg = cfg.detector
    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)

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
    
    # Initialize Habitat environment
    env = habitat.Env(cfg)
    print("Environment creation successful")

    # Skip episodes to reach the desired starting index
    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1
    observations = env.reset()
    observations["rgb"] = transform_rgb_bgr(observations["rgb"])

    # Display first observation frame
    info = env.get_metrics()
    frame = observations_to_image(observations, info)
    cv2.imshow("Observations", frame)

    camera_pitch = 0.0
    observations["camera_pitch"] = camera_pitch
    msg_observations = deepcopy(observations)

    # Initialize ROS publishers and timer for periodic observation publishing
    ros_pub = habitat_publisher.ROSPublisher()
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )

    print("Agent stepping around inside environment.")
    print_manual_controls()

    label = env.current_episode.object_category

    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    llm_answer, room, fusion_threshold = read_answer(
        llm_answer_path, llm_response_path, label, llm_client
    )

    cld_with_score_msg = MultipleMasksWithConfidence()
    count_steps = 0

    # Manual control loop
    while not rospy.is_shutdown() and not env.episode_over:
        print(f"\n-------------Step: {count_steps}-------------")
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LOOK_UP_KEY):
            action = HabitatSimActions.look_up
            camera_pitch = camera_pitch + np.pi / 6.0
            print("action: LOOK_UP")
        elif keystroke == ord(LOOK_DOWN_KEY):
            action = HabitatSimActions.look_down
            camera_pitch = camera_pitch - np.pi / 6.0
            print("action: LOOK_DOWN")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        timer.shutdown()
        print(f"I'm finding {label}")
        observations = env.step(action)
        count_steps += 1

        info = env.get_metrics()

        # Calculate ITM cosine similarity score
        cosine = get_itm_message_cosine(observations["rgb"], label, room)
        print(f"Target related room: {room}")
        print(f"ITM cosine similarity: {cosine:.3f}")
        publish_float64(itm_score_pub, cosine)

        # Detect objects in the current observation
        detect_img, score_list, object_masks_list, label_list = get_object(
            label, observations["rgb"], detector_cfg, llm_answer
        )

        observations["rgb"] = detect_img
        observations["camera_pitch"] = camera_pitch
        ros_pub.habitat_publish_ros_topic(observations)
        observations["rgb"] = transform_rgb_bgr(detect_img)
        del observations["camera_pitch"]
        frame = observations_to_image(observations, info)

        # Generate and publish object point clouds
        obj_point_cloud_list = get_object_point_cloud(
            cfg, observations, object_masks_list
        )

        cld_with_score_msg.point_clouds = obj_point_cloud_list
        cld_with_score_msg.confidence_scores = score_list
        cld_with_score_msg.label_indices = label_list
        cld_with_score_pub.publish(cld_with_score_msg)

        # Show updated visualization frame
        cv2.imshow("Observations", frame)

    env.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_ros_publisher", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
