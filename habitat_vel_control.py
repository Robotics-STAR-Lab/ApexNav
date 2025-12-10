import os
import signal
import gzip
import json
import time

import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig
from habitat.config.default import patch_config
import hydra  # noqa
from habitat2ros import habitat_publisher
import rospy
from copy import deepcopy
from std_msgs.msg import Float64, String
from vlm.Labels import MP3D_ID_TO_NAME
from geometry_msgs.msg import Twist
import habitat_sim
from habitat_sim.utils import common as utils

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import observations_to_image


def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    global msg_observations, fusion_score
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    msg = Float64()
    msg.data = fusion_score
    confidence_threshold_pub.publish(msg)


def cmd_vel_callback(msg):
    global cmd_vel, cmd_omega
    cmd_vel = msg.linear.x
    cmd_omega = msg.angular.z


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="habitat_vel_control",
)
def main(cfg: DictConfig) -> None:
    global msg_observations, fusion_score
    global ros_pub, confidence_threshold_pub
    global obj_point_cloud
    global obj_point_cloud_pub
    global cmd_vel, cmd_omega

    cmd_vel = 0.0
    cmd_omega = 0.0

    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }


    cfg = patch_config(cfg)
    env_count = cfg.test_epi_num
    print(env_count)
    cfg_rgb_sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor

    height = cfg_rgb_sensor["height"]
    width = cfg_rgb_sensor["width"]
    fusion_score = 0.3

    # Control-related parameters
    fps = 30.0
    time_step = 1.0 / fps

    # No LLM output directory required when LLM is disabled/removed

    # Add top_down_map and collision measurements
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
    sim = env.sim
    sim.set_gravity(np.array([0.0, 0.0, 0.0]))
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.ang_vel_is_local = True

    print("Environment creation successful")
    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1
    observations = env.reset()
    observations["rgb"] = transform_rgb_bgr(observations["rgb"])

    agent = sim.agents[0]

    info = env.get_metrics()
    frame = observations_to_image(observations, info)
    # cv2.imshow("Observations", frame)

    camera_pitch = 0.0
    observations["camera_pitch"] = camera_pitch
    observations["linear_velocity"] = 0.0
    observations["angular_velocity"] = 0.0
    msg_observations = deepcopy(observations)

    ros_pub = habitat_publisher.ROSPublisher()
    cmd_sub = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback, queue_size=10)
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    # clouds-with-scores publisher removed (not used in this script)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    # Publish the target label so other nodes can subscribe
    label_pub = rospy.Publisher("/detector/label", String, queue_size=1, latch=True)

    print("Agent stepping around inside environment.")

    label = env.current_episode.object_category

    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    # Publish the selected label so external nodes (e.g. real-world node) can receive it
    try:
        label_pub.publish(String(data=label))
        rospy.loginfo("Published target label: %s", label)
    except Exception as e:
        print(f"Failed to publish label: {e}")

    rate = rospy.Rate(fps)

    tmp_cnt = 0
    while not rospy.is_shutdown() and not env.episode_over:
        loop_begin_time = rospy.Time.now()
        object_mask = np.zeros((height, width), dtype=np.uint8)
        vel_control.linear_velocity = np.array([0.0, 0.0, 0.0])  # y+ None x-
        vel_control.angular_velocity = np.array([0.0, 0.0, 0.0])
        timer.shutdown()

        vel_control.linear_velocity = np.array([0.0, 0.0, -cmd_vel])
        vel_control.angular_velocity = np.array([0.0, cmd_omega, 0.0])

        tmp_cnt += 1
        if tmp_cnt >= 1 and tmp_cnt <= 4.0 * fps + 5:
            vel_control.angular_velocity = np.array([0.0, np.pi / 2.0, 0.0])

        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state
        )
        end_pos = sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
        agent.set_state(agent_state)

        rospy.loginfo_throttle(5.0, f"I'm finding {label}")

        observations = env.step(HabitatSimActions.move_forward)

        habitat_env_time = rospy.Time.now() - loop_begin_time

        info = env.get_metrics()

        observations["camera_pitch"] = camera_pitch
        observations["linear_velocity"] = cmd_vel
        observations["angular_velocity"] = cmd_omega
        ros_pub.habitat_publish_ros_topic(observations)
        msg = Float64()
        msg.data = 0.5
        confidence_threshold_pub.publish(msg)
        observations["rgb"] = transform_rgb_bgr(observations["rgb"])
        del observations["camera_pitch"]
        del observations["linear_velocity"]
        del observations["angular_velocity"]
        frame = observations_to_image(observations, info)

        if habitat_env_time.to_sec() >= time_step:
            print(
                f"env step time: {habitat_env_time.to_sec()*1000.0:.1f}ms VS {time_step*1000.0:.1f}ms"
            )

        rate.sleep()

    env.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_ros_publisher", anonymous=True)

    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
