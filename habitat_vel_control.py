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
from habitat2ros.habitat_publisher import ROSPublisherNonNode
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from copy import deepcopy
from std_msgs.msg import Float64, String
from vlm.Labels import MP3D_ID_TO_NAME
from geometry_msgs.msg import Twist, PoseStamped
import habitat_sim
from habitat_sim.utils import common as utils

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import observations_to_image


class HabitatVelControlNode(Node):
    def __init__(self):
        super().__init__('habitat_ros_publisher')

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State variables
        self.msg_observations = None
        self.fusion_score = 0.3
        self.cmd_vel = 0.0
        self.cmd_omega = 0.0
        self.pub_timer = None

        # Publishers
        self.itm_score_pub = self.create_publisher(Float64, "/blip2/cosine_score", qos)
        self.confidence_threshold_pub = self.create_publisher(
            Float64, "/detector/confidence_threshold", qos)
        self.label_pub = self.create_publisher(String, "/detector/label", 1)
        self.trigger_pub = self.create_publisher(PoseStamped, "/move_base_simple/goal", qos)

        # ROS Publisher for habitat topics
        self.ros_pub = ROSPublisherNonNode(self)

        # Subscribers
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, qos)

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg.linear.x
        self.cmd_omega = msg.angular.z

    def publish_observations_callback(self):
        """Timer callback to publish habitat observations"""
        if self.msg_observations is None:
            return
        tmp = deepcopy(self.msg_observations)
        self.ros_pub.habitat_publish_ros_topic(tmp)
        msg = Float64()
        msg.data = float(self.fusion_score)
        self.confidence_threshold_pub.publish(msg)

    def start_observation_timer(self):
        """Start timer for publishing observations"""
        self.pub_timer = self.create_timer(0.1, self.publish_observations_callback)

    def stop_observation_timer(self):
        """Stop observation timer"""
        if self.pub_timer is not None:
            self.pub_timer.cancel()
            self.pub_timer = None


def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    rclpy.shutdown()
    os._exit(0)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="habitat_vel_control",
)
def main(cfg: DictConfig) -> None:
    rclpy.init()
    node = HabitatVelControlNode()

    try:
        run_simulation(cfg, node)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rclpy.shutdown()
        os._exit(1)
    finally:
        rclpy.shutdown()


def run_simulation(cfg: DictConfig, node: HabitatVelControlNode):
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
    node.fusion_score = 0.3

    # Control-related parameters
    fps = 30.0
    time_step = 1.0 / fps

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
    node.msg_observations = deepcopy(observations)

    # Start timer for periodic observation publishing
    node.start_observation_timer()

    print("Agent stepping around inside environment.")

    label = env.current_episode.object_category

    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    # Publish the selected label so external nodes (e.g. real-world node) can receive it
    try:
        label_msg = String()
        label_msg.data = label
        node.label_pub.publish(label_msg)
        node.get_logger().info(f"Published target label: {label}")
    except Exception as e:
        print(f"Failed to publish label: {e}")

    tmp_cnt = 0
    last_time = node.get_clock().now()

    while rclpy.ok() and not env.episode_over:
        loop_begin_time = node.get_clock().now()

        # Process ROS callbacks
        rclpy.spin_once(node, timeout_sec=0.0)

        object_mask = np.zeros((height, width), dtype=np.uint8)
        vel_control.linear_velocity = np.array([0.0, 0.0, 0.0])  # y+ None x-
        vel_control.angular_velocity = np.array([0.0, 0.0, 0.0])
        node.stop_observation_timer()

        vel_control.linear_velocity = np.array([0.0, 0.0, -node.cmd_vel])
        vel_control.angular_velocity = np.array([0.0, node.cmd_omega, 0.0])

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

        # Log periodically
        current_time = node.get_clock().now()
        if (current_time - last_time).nanoseconds / 1e9 >= 5.0:
            node.get_logger().info(f"I'm finding {label}")
            last_time = current_time

        observations = env.step(HabitatSimActions.move_forward)

        habitat_env_time = node.get_clock().now() - loop_begin_time

        info = env.get_metrics()

        observations["camera_pitch"] = camera_pitch
        observations["linear_velocity"] = node.cmd_vel
        observations["angular_velocity"] = node.cmd_omega
        node.ros_pub.habitat_publish_ros_topic(observations)
        msg = Float64()
        msg.data = 0.5
        node.confidence_threshold_pub.publish(msg)

        # Republish label periodically so late-joining nodes can receive it
        label_msg = String()
        label_msg.data = label
        node.label_pub.publish(label_msg)

        observations["rgb"] = transform_rgb_bgr(observations["rgb"])
        del observations["camera_pitch"]
        del observations["linear_velocity"]
        del observations["angular_velocity"]
        frame = observations_to_image(observations, info)

        habitat_env_time_sec = habitat_env_time.nanoseconds / 1e9
        if habitat_env_time_sec >= time_step:
            print(
                f"env step time: {habitat_env_time_sec*1000.0:.1f}ms VS {time_step*1000.0:.1f}ms"
            )

        # Rate limiting
        elapsed = (node.get_clock().now() - loop_begin_time).nanoseconds / 1e9
        sleep_time = time_step - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rclpy.shutdown()
        os._exit(1)
