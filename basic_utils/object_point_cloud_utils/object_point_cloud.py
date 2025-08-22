import cv2
import numpy as np
import math

from sensor_msgs.msg import PointCloud2, PointField
import rospy

from basic_utils.object_point_cloud_utils.geometry_utils import (
    get_point_cloud,
    xyz_yaw_to_tf_matrix,
    too_offset,
    transform_points,
)


def get_object_point_cloud(cfg, observations, object_masks_list):
    """
    Extract 3D point clouds for detected objects from sensor observations
    
    This function processes depth images and object masks to generate 3D point clouds
    for each detected object, transforming them from camera coordinates to world coordinates.
    
    Args:
        cfg: Configuration object containing sensor parameters
        observations: Dictionary containing sensor data (depth, gps, compass)
        object_masks_list: List of binary masks for detected objects
        
    Returns:
        list: List of ROS PointCloud2 messages for each object
    """
    obj_point_cloud_list = []
    depth = observations["depth"]
    y = observations["gps"][0]
    x = observations["gps"][2]
    camera_yaw = observations["compass"][0].item()
    cfg_depth_sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera_height = cfg_depth_sensor.position[1]
    camera_min_depth = cfg_depth_sensor.min_depth
    camera_max_depth = cfg_depth_sensor.max_depth
    hfov = cfg_depth_sensor["hfov"]
    height = cfg_depth_sensor["height"]
    width = cfg_depth_sensor["width"]
    fx = width / (2 * math.tan(hfov * np.pi / 360.0))
    fy = height / (2 * math.tan(hfov / width * height * np.pi / 360.0))
    for object_mask in object_masks_list:
        local_cloud = extract_object_cloud(
            depth, object_mask, camera_min_depth, camera_max_depth, fx, fy
        )
        camera_position = np.array([-x, -y, camera_height])
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        if len(local_cloud) == 0:
            obj_point_cloud_list.append(PointCloud2())
            continue
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            within_range = (
                local_cloud[:, 0] <= camera_max_depth * 0.95
            ) * 1.0  # 5% margin
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
        obj_point_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        obj_point_cloud = np.concatenate(
            (obj_point_cloud, within_range[:, None]), axis=1
        )
        pc2 = convert_to_pointcloud2(obj_point_cloud)
        obj_point_cloud_list.append(pc2)
    return obj_point_cloud_list


def extract_object_cloud(
    depth: np.ndarray,
    object_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
) -> np.ndarray:
    """
    Extract 3D point cloud from depth image using object mask
    
    Args:
        depth: Depth image array
        object_mask: Binary mask indicating object pixels
        min_depth, max_depth: Depth sensor range limits
        fx, fy: Camera focal length parameters
        
    Returns:
        np.ndarray: 3D point cloud in camera coordinates
    """
    erosion_size = 1
    final_mask = object_mask * 255
    final_mask = cv2.erode(final_mask, None, iterations=erosion_size)  # type: ignore
    valid_depth = depth.copy()
    # valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
    valid_depth = valid_depth * (max_depth - min_depth) + min_depth
    valid_depth_img = valid_depth[:, :, 0]

    cloud = get_point_cloud(valid_depth_img, final_mask, fx, fy)

    return cloud


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    Randomly sample a subset of points from point cloud
    
    Args:
        points: Input point cloud array
        size: Number of points to sample
        
    Returns:
        np.ndarray: Randomly sampled subset of points
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def convert_to_pointcloud2(obj_point_cloud):
    """
    Convert numpy point cloud to ROS PointCloud2 message
    
    Args:
        obj_point_cloud: Numpy array of 3D points
        
    Returns:
        PointCloud2: ROS message containing the point cloud
    """
    obj_point_cloud = obj_point_cloud.astype(np.float32)

    # Create PointCloud2 message
    pc2 = PointCloud2()
    pc2.header.stamp = rospy.Time.now()
    pc2.header.frame_id = "world"
    pc2.height = 1
    pc2.width = obj_point_cloud.shape[0]
    pc2.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    pc2.is_bigendian = False
    pc2.point_step = 16
    pc2.row_step = pc2.point_step * pc2.width
    pc2.is_dense = True
    pc2.data = obj_point_cloud.tobytes()
    return pc2
