import math
import numpy as np
import cv2


def get_point_cloud(
    depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """
    Convert depth image and mask to 3D point cloud
    
    Args:
        depth_image: Depth values in camera frame
        mask: Binary mask indicating valid pixels
        fx, fy: Camera intrinsic parameters (focal lengths)
        
    Returns:
        np.ndarray: 3D points in camera coordinate system [z, -x, -y]
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def transform_points(
    transformation_matrix: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to 3D points
    
    Args:
        transformation_matrix: 4x4 homogeneous transformation matrix
        points: Nx3 array of 3D points
        
    Returns:
        np.ndarray: Transformed 3D points
    """
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    return transformed_points[:, :3] / transformed_points[:, 3:]


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """
    Convert position and yaw to 4x4 transformation matrix
    
    Args:
        xyz: 3D position [x, y, z]
        yaw: Rotation around z-axis in radians
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def too_offset(mask: np.ndarray) -> bool:
    """
    Check if object mask is too close to image edges
    
    This function determines if an object detection is too close to the image
    boundaries, which might indicate partial visibility or unreliable detection.
    
    Args:
        mask: Binary mask of detected object
        
    Returns:
        bool: True if object is too close to edges
    """
    x, y, w, h = cv2.boundingRect(mask)
    third = mask.shape[1] // 3
    if x + w <= third:
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
