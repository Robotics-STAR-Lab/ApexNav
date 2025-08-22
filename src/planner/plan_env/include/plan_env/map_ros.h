#ifndef _MAP_ROS_H
#define _MAP_ROS_H

// ROS core and message handling
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

// Custom messages and mapping components
#include <plan_env/MultipleMasksWithConfidence.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/value_map2d.h>

// OpenCV for image processing
#include <cv_bridge/cv_bridge.h>

// Standard ROS messages
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float64.h>

// PCL for point cloud processing
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/common.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/conditional_removal.h>
#include <unordered_set>

// Type aliases for convenience
using std::shared_ptr;
using Point3D = pcl::PointXYZ;                  ///< 3D point type for PCL operations
using PointCloud3D = pcl::PointCloud<Point3D>;  ///< 3D point cloud type
using Point2D = pcl::PointXY;                   ///< 2D point type for mapping
using PointCloud2D = pcl::PointCloud<Point2D>;  ///< 2D point cloud type for occupancy grid

namespace apexnav_planner {
class SDFMap2D;

class MapROS {
public:
  MapROS() = default;
  ~MapROS() = default;
  // Core interface functions
  void setMap(SDFMap2D* map);
  void init();

private:
  // ROS callback functions
  void depthPoseCallback(  ///< Process synchronized depth image and pose data
      const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose);
  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void detectedObjectCloudCallback(const plan_env::MultipleMasksWithConfidenceConstPtr& msg);
  void itmScoreCallback(const std_msgs::Float64ConstPtr& msg);
  void visCallback(const ros::TimerEvent& /*event*/);

  // Visualization publishing functions
  void publishOccupied();
  void publishInfOccupied();
  void publishFree();
  void publishUnknown();

  void publishObjectMap();
  void publishESDFMap();
  void publishValueMap();
  void publishConfidenceMap();
  void publishUpdateRange();
  void publishPointCloud(const ros::Publisher& pub, const PointCloud3D::Ptr& point_cloud);

  // Data processing functions
  void processDepthImage();     ///< Process raw depth image into 3D point cloud
  void filterPointCloudToXY();  ///< Filter 3D points to 2D occupancy grid representation
  void getObservationObjectsCloud(
      const std::vector<int>& filter_object_ids);  ///< Extract undetected objects from depth data

  // Utility functions
  bool interpolateLineAtZ(
      const Eigen::Vector3d& A, const Eigen::Vector3d& B, double target_z, Eigen::Vector2d& P);
  PointCloud3D::Ptr dbscan(const PointCloud3D::Ptr& cloud, double eps, int minPts);
  void dilateGrids(std::vector<Eigen::Vector2i>& grids, int dilation_radius);

  // Core mapping interface
  SDFMap2D* map_;

  // Message synchronization types
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImagePose;  ///< Policy for synchronizing depth images with pose data
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;

  // ROS communication interfaces
  ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> pose_sub_;
  SynchronizerImagePose sync_image_pose_;  ///< Synchronizer for depth and pose messages

  // ROS publishers for map visualization
  ros::Publisher occupied_pub_, occupied_inflate_pub_, unknown_pub_, free_pub_, esdf_pub_,
      object_grid_pub_, update_range_pub_, depth_cloud_pub_, filtered_depth_cloud_pub_,
      filtered_object_cloud_pub_, all_object_cloud_pub_, over_depth_object_cloud_pub_,
      value_map_pub_, confidence_map_pub_;

  // ROS subscribers for sensor data
  ros::Subscriber detected_object_cloud_sub_, itm_score_sub_;

  // ROS timers for periodic updates
  ros::Timer esdf_timer_, vis_timer_;

  // Camera intrinsic parameters
  double cx_, cy_, fx_, fy_;

  // Depth filtering parameters
  double depth_filter_maxdist_, depth_filter_mindist_;  ///< Valid depth range for filtering
  double filter_min_height_, filter_max_height_;        ///< Height range for obstacle detection
  int depth_filter_margin_;        ///< Margin pixels to ignore near image borders
  double k_depth_scaling_factor_;  ///< Depth value scaling factor for different sensors
  int skip_pixel_;                 ///< Pixel skip factor for processing efficiency
  std::string frame_id_;           ///< Reference frame ID for published data
  double virtual_ground_height_;   ///< Virtual ground plane offset for navigation

  // Map state flags
  bool local_updated_, esdf_need_update_;

  // Current sensor data
  Eigen::Vector3d camera_pos_;                ///< Current camera position in world frame
  Eigen::Quaterniond camera_q_;               ///< Current camera orientation quaternion
  unique_ptr<cv::Mat> depth_image_;           ///< Current depth image data
  vector<Eigen::Vector3d> proj_points_;       ///< Projected 3D points from depth image
  int proj_points_cnt_;                       ///< Count of valid projected points
  PointCloud3D::Ptr depth_cloud_;             ///< Raw 3D point cloud from depth sensor
  PointCloud2D::Ptr filtered_depth_cloud2d_;  ///< Filtered 2D point cloud for occupancy mapping

  // Object detection and ITM integration
  int continue_over_depth_count_;  ///< Counter for maintaining over-depth object consistency
  double itm_score_;               ///< Current image-text matching score
  ros::Time map_start_time_;       ///< Timestamp of mapping system initialization

  friend SDFMap2D;
};

inline void MapROS::dilateGrids(std::vector<Eigen::Vector2i>& grids, int dilation_radius)
{
  if (grids.empty() || dilation_radius <= 0)
    return;

  // Use unordered_set for deduplication (more efficient than set)
  std::unordered_set<uint64_t> dilated_grid_set;

  // Hash function: combine (x,y) coordinates into a single 64-bit integer
  auto hash_grid = [](int x, int y) -> uint64_t {
    return (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
  };

  // Precompute circular dilation template
  std::vector<Eigen::Vector2i> dilation_template;
  for (int dx = -dilation_radius; dx <= dilation_radius; ++dx) {
    for (int dy = -dilation_radius; dy <= dilation_radius; ++dy) {
      if (dx * dx + dy * dy <= dilation_radius * dilation_radius) {
        dilation_template.emplace_back(dx, dy);
      }
    }
  }

  // Apply dilation to each original grid point
  for (const auto& grid : grids) {
    for (const auto& offset : dilation_template) {
      Eigen::Vector2i new_grid = grid + offset;

      // Check if the new grid is within map bounds
      Eigen::Vector2d new_pos;
      map_->indexToPos(new_grid, new_pos);
      if (map_->isInMap(new_pos)) {
        dilated_grid_set.insert(hash_grid(new_grid.x(), new_grid.y()));
      }
    }
  }

  // Convert back to vector format
  grids.clear();
  grids.reserve(dilated_grid_set.size());

  for (const auto& grid_hash : dilated_grid_set) {
    int x = static_cast<int>(grid_hash >> 32);
    int y = static_cast<int>(grid_hash & 0xFFFFFFFF);
    grids.emplace_back(x, y);
  }
}

inline void MapROS::publishObjectMap()
{
  Point3D pt;
  PointCloud3D cloud;
  // Iterate through updated map region to find object cells
  for (int x = map_->md_->update_min_(0); x < map_->md_->update_max_(0); ++x)
    for (int y = map_->md_->update_min_(1); y < map_->md_->update_max_(1); ++y) {
      if (map_->object_map2d_->getObjectGrid(map_->toAddress(x, y)) == 1) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.05;  // Fixed height for visualization
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  object_grid_pub_.publish(cloud_msg);
}

inline void MapROS::publishOccupied()
{
  Point3D pt;
  PointCloud3D cloud;
  // Iterate through updated region to find occupied cells
  for (int x = map_->md_->update_min_(0); x < map_->md_->update_max_(0); ++x)
    for (int y = map_->md_->update_min_(1); y < map_->md_->update_max_(1); ++y) {
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  occupied_pub_.publish(cloud_msg);
}

inline void MapROS::publishInfOccupied()
{
  Point3D pt;
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  occupied_inflate_pub_.publish(cloud_msg);
}

inline void MapROS::publishUnknown()
{
  Point3D pt;
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      // avoid influence of occupancy inflation visualization
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

inline void MapROS::publishFree()
{
  Point3D pt;
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      // Skip inflated occupied cells
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      // Skip unknown cells (below minimum threshold)
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
        continue;
      // Skip occupied cells (above occupancy threshold)
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;
      // Remaining cells are free space
      Eigen::Vector2d pos;
      map_->indexToPos(Eigen::Vector2i(x, y), pos);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = 0.0;
      cloud.push_back(pt);
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  free_pub_.publish(cloud_msg);
}

inline void MapROS::publishConfidenceMap()
{
  double value;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_value = 0.0;
  const double max_value = 1.0;

  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x) {
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      value = map_->value_map_->getConfidence(Eigen::Vector2i(x, y));
      if (value > 1e-3) {
        Eigen::Vector2d pos_2d;
        map_->indexToPos(Eigen::Vector2i(x, y), pos_2d);
        // Normalize confidence value for visualization
        value = std::min(value, max_value);
        value = std::max(value, min_value);
        pt.x = pos_2d(0);
        pt.y = pos_2d(1);
        pt.z = 0.08;
        pt.intensity = (value - min_value) / (max_value - min_value);
        cloud.push_back(pt);
      }
    }
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  confidence_map_pub_.publish(cloud_msg);
}

inline void MapROS::publishValueMap()
{
  double value;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_value = 0.0;
  const double max_value = 1.0;

  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x) {
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      value = map_->value_map_->getValue(Eigen::Vector2i(x, y));
      if (value > 1e-3) {
        // Only show values in free space (not inflated occupied or unknown)
        if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
          continue;
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
          continue;
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
          continue;

        Eigen::Vector2d pos_2d;
        map_->indexToPos(Eigen::Vector2i(x, y), pos_2d);
        // Normalize value for visualization
        value = std::min(value, max_value);
        value = std::max(value, min_value);
        pt.x = pos_2d(0);
        pt.y = pos_2d(1);
        pt.z = 0.08;
        pt.intensity = (value - min_value) / (max_value - min_value);
        cloud.push_back(pt);
      }
    }
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  value_map_pub_.publish(cloud_msg);
}

inline void MapROS::publishPointCloud(
    const ros::Publisher& pub, const PointCloud3D::Ptr& point_cloud)
{
  Point3D pt;
  PointCloud3D cloud;
  // Copy all points from input cloud
  for (int i = 0; i < (int)point_cloud->points.size(); ++i) cloud.push_back(point_cloud->points[i]);

  // Set cloud metadata
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;

  // Convert and publish
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  pub.publish(cloud_msg);
}

inline void MapROS::publishESDFMap()
{
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  // Use local bounds for better visualization performance
  Eigen::Vector2i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector2i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x) {
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      // Skip inflated occupied cells
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      // Skip unknown cells
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
        continue;
      // Skip occupied cells
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;

      Eigen::Vector2d pos_2d;
      map_->indexToPos(Eigen::Vector2i(x, y), pos_2d);
      dist = map_->getDistance(pos_2d);

      // Clamp distance values for visualization
      dist = std::min(dist, max_dist);
      dist = std::max(dist, min_dist);

      pt.x = pos_2d(0);
      pt.y = pos_2d(1);
      pt.z = 0.08;  // Fixed height for 2D visualization
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);
}

inline void MapROS::publishUpdateRange()
{
  Eigen::Vector2d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;

  // Convert grid indices to world coordinates
  map_->indexToPos(map_->md_->local_update_min_, esdf_min_pos);
  map_->indexToPos(map_->md_->local_update_max_, esdf_max_pos);

  // Calculate center and size of update region
  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;

  // Configure marker properties
  mk.header.frame_id = frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;

  // Set position and scale
  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = 0.0;
  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = 0.1;  // Thin box for 2D visualization

  // Set color (transparent red)
  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;

  // Set orientation
  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

}  // namespace apexnav_planner

#endif