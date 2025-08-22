#ifndef _FRONTIER_MAP2D_H_
#define _FRONTIER_MAP2D_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>

#include <plan_env/sdf_map2d.h>
#include <plan_env/raycast2d.h>
#include <plan_env/perception_utils2d.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

// Type aliases for improved readability
using Eigen::Vector2d;
using Eigen::Vector2i;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster2D;

namespace apexnav_planner {
class PerceptionUtils2D;
class SDFMap2D;

// A frontier cluster
struct Frontier2D {
  /// Complete grid cells belonging to this frontier cluster
  vector<Vector2d> cells_;

  /// Geometric centroid of all frontier cells
  Vector2d average_;

  /// Unique identifier for this frontier cluster
  int id_;

  /// Axis-aligned bounding box min/max coordinates
  Vector2d box_min_, box_max_;
};

class FrontierMap2D {
public:
  FrontierMap2D(const shared_ptr<SDFMap2D>& sdf_map, ros::NodeHandle& nh);
  ~FrontierMap2D(){};

  void searchFrontiers();
  bool dormantSeenFrontiers(Vector2d sensor_pos, double sensor_yaw);
  void setForceDormantFrontier(const Vector2d& frontier_center);

  void getFrontiers(vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages);
  void getDormantFrontiers(vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages);
  void getFrontierBoxes(vector<pair<Vector2d, Vector2d>>& boxes);
  bool isAnyFrontierChanged();
  void wrapYaw(double& yaw);

  shared_ptr<PerceptionUtils2D> percep_utils_;

private:
  enum FRONTIER_STATE { NONE, ACTIVE, DORMANT, FORCE_DORMANT };
  void expandFrontier(const Eigen::Vector2i& first);
  void splitLargeFrontiers(list<Frontier2D>& frontiers);
  bool splitHorizontally(const Frontier2D& frontier, list<Frontier2D>& splits);
  bool isFrontierChanged(const Frontier2D& ft);
  bool haveOverlap(
      const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2);
  void computeFrontierInfo(Frontier2D& frontier);

  // Utils
  int countConnectUnknownGrids(const Eigen::Vector2d& pos);
  vector<Eigen::Vector2i> fourNeighbors(const Eigen::Vector2i& idx);
  vector<Eigen::Vector2i> allNeighbors(const Eigen::Vector2i& idx);
  bool isSatisfyFrontier(const Eigen::Vector2i& idx);

  // Wrapper of sdf map
  int toAdr(const Eigen::Vector2i& idx);
  bool inMap(const Eigen::Vector2i& idx);
  bool isNeighborUnknown(const Eigen::Vector2i& idx);
  bool isNeighborFree(const Eigen::Vector2i& idx);
  bool knownFree(const Eigen::Vector2i& idx);
  bool knownUnknown(const Eigen::Vector2i& idx);

  // Data
  vector<char> frontier_flag_;
  list<Frontier2D> frontiers_, dormant_frontiers_, candidate_frontiers_;

  // Params
  int cluster_min_;
  double cluster_size_xy_;
  double min_view_finish_fraction_, resolution_;
  int min_contain_unknown_;

  shared_ptr<SDFMap2D> sdf_map_;
  unique_ptr<RayCaster2D> raycaster_;
};

inline bool FrontierMap2D::haveOverlap(
    const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2)
{
  // Compute intersection bounds
  Vector2d bmin, bmax;
  for (int i = 0; i < 2; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3)  // No overlap in this dimension
      return false;
  }
  return true;  // Overlap exists in all dimensions
}

inline void FrontierMap2D::wrapYaw(double& yaw)
{
  while (yaw < -M_PI) yaw += 2 * M_PI;
  while (yaw > M_PI) yaw -= 2 * M_PI;
}

inline vector<Eigen::Vector2i> FrontierMap2D::fourNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(4);
  neighbors[0] = idx + Eigen::Vector2i(-1, 0);
  neighbors[1] = idx + Eigen::Vector2i(1, 0);
  neighbors[2] = idx + Eigen::Vector2i(0, -1);
  neighbors[3] = idx + Eigen::Vector2i(0, 1);
  return neighbors;
}

inline vector<Eigen::Vector2i> FrontierMap2D::allNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(8);
  int count = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0)
        continue;
      neighbors[count++] = idx + Eigen::Vector2i(x, y);
    }
  }
  return neighbors;
}

inline bool FrontierMap2D::isNeighborUnknown(const Eigen::Vector2i& idx)
{
  // At least one neighbor is unknown
  auto nbrs = fourNeighbors(idx);
  for (auto nbr : nbrs) {
    if (sdf_map_->getOccupancy(nbr) == SDFMap2D::UNKNOWN)
      return true;
  }
  return false;
}

inline bool FrontierMap2D::isNeighborFree(const Eigen::Vector2i& idx)
{
  // At least one neighbor is unknown
  auto nbrs = fourNeighbors(idx);
  for (auto nbr : nbrs) {
    if (sdf_map_->getOccupancy(nbr) == SDFMap2D::FREE)
      return true;
  }
  return false;
}

inline int FrontierMap2D::toAdr(const Eigen::Vector2i& idx)
{
  return sdf_map_->toAddress(idx);
}

inline bool FrontierMap2D::knownFree(const Eigen::Vector2i& idx)
{
  return sdf_map_->getOccupancy(idx) == SDFMap2D::FREE;
}

inline bool FrontierMap2D::knownUnknown(const Eigen::Vector2i& idx)
{
  return sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN;
}

inline bool FrontierMap2D::inMap(const Eigen::Vector2i& idx)
{
  return sdf_map_->isInMap(idx);
}
}  // namespace apexnav_planner
#endif