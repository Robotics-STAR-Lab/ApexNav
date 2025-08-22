#ifndef _SDF_MAP2D_H
#define _SDF_MAP2D_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <iostream>
#include <utility>
#include <queue>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using std::cout;
using std::endl;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

using namespace std;

namespace cv {
class Mat;
}

class RayCaster2D;

namespace apexnav_planner {
struct MapParam2D;
struct MapData2D;
struct DetectedObject;
class ObjectMap2D;
class ValueMap;
class MapROS;

class SDFMap2D {
public:
  SDFMap2D() = default;
  ~SDFMap2D();  // Explicit destructor declaration for unique_ptr with forward declaration

  /// Grid occupancy states
  enum GRID_STATE { UNKNOWN, FREE, OCCUPIED };

  // Core map management functions
  void initMap(ros::NodeHandle& nh);
  void inputDepthCloud2D(const pcl::PointCloud<pcl::PointXY>::Ptr& points,
      const Eigen::Vector3d& camera_pos, vector<Eigen::Vector2i>& free_grids);
  void inputObjectCloud2D(
      const vector<DetectedObject>& detected_objects, vector<int>& object_cluster_ids);
  void inputVirtualGround(const pcl::PointCloud<pcl::PointXY>::Ptr& points);

  // Coordinate transformation utilities
  void posToIndex(const Eigen::Vector2d& pos, Eigen::Vector2i& id);
  void indexToPos(const Eigen::Vector2i& id, Eigen::Vector2d& pos);
  void boundIndex(Eigen::Vector2i& id);
  Eigen::Vector2i addressToIdx(const int& address);
  int toAddress(const Eigen::Vector2i& id);
  int toAddress(const int& x, const int& y);

  // Map query functions
  bool isInMap(const Eigen::Vector2d& pos);
  bool isInMap(const Eigen::Vector2i& idx);
  int getOccupancy(const Eigen::Vector2d& pos);
  int getOccupancy(const Eigen::Vector2i& id);
  double getDistance(const Eigen::Vector2d& pos);
  double getDistance(const Eigen::Vector2i& id);
  double getDistWithGrad(const Eigen::Vector2d& pos, Eigen::Vector2d& grad);
  int getInflateOccupancy(const Eigen::Vector2d& pos);
  int getInflateOccupancy(const Eigen::Vector2i& id);

  // Map processing functions
  void updateESDFMap();

  // Map property accessors
  void getRegion(Eigen::Vector2d& ori, Eigen::Vector2d& size);
  void getMapBoundary(Eigen::Vector2d& bmin, Eigen::Vector2d& bmax);
  void getLocalUpdatedBox(Eigen::Vector2d& bmin, Eigen::Vector2d& bmax);
  double getResolution();
  int getVoxelNum();
  void setForceOccGrid(const Eigen::Vector2d& pos);

  // Integrated mapping components
  shared_ptr<ObjectMap2D> object_map2d_;
  shared_ptr<ValueMap> value_map_;

private:
  // Internal map processing functions
  void clearAndInflateLocalMap();
  void inflatePoint(const Eigen::Vector2i& pt, int step, vector<Eigen::Vector2i>& pts);
  void setCacheOccupancy(const int& adr, const int& occ);
  Eigen::Vector2d closetPointInMap(const Eigen::Vector2d& pt, const Eigen::Vector2d& camera_pt);
  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

  // Core map data structures
  unique_ptr<MapParam2D> mp_;       ///< Map parameters and configuration
  unique_ptr<MapData2D> md_;        ///< Map data buffers and state
  unique_ptr<MapROS> map_ros_;      ///< ROS interface and visualization
  unique_ptr<RayCaster2D> caster_;  ///< Raycasting utility for occupancy updates

  friend MapROS;  ///< Allow MapROS to access private members

public:
  typedef std::shared_ptr<SDFMap2D> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MapParam2D {
  // Map geometry parameters
  Eigen::Vector2d map_origin_, map_size_;  ///< Map origin and size in world coordinates
  Eigen::Vector2d map_min_boundary_, map_max_boundary_;  ///< Map boundary limits
  Eigen::Vector2i map_voxel_num_;                        ///< Number of voxels in each dimension
  int ray_mode_;                                         ///< Raycasting mode configuration
  double resolution_, resolution_inv_;                   ///< Map resolution and its inverse
  double obstacles_inflation_;                           ///< Inflation radius for obstacles
  double default_dist_;            ///< Default distance value for unobserved areas
  bool optimistic_, signed_dist_;  ///< Mapping mode flags

  // Probabilistic occupancy parameters
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  ///< Occupancy probability parameters
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_,
      min_occupancy_log_;  ///< Log-odds probability parameters
  double max_ray_length_;  ///< Maximum ray length for raycasting
  double local_bound_;     ///< Local mapping boundary limit
  double unknown_flag_;    ///< Flag value for unknown regions
  int buffer_size_;        ///< Size of map data buffers
};

struct MapData2D {
  // Main map data structures
  std::vector<double> occupancy_buffer_;        ///< Primary occupancy probability map
  std::vector<char> occupancy_buffer_inflate_;  ///< Inflated occupancy map for collision checking
  std::vector<double> distance_buffer_neg_;     ///< Negative distance field (inside obstacles)
  std::vector<double> distance_buffer_;         ///< Positive distance field (outside obstacles)
  std::vector<double> tmp_buffer_;              ///< Temporary buffer for computations
  std::vector<char> virtual_ground_buffer_;     ///< Virtual ground plane representation
  std::vector<Eigen::Vector2i> occupancy_need_clear_;  ///< List of cells needing to be cleared

  // Probabilistic update tracking
  vector<short> count_hit_, count_miss_, count_hit_and_miss_;  ///< Ray hit/miss counters
  vector<char> flag_rayend_;                                   ///< Ray endpoint flags
  char raycast_num_;                                           ///< Current raycast iteration number
  queue<int> cache_voxel_;                                     ///< Queue of voxels pending updates

  // Boundary tracking for efficient updates
  Eigen::Vector2i local_bound_min_, local_bound_max_;      ///< Local map boundary indices
  Eigen::Vector2i local_update_min_, local_update_max_;    ///< Local update region indices
  Eigen::Vector2d local_update_mind_, local_update_maxd_;  ///< Local update region coordinates
  Eigen::Vector2i update_min_, update_max_;                ///< Global update boundary indices
  Eigen::Vector2d update_mind_, update_maxd_;              ///< Global update boundary coordinates

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
/// Convert world position to grid index coordinates
inline void SDFMap2D::posToIndex(const Eigen::Vector2d& pos, Eigen::Vector2i& id)
{
  for (int i = 0; i < 2; ++i) id(i) = floor((pos(i) - mp_->map_origin_(i)) * mp_->resolution_inv_);
}

/// Convert grid index to world position coordinates
inline void SDFMap2D::indexToPos(const Eigen::Vector2i& id, Eigen::Vector2d& pos)
{
  for (int i = 0; i < 2; ++i) pos(i) = (id(i) + 0.5) * mp_->resolution_ + mp_->map_origin_(i);
}

/// Clamp grid indices to valid map bounds
inline void SDFMap2D::boundIndex(Eigen::Vector2i& id)
{
  Eigen::Vector2i id1;
  id1(0) = max(min(id(0), mp_->map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_->map_voxel_num_(1) - 1), 0);
  id = id1;
}

/// Convert 2D coordinates to linear array address
inline int SDFMap2D::toAddress(const int& x, const int& y)
{
  return x * mp_->map_voxel_num_(1) + y;
}

/// Convert 2D index to linear array address
inline int SDFMap2D::toAddress(const Eigen::Vector2i& id)
{
  return toAddress(id[0], id[1]);
}

/// Convert linear array address to 2D grid index
inline Eigen::Vector2i SDFMap2D::addressToIdx(const int& address)
{
  int y = address % mp_->map_voxel_num_(1);
  int x = address / mp_->map_voxel_num_(1);
  return Eigen::Vector2i(x, y);
}

/// Check if world position is within map bounds
inline bool SDFMap2D::isInMap(const Eigen::Vector2d& pos)
{
  if (pos(0) < mp_->map_min_boundary_(0) + 1e-4 || pos(1) < mp_->map_min_boundary_(1) + 1e-4)
    return false;
  if (pos(0) > mp_->map_max_boundary_(0) - 1e-4 || pos(1) > mp_->map_max_boundary_(1) - 1e-4)
    return false;
  return true;
}

/// Check if grid index is within map bounds
inline bool SDFMap2D::isInMap(const Eigen::Vector2i& idx)
{
  if (idx(0) < 0 || idx(1) < 0)
    return false;
  if (idx(0) > mp_->map_voxel_num_(0) - 1 || idx(1) > mp_->map_voxel_num_(1) - 1)
    return false;
  return true;
}

/// Get occupancy state at grid index (UNKNOWN=-1, FREE=0, OCCUPIED=1)
inline int SDFMap2D::getOccupancy(const Eigen::Vector2i& id)
{
  if (!isInMap(id))
    return -1;
  double occ = md_->occupancy_buffer_[toAddress(id)];
  if (occ < mp_->clamp_min_log_ - 1e-3)
    return UNKNOWN;
  if (occ > mp_->min_occupancy_log_)
    return OCCUPIED;
  return FREE;
}

/// Get occupancy state at world position
inline int SDFMap2D::getOccupancy(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i id;
  posToIndex(pos, id);
  return getOccupancy(id);
}

/// Get distance value at grid index
inline double SDFMap2D::getDistance(const Eigen::Vector2i& id)
{
  if (!isInMap(id))
    return -1;
  return md_->distance_buffer_[toAddress(id)];
}

/// Get distance value at world position
inline double SDFMap2D::getDistance(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i id;
  posToIndex(pos, id);
  return getDistance(id);
}

/// Get inflated occupancy state at grid index
inline int SDFMap2D::getInflateOccupancy(const Eigen::Vector2i& id)
{
  if (!isInMap(id))
    return -1;
  return int(md_->occupancy_buffer_inflate_[toAddress(id)]);
}

/// Get inflated occupancy state at world position
inline int SDFMap2D::getInflateOccupancy(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i id;
  posToIndex(pos, id);
  return getInflateOccupancy(id);
}

/// Generate inflated points around a given point within inflation radius
inline void SDFMap2D::inflatePoint(
    const Eigen::Vector2i& pt, int step, vector<Eigen::Vector2i>& pts)
{
  pts.clear();
  Eigen::Vector2d pt_pos;
  indexToPos(pt, pt_pos);
  // Generate all points within the inflation radius
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y) {
      Eigen::Vector2i inf_pt = Eigen::Vector2i(pt(0) + x, pt(1) + y);
      Eigen::Vector2d inf_pos;
      indexToPos(inf_pt, inf_pos);
      if ((inf_pos - pt_pos).norm() > mp_->obstacles_inflation_)
        continue;
      if (!isInMap(inf_pt))
        continue;
      pts.push_back(inf_pt);
    }
}

/// Get map resolution (meters per pixel)
inline double SDFMap2D::getResolution()
{
  return mp_->resolution_;
}

/// Get total number of voxels in the map
inline int SDFMap2D::getVoxelNum()
{
  return mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1];
}

/// Get map origin and size
inline void SDFMap2D::getRegion(Eigen::Vector2d& ori, Eigen::Vector2d& size)
{
  ori = mp_->map_origin_, size = mp_->map_size_;
}

/// Get map boundary coordinates
inline void SDFMap2D::getMapBoundary(Eigen::Vector2d& bmin, Eigen::Vector2d& bmax)
{
  bmin = mp_->map_min_boundary_;
  bmax = mp_->map_max_boundary_;
}

/// Get local updated region boundary
inline void SDFMap2D::getLocalUpdatedBox(Eigen::Vector2d& bmin, Eigen::Vector2d& bmax)
{
  bmin = md_->local_update_mind_;
  bmax = md_->local_update_maxd_;
}
}  // namespace apexnav_planner

#endif