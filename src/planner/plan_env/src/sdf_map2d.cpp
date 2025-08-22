/**
 * @file sdf_map2d.cpp
 * @brief Implementation of 2D Signed Distance Field mapping system for autonomous navigation
 *
 * This file contains the complete implementation of the SDFMap2D class,
 * providing probabilistic occupancy mapping, Euclidean Signed Distance Field (ESDF) computation,
 * raycasting-based sensor fusion, and obstacle inflation capabilities. The system integrates
 * depth sensor data with probabilistic occupancy models to create robust spatial representations
 * for path planning and collision avoidance.
 *
 * @author Zager-Zhang
 */

#include <plan_env/sdf_map2d.h>
#include <plan_env/map_ros.h>
#include <plan_env/object_map2d.h>
#include <plan_env/value_map2d.h>
#include <unordered_map>

namespace apexnav_planner {
SDFMap2D::~SDFMap2D() = default;

void SDFMap2D::initMap(ros::NodeHandle& nh)
{
  mp_.reset(new MapParam2D);
  md_.reset(new MapData2D);
  map_ros_.reset(new MapROS);

  // Load map properties from ROS parameters
  double x_size, y_size;
  nh.param("sdf_map/ray_mode", mp_->ray_mode_, 0);
  nh.param("sdf_map/resolution", mp_->resolution_, -1.0);
  nh.param("sdf_map/map_size_x", x_size, -1.0);
  nh.param("sdf_map/map_size_y", y_size, -1.0);
  nh.param("sdf_map/obstacles_inflation", mp_->obstacles_inflation_, -1.0);
  nh.param("sdf_map/local_bound", mp_->local_bound_, 1.0);
  nh.param("sdf_map/optimistic", mp_->optimistic_, true);
  nh.param("sdf_map/signed_dist", mp_->signed_dist_, false);
  mp_->default_dist_ = 0.0;

  // Calculate map boundaries and resolution parameters
  mp_->local_bound_ = max(mp_->resolution_, mp_->local_bound_);
  mp_->resolution_inv_ = 1 / mp_->resolution_;
  mp_->map_origin_ = Eigen::Vector2d(-x_size / 2.0, -y_size / 2.0);
  mp_->map_size_ = Eigen::Vector2d(x_size, y_size);
  for (int i = 0; i < 2; ++i) mp_->map_voxel_num_(i) = ceil(mp_->map_size_(i) / mp_->resolution_);
  mp_->map_min_boundary_ = mp_->map_origin_;
  mp_->map_max_boundary_ = mp_->map_origin_ + mp_->map_size_;

  // Load raycasting parameters for probabilistic occupancy fusion
  nh.param("sdf_map/p_hit", mp_->p_hit_, 0.70);
  nh.param("sdf_map/p_miss", mp_->p_miss_, 0.35);
  nh.param("sdf_map/p_min", mp_->p_min_, 0.12);
  nh.param("sdf_map/p_max", mp_->p_max_, 0.97);
  nh.param("sdf_map/p_occ", mp_->p_occ_, 0.80);
  nh.param("sdf_map/max_ray_length", mp_->max_ray_length_, -0.1);

  // Check if using habitat simulator and override parameters if necessary
  bool is_real_world;
  nh.param("fsm/is_real_world", is_real_world, false);

  if (!is_real_world) {
    double habitat_max_depth, agent_radius;
    nh.param("/habitat/simulator/agents/main_agent/sim_sensors/depth_sensor/max_depth",
        habitat_max_depth, -1.0);
    nh.param("/habitat/simulator/agents/main_agent/radius", agent_radius, -1.0);
    if (habitat_max_depth != -1.0) {
      mp_->max_ray_length_ = habitat_max_depth - 1e-3;
      ROS_WARN(
          "Using habitat simulator params, set max_ray_length_ = %.2f m", mp_->max_ray_length_);
    }
    if (agent_radius != -1.0) {
      mp_->obstacles_inflation_ = agent_radius;
      ROS_WARN("Using habitat simulator params, set obstacles_inflation_ = %.2f m",
          mp_->obstacles_inflation_);
    }
  }

  // Convert probabilities to log-odds for efficient computation
  auto logit = [](const double& x) { return log(x / (1 - x)); };
  mp_->prob_hit_log_ = logit(mp_->p_hit_);
  mp_->prob_miss_log_ = logit(mp_->p_miss_);
  mp_->clamp_min_log_ = logit(mp_->p_min_);
  mp_->clamp_max_log_ = logit(mp_->p_max_);
  mp_->min_occupancy_log_ = logit(mp_->p_occ_);
  mp_->unknown_flag_ = 0.01;
  ROS_INFO("prob_hit_log = %f, prob_miss_log = %f", mp_->prob_hit_log_, mp_->prob_miss_log_);

  // Initialize map data structures and buffers
  mp_->buffer_size_ = mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1);
  md_->occupancy_buffer_ =
      vector<double>(mp_->buffer_size_, mp_->clamp_min_log_ - mp_->unknown_flag_);
  md_->occupancy_buffer_inflate_ = vector<char>(mp_->buffer_size_, 0);
  md_->count_hit_and_miss_ = vector<short>(mp_->buffer_size_, 0);
  md_->count_hit_ = vector<short>(mp_->buffer_size_, 0);
  md_->count_miss_ = vector<short>(mp_->buffer_size_, 0);
  md_->flag_rayend_ = vector<char>(mp_->buffer_size_, -1);
  md_->distance_buffer_neg_ = vector<double>(mp_->buffer_size_, mp_->default_dist_);
  md_->distance_buffer_ = vector<double>(mp_->buffer_size_, mp_->default_dist_);
  md_->tmp_buffer_ = vector<double>(mp_->buffer_size_, 0);
  md_->virtual_ground_buffer_ = vector<char>(mp_->buffer_size_, 0);

  // Initialize tracking variables for map updates
  md_->raycast_num_ = 0;
  md_->local_update_min_ = md_->local_update_max_ = Eigen::Vector2i(0, 0);
  md_->local_update_mind_ = md_->local_update_maxd_ = Eigen::Vector2d(0, 0);
  md_->update_min_ = md_->update_max_ = Eigen::Vector2i(0, 0);
  md_->update_mind_ = md_->update_maxd_ = Eigen::Vector2d(0, 0);

  // Initialize ROS components and raycaster
  object_map2d_.reset(new ObjectMap2D(this, nh));
  value_map_.reset(new ValueMap(this, nh));
  map_ros_->setMap(this);
  map_ros_->node_ = nh;
  map_ros_->init();

  caster_.reset(new RayCaster2D);
  caster_->setParams(mp_->resolution_, mp_->map_origin_);
}

void SDFMap2D::setCacheOccupancy(const int& adr, const int& occ)
{
  // Add to update queue if this voxel is being visited for the first time
  if (md_->count_hit_[adr] == 0 && md_->count_miss_[adr] == 0)
    md_->cache_voxel_.push(adr);

  // Update hit/miss counters based on occupancy observation
  if (occ == 0)
    md_->count_miss_[adr] += 1;
  else if (occ == 1)
    md_->count_hit_[adr] += 1;
}

void SDFMap2D::inputVirtualGround(const pcl::PointCloud<pcl::PointXY>::Ptr& points)
{
  int point_num = points->points.size();
  if (point_num == 0)
    return;

  // Mark virtual ground points in the map buffer
  Eigen::Vector2i idx;
  for (int i = 0; i < point_num; ++i) {
    Eigen::Vector2d pt_w;
    pt_w << points->points[i].x, points->points[i].y;
    if (!isInMap(pt_w))
      continue;

    posToIndex(pt_w, idx);
    int vox_adr = toAddress(idx);
    md_->virtual_ground_buffer_[vox_adr] = 1;
  }
}

void SDFMap2D::inputObjectCloud2D(
    const vector<DetectedObject>& detected_objects, vector<int>& object_cluster_ids)
{
  object_cluster_ids.clear();
  // Process each detected object and try to associate it with existing clusters
  for (auto detected_object : detected_objects) {
    if (detected_object.cloud->points.empty())
      continue;

    int object_cluster_id = object_map2d_->searchSingleObjectCluster(detected_object);
    if (object_cluster_id != -1)
      object_cluster_ids.push_back(object_cluster_id);
  }
}

void SDFMap2D::inputDepthCloud2D(const pcl::PointCloud<pcl::PointXY>::Ptr& points,
    const Eigen::Vector3d& camera_pos, vector<Eigen::Vector2i>& free_grids)
{
  free_grids.clear();
  int point_num = points->points.size();
  if (point_num == 0)
    return;
    
  // Initialize raycast tracking and clear occupancy updates
  md_->raycast_num_ += 1;
  md_->occupancy_need_clear_.clear();

  // Convert 3D camera position to 2D sensor position
  Eigen::Vector2d sensor_pos = Eigen::Vector2d(camera_pos(0), camera_pos(1));
  Eigen::Vector2d update_mind = sensor_pos;
  Eigen::Vector2d update_maxd = sensor_pos;

  // Calculate local bounds for this update
  Eigen::Vector2d bound_inf(mp_->local_bound_, mp_->local_bound_);
  Eigen::Vector2d local_bound_mind = sensor_pos - bound_inf;
  Eigen::Vector2d local_bound_maxd = sensor_pos + bound_inf;
  posToIndex(local_bound_mind, md_->local_bound_min_);
  posToIndex(local_bound_maxd, md_->local_bound_max_);
  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);

  Eigen::Vector2d pt_w, tmp;
  Eigen::Vector2i idx;
  int vox_adr;
  double length;
  std::unordered_map<int, char> flag_occ, flag_free;

  // First pass: Mark all occupied grids from depth points
  for (int i = 0; i < point_num; ++i) {
    auto& pt = points->points[i];
    pt_w << pt.x, pt.y;
    int tmp_flag;
    
    // Process point and determine if it should be marked as occupied
    if (!isInMap(pt_w)) {
      // Find closest point in map and set as free
      pt_w = closetPointInMap(pt_w, sensor_pos);
      length = (pt_w - sensor_pos).norm();
      if (length > mp_->max_ray_length_)
        pt_w = (pt_w - sensor_pos) / length * mp_->max_ray_length_ + sensor_pos;
      tmp_flag = 0;
    }
    else {
      length = (pt_w - sensor_pos).norm();
      if (length > mp_->max_ray_length_) {
        pt_w = (pt_w - sensor_pos) / length * mp_->max_ray_length_ + sensor_pos;
        tmp_flag = 0;
      }
      else
        tmp_flag = 1;
    }
    posToIndex(pt_w, idx);
    vox_adr = toAddress(idx);
    if (tmp_flag)
      flag_occ[vox_adr] = 1;  // Mark as occupied in hash map
  }

  // Second pass: Perform raycasting to mark free space, excluding occupied grids
  for (int i = 0; i < point_num; ++i) {
    auto& pt = points->points[i];
    pt_w << pt.x, pt.y;
    int tmp_flag;
    
    // Process point and determine occupancy flag
    if (!isInMap(pt_w)) {
      // Find closest point in map and set as free
      pt_w = closetPointInMap(pt_w, sensor_pos);
      length = (pt_w - sensor_pos).norm();
      if (length > mp_->max_ray_length_)
        pt_w = (pt_w - sensor_pos) / length * mp_->max_ray_length_ + sensor_pos;
      tmp_flag = 0;
    }
    else {
      length = (pt_w - sensor_pos).norm();
      if (length > mp_->max_ray_length_) {
        pt_w = (pt_w - sensor_pos) / length * mp_->max_ray_length_ + sensor_pos;
        tmp_flag = 0;
      }
      else
        tmp_flag = 1;
    }
    posToIndex(pt_w, idx);
    vox_adr = toAddress(idx);
    if (tmp_flag == 1)
      setCacheOccupancy(vox_adr, tmp_flag);

    // Update the bounding box of affected area
    for (int k = 0; k < 2; ++k) {
      update_mind[k] = min(update_mind[k], pt_w[k]);
      update_maxd[k] = max(update_maxd[k], pt_w[k]);
    }
    
    // Skip raycasting if this ray endpoint was already processed
    if (md_->flag_rayend_[vox_adr] == md_->raycast_num_)
      continue;
    else
      md_->flag_rayend_[vox_adr] = md_->raycast_num_;

    // Perform raycasting based on ray mode
    if (mp_->ray_mode_ == 0) {
      // Ray mode 0: Cast from point to sensor
      caster_->input(pt_w, sensor_pos);
      caster_->nextId(idx);
      setCacheOccupancy(toAddress(idx), 0);
      if (!flag_free.count(toAddress(idx))) {
        flag_free[toAddress(idx)] = 1;
        free_grids.push_back(idx);
      }
      while (caster_->nextId(idx)) {
        int adr = toAddress(idx);
        if (flag_occ.count(adr) && flag_occ[adr] == 1)  // Skip if marked as occupied
          continue;
        if (md_->virtual_ground_buffer_[adr])  // Skip virtual ground
          continue;
        setCacheOccupancy(adr, 0);
        if (!flag_free.count(adr)) {
          flag_free[adr] = 1;
          free_grids.push_back(idx);
        }
      }
      setCacheOccupancy(toAddress(idx), 0);
      if (!flag_free.count(toAddress(idx))) {
        flag_free[toAddress(idx)] = 1;
        free_grids.push_back(idx);
      }
    }
    else {
      // Ray mode 1: Cast from sensor to point
      caster_->input(sensor_pos, pt_w);
      while (caster_->nextId(idx)) {
        int adr = toAddress(idx);
        if (flag_occ.count(adr) && flag_occ[adr] == 1)  // Stop if hit occupied grid
          break;
        if (md_->virtual_ground_buffer_[adr])  // Stop at virtual ground
          break;
        setCacheOccupancy(adr, 0);
        if (!flag_free.count(adr)) {
          flag_free[adr] = 1;
          free_grids.push_back(idx);
        }
      }
    }
  }

  // Update map boundaries based on processed points
  md_->local_update_mind_ = update_mind;
  md_->local_update_maxd_ = update_maxd;
  posToIndex(md_->local_update_mind_, md_->local_update_min_);
  posToIndex(md_->local_update_maxd_, md_->local_update_max_);
  
  // Expand global update boundary to include current update
  for (int k = 0; k < 2; ++k) {
    md_->update_mind_[k] = min(update_mind[k], md_->update_mind_[k]);
    md_->update_maxd_[k] = max(update_maxd[k], md_->update_maxd_[k]);
  }
  posToIndex(md_->update_mind_, md_->update_min_);
  posToIndex(md_->update_maxd_, md_->update_max_);
  boundIndex(md_->update_min_);
  boundIndex(md_->update_max_);
  map_ros_->local_updated_ = true;

  // Process all cached voxels and update their occupancy probabilities
  while (!md_->cache_voxel_.empty()) {
    int adr = md_->cache_voxel_.front();
    md_->cache_voxel_.pop();
    
    // Determine log-odds update based on hit/miss ratio
    double log_odds_update =
        md_->count_hit_[adr] >= md_->count_miss_[adr] ? mp_->prob_hit_log_ : mp_->prob_miss_log_;
    md_->count_hit_[adr] = md_->count_miss_[adr] = 0;
    
    // Initialize unknown voxels with minimum occupancy
    if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)
      md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;

    // Update occupancy with clamping
    double last_occupancy = md_->occupancy_buffer_[adr];
    md_->occupancy_buffer_[adr] =
        std::min(std::max(md_->occupancy_buffer_[adr] + log_odds_update, mp_->clamp_min_log_),
            mp_->clamp_max_log_);
    double now_occupancy = md_->occupancy_buffer_[adr];
    
    // Track voxels that changed from occupied to free for clearing inflation
    if (last_occupancy > mp_->min_occupancy_log_ && now_occupancy < mp_->min_occupancy_log_) {
      md_->occupancy_need_clear_.push_back(addressToIdx(adr));
    }
  }
}

void SDFMap2D::setForceOccGrid(const Eigen::Vector2d& pos)
{
  // Force a grid cell to be occupied (used for debugging or special cases)
  Eigen::Vector2i idx;
  posToIndex(pos, idx);
  int adr = toAddress(idx);
  md_->occupancy_buffer_[adr] = mp_->clamp_max_log_;
}

Eigen::Vector2d SDFMap2D::closetPointInMap(
    const Eigen::Vector2d& pt, const Eigen::Vector2d& camera_pt)
{
  // Find the closest point within map boundaries along the ray from camera to point
  Eigen::Vector2d diff = pt - camera_pt;
  Eigen::Vector2d max_tc = mp_->map_max_boundary_ - camera_pt;
  Eigen::Vector2d min_tc = mp_->map_min_boundary_ - camera_pt;
  double min_t = std::numeric_limits<double>::max();
  
  // Check intersection with all boundary planes
  for (int i = 0; i < 2; ++i) {
    if (fabs(diff[i]) > 0) {
      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t)
        min_t = t1;
      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t)
        min_t = t2;
    }
  }
  return camera_pt + (min_t - 1e-3) * diff;
}

template <typename F_get_val, typename F_set_val>
void SDFMap2D::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim)
{
  // Fast marching method for computing Euclidean Signed Distance Field (ESDF)
  // This implements the algorithm from Felzenszwalb & Huttenlocher
  int v[mp_->map_voxel_num_(dim)];
  double z[mp_->map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  // Build lower envelope of parabolas
  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  // Query lower envelope to get distance values
  k = start;
  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q) k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

void SDFMap2D::updateESDFMap()
{
  // Update Euclidean Signed Distance Field within local bounds
  Eigen::Vector2i min_esdf = md_->local_bound_min_;
  Eigen::Vector2i max_esdf = md_->local_bound_max_;

  // First pass: compute distance transform along Y-axis
  if (mp_->optimistic_) {
    // Optimistic mode: only consider known occupied cells
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
      fillESDF(
          [&](int y) {
            int adr = toAddress(x, y);
            return md_->occupancy_buffer_inflate_[adr] == 1 ? 0 :
                                                              std::numeric_limits<double>::max();
          },
          [&](int y, double val) { md_->tmp_buffer_[toAddress(x, y)] = val; }, min_esdf[1],
          max_esdf[1], 1);
    }
  }
  else {
    // Conservative mode: consider both occupied and unknown cells as obstacles
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
      fillESDF(
          [&](int y) {
            int adr = toAddress(x, y);
            return (md_->occupancy_buffer_inflate_[adr] == 1 ||
                       md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3) ?
                       0 :
                       std::numeric_limits<double>::max();
          },
          [&](int y, double val) { md_->tmp_buffer_[toAddress(x, y)] = val; }, min_esdf[1],
          max_esdf[1], 1);
    }
  }

  // Second pass: compute distance transform along X-axis
  for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
    fillESDF([&](int x) { return md_->tmp_buffer_[toAddress(x, y)]; },
        [&](int x, double val) {
          md_->distance_buffer_[toAddress(x, y)] = mp_->resolution_ * std::sqrt(val);
        },
        min_esdf[0], max_esdf[0], 0);
  }

  // Compute signed distance field if requested
  if (mp_->signed_dist_) {
    // Compute negative distances (inside obstacles)
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
      fillESDF(
          [&](int y) {
            int adr = toAddress(x, y);
            return md_->occupancy_buffer_inflate_[adr] == 0 ? 0 :
                                                              std::numeric_limits<double>::max();
          },
          [&](int y, double val) { md_->tmp_buffer_[toAddress(x, y)] = val; }, min_esdf[1],
          max_esdf[1], 1);
    }

    for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
      fillESDF([&](int x) { return md_->tmp_buffer_[toAddress(x, y)]; },
          [&](int x, double val) {
            md_->distance_buffer_neg_[toAddress(x, y)] = mp_->resolution_ * std::sqrt(val);
          },
          min_esdf[0], max_esdf[0], 0);
    }

    // Combine positive and negative distances to create signed distance field
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        int idx = toAddress(x, y);
        if (md_->distance_buffer_neg_[idx] > 0.0)
          md_->distance_buffer_[idx] += (-md_->distance_buffer_neg_[idx] + mp_->resolution_);
      }
    }
  }
}

void SDFMap2D::clearAndInflateLocalMap()
{
  // Clear previous inflation and inflate obstacles in local map area
  int inf_step = ceil(mp_->obstacles_inflation_ / mp_->resolution_);
  vector<Eigen::Vector2i> inf_pts;
  Eigen::Vector2i range_min = md_->local_update_min_;
  Eigen::Vector2i range_max = md_->local_update_max_;

  // Clear inflation for voxels that changed from occupied to free
  for (auto idx : md_->occupancy_need_clear_) {
    inflatePoint(idx, inf_step, inf_pts);
    for (auto& inf_pt : inf_pts) {
      int idx_inf = toAddress(inf_pt(0), inf_pt(1));
      if (idx_inf >= 0 && idx_inf < mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1)) {
        md_->occupancy_buffer_inflate_[idx_inf] = 0;
      }
    }
  }

  // Inflate newly occupied voxels
  for (int x = range_min(0); x <= range_max(0); ++x)
    for (int y = range_min(1); y <= range_max(1); ++y) {
      int id1 = toAddress(x, y);
      if (md_->occupancy_buffer_[id1] > mp_->min_occupancy_log_) {
        inflatePoint(Eigen::Vector2i(x, y), inf_step, inf_pts);

        for (auto& inf_pt : inf_pts) {
          int idx_inf = toAddress(inf_pt(0), inf_pt(1));
          if (idx_inf >= 0 && idx_inf < mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1)) {
            md_->occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }
    }
}

double SDFMap2D::getDistWithGrad(const Eigen::Vector2d& pos, Eigen::Vector2d& grad)
{
  if (!isInMap(pos)) {
    grad.setZero();
    return 0;
  }

  // Compute distance and gradient using bilinear interpolation
  Eigen::Vector2d pos_m = pos - 0.5 * mp_->resolution_ * Eigen::Vector2d::Ones();
  Eigen::Vector2i idx;
  posToIndex(pos_m, idx);
  Eigen::Vector2d idx_pos, diff;
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_->resolution_inv_;

  // Sample distance values from 2x2 neighborhood
  double values[2][2];
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++) {
      Eigen::Vector2i current_idx = idx + Eigen::Vector2i(x, y);
      values[x][y] = getDistance(current_idx);
    }

  // Bilinear interpolation for distance
  double v0 = (1 - diff[0]) * values[0][0] + diff[0] * values[1][0];
  double v1 = (1 - diff[0]) * values[0][1] + diff[0] * values[1][1];
  double dist = (1 - diff[1]) * v0 + diff[1] * v1;

  // Compute gradient using finite differences
  grad[1] = (v1 - v0) * mp_->resolution_inv_;
  grad[0] = (1 - diff[1]) * (values[1][0] - values[0][0]) + diff[1] * (values[1][1] - values[0][1]);
  grad[0] *= mp_->resolution_inv_;

  return dist;
}

}  // namespace apexnav_planner
