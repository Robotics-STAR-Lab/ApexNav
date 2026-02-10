/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <lkh_mtsp_solver/srv/solve_mtsp.hpp>
#include <plan_env/map_ros.h>
#include <path_searching/kino_astar.h>
#include <trajectory_manager/optimizer.h>
#include <cmath>

using namespace Eigen;

namespace apexnav_planner {

ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(rclcpp::Node::SharedPtr node)
{
  node_ = node;

  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);
  sdf_map_->initMap(node_);
  object_map2d_ = sdf_map_->object_map2d_;

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, node_));
  path_finder_.reset(new Astar2D);
  path_finder_->init(node_, sdf_map_);

  // Initialize exploration data and parameter containers
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // Load exploration parameters from ROS2 parameter server
  if (!node_->has_parameter("exploration/policy")) {
    node_->declare_parameter("exploration/policy", 0);
  }
  if (!node_->has_parameter("exploration/sigma_threshold")) {
    node_->declare_parameter("exploration/sigma_threshold", 0.030);
  }
  if (!node_->has_parameter("exploration/max_to_mean_threshold")) {
    node_->declare_parameter("exploration/max_to_mean_threshold", 1.2);
  }
  if (!node_->has_parameter("exploration/max_to_mean_percentage")) {
    node_->declare_parameter("exploration/max_to_mean_percentage", 0.95);
  }
  if (!node_->has_parameter("exploration/tsp_dir")) {
    node_->declare_parameter("exploration/tsp_dir", std::string("null"));
  }

  ep_->policy_mode_ = node_->get_parameter("exploration/policy").as_int();
  ep_->sigma_threshold_ = node_->get_parameter("exploration/sigma_threshold").as_double();
  ep_->max_to_mean_threshold_ = node_->get_parameter("exploration/max_to_mean_threshold").as_double();
  ep_->max_to_mean_percentage_ = node_->get_parameter("exploration/max_to_mean_percentage").as_double();
  ep_->tsp_dir_ = node_->get_parameter("exploration/tsp_dir").as_string();

  // Get map parameters for ray casting initialization
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);
  tsp_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  tsp_client_ = node_->create_client<lkh_mtsp_solver::srv::SolveMTSP>(
      "/solve_tsp", rmw_qos_profile_services_default, tsp_cb_group_);

  // Initialize KinoAstar and GCopter for real-world trajectory planning
  kinoastar_.reset(new KinoAstar(node_, sdf_map_));
  kinoastar_->init();

  Config gcopter_config(node_);
  gcopter_.reset(new Gcopter(gcopter_config, node_, sdf_map_, kinoastar_));

  RCLCPP_INFO(node_->get_logger(), "[ExplorationManager] KinoAstar and GCopter initialized for real-world mode");
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw)
{
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  auto t1 = node_->get_clock()->now();
  auto t2 = t1;

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================
  if (!object_clouds.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
        return SEARCH_BEST_OBJECT;
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Navigation Mode (Over Depth)] Get over depth object cloud");
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (object_clouds.size() >= 1)
    top_object_cloud = object_clouds[0];

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Maybe no passable frontier.");

    // Try suspicious objects as backup
    if (!top_object_cloud->points.empty() &&
        searchObjectPath(pos, top_object_cloud, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_SUSPICIOUS_OBJECT;
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "search exterme case!!!");

      // Try extreme object search with relaxed constraints
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Include lower confidence objects in extreme search
      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Try cached over-depth objects as final option
      static auto last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;
      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "No coverable frontier!!");
        return NO_COVERABLE_FRONTIER;
      }
      else {
        RCLCPP_ERROR(node_->get_logger(), "No passable frontier!!");
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;

  // Performance monitoring
  double total_time = (node_->get_clock()->now() - t2).seconds();
  if (total_time > 0.25) {
    RCLCPP_ERROR(node_->get_logger(), "[Plan NBV] Total time %.2lf s too long!!!", total_time);
  }

  return EXPLORATION;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      RCLCPP_WARN(node_->get_logger(), "[Exploration Mode] Find Closest Frontier");
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::SEMANTIC:
      RCLCPP_WARN(node_->get_logger(), "[Exploration Mode] Find Highest Semantic Value Frontier");
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HYBRID:
      RCLCPP_WARN(node_->get_logger(), "[Exploration Mode] Working on Hybrid Mode");
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::TSP_DIST:
      RCLCPP_WARN(node_->get_logger(), "[Exploration Mode] Working on TSP Distance Mode");
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    default:
      RCLCPP_WARN(node_->get_logger(), "[Exploration Mode] Unknown Mode");
      break;
  }
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  double std_dev_threshold = ep_->sigma_threshold_;
  double max_to_mean_threshold = ep_->max_to_mean_threshold_;
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  double std_dev, max_to_mean, mean;
  calcSemanticFrontierInfo(sem_frontiers, std_dev, max_to_mean, mean);

  // Decide between exploitation and exploration based on semantic statistics
  if (std_dev > std_dev_threshold && max_to_mean > max_to_mean_threshold) {
    RCLCPP_WARN(node_->get_logger(), "Exploit the semantic value (TSP)!!");
    vector<Vector2d> high_sem_frontiers;

    // Select high-value frontiers for TSP optimization
    for (auto sem_frontier : sem_frontiers) {
      double auto_max_to_mean_threshold =
          std::max(max_to_mean_threshold, ep_->max_to_mean_percentage_ * max_to_mean);
      if (sem_frontier.semantic_value / mean < auto_max_to_mean_threshold)
        break;
      high_sem_frontiers.push_back(sem_frontier.position);
    }
    findTSPTourPolicy(cur_pos, high_sem_frontiers, next_best_pos, next_best_path);
  }
  else {
    RCLCPP_WARN(node_->get_logger(), "Explore the environment (Closest)!!");
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
  }
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Compute semantic value for each frontier
  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    // Find maximum semantic value in local neighborhood
    double value = sdf_map_->value_map_->getValue(idx);
    for (auto nbr : nbrs) value = std::max(value, sdf_map_->value_map_->getValue(nbr));

    frontier_values.emplace_back(frontier, value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = node_->get_clock()->now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (node_->get_clock()->now() - t1).seconds();
  t1 = node_->get_clock()->now();

  // Initialize ATSP par file
  // Create problem file
  std::ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + std::to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << std::to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  auto par_dir = ep_->tsp_dir_ + "/atsp_tour.atsp";

  auto request = std::make_shared<lkh_mtsp_solver::srv::SolveMTSP::Request>();
  request->prob = 1;

  auto result = tsp_client_->async_send_request(request);
  // Wait for service response without re-spinning the node (avoids executor conflict)
  auto status = result.wait_for(std::chrono::seconds(5));
  if (status != std::future_status::ready) {
    RCLCPP_ERROR(node_->get_logger(), "Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  std::ifstream res_file(ep_->tsp_dir_ + "/atsp_tour.tour");
  std::string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  double tsp_time = (node_->get_clock()->now() - t1).seconds();
  RCLCPP_WARN(node_->get_logger(), "[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  if (!object_cloud || object_cloud->empty()) {
    RCLCPP_ERROR(node_->get_logger(), "[Bug] Object cloud is empty.");
    return Vector2d(-1000.0, -1000.0);
  }

  // Filter out NaN/Inf points
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> valid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& pt : object_cloud->points) {
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
      valid_cloud->points.push_back(pt);
    }
  }
  if (valid_cloud->empty()) {
    RCLCPP_ERROR(node_->get_logger(), "[Bug] All object cloud points are NaN/Inf.");
    return Vector2d(-1000.0, -1000.0);
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(valid_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  // Check if search point is valid
  if (!std::isfinite(cur_pt.x) || !std::isfinite(cur_pt.y) || !std::isfinite(cur_pt.z)) {
    RCLCPP_ERROR(node_->get_logger(), "[Bug] Start point has NaN/Inf coordinates.");
    return Vector2d(-1000.0, -1000.0);
  }

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    RCLCPP_ERROR(node_->get_logger(), "[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = valid_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        RCLCPP_WARN(node_->get_logger(), "%s", debug_msg.c_str());
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  const double max_search_time = 0.2;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Find nearest accessible point in object cloud
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error indicator from findNearestObjectPoint

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85 };
  const std::vector<std::string> debug_messages = { "I'm going to the object! dist = 0.5m!",
    "I'm going to the object! dist = 0.70m!", "I'm going to the object! dist = 0.85m!" };

  // Attempt path planning with each safety distance
  for (size_t i = 0; i < distances.size(); ++i) {
    if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
            refined_pos, refined_path, debug_messages[i])) {
      return true;
    }
  }

  RCLCPP_ERROR(node_->get_logger(), "Failed to find object path.");
  return false;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on semantic values and reachability
  sem_frontiers.clear();

  for (auto& frontier : frontiers) {
    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double value = sdf_map_->value_map_->getValue(idx);

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(idx) == 1 ||
          sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED)
        continue;
      value = std::max(value, sdf_map_->value_map_->getValue(nbr));
    }
    sem_frontier.semantic_value = value;

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = std::max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

bool ExplorationManager::planTrajectory(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl)
{
  if (!gcopter_ || !kinoastar_) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
        "[ExplorationManager] GCopter or KinoAstar not initialized for real-world mode");
    return false;
  }

  Eigen::VectorXd goal_state, current_state;
  Vector3d control = ctrl;
  goal_state = end;
  current_state = start;

  // Kinodynamic A* search
  kinoastar_->reset();
  kinoastar_->search(goal_state, current_state, control);
  kinoastar_->getKinoNode();

  if (kinoastar_->has_path_) {
    kinoastar_->kinoastarFlatPathPub(kinoastar_->flat_trajs_);
    gcopter_->minco_plan();
    std::vector<Trajectory<7, 3>> final_trajes = gcopter_->final_trajes;
    gcopter_->mincoPathPub(gcopter_->final_trajes, gcopter_->final_singuls);
    return true;
  }

  return false;
}

}  // namespace apexnav_planner
