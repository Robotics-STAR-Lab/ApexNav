#include <path_searching/astar2d.h>
#include <sstream>

using namespace std;
using namespace Eigen;

namespace apexnav_planner {
Astar2D::~Astar2D()
{
  for (int i = 0; i < allocate_num_; i++) delete path_node_pool_[i];
}

void Astar2D::init(ros::NodeHandle& nh, const SDFMap2D::Ptr& sdf_map)
{
  nh.param("astar/resolution_astar", resolution_, -1.0);
  nh.param("astar/lambda_heu", lambda_heu_, -1.0);
  allocate_num_ = 1000000;

  this->sdf_map_ = sdf_map;

  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  sdf_map_->getRegion(origin_, map_size_2d_);
  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_2d_.transpose() << endl;

  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) path_node_pool_[i] = new Node2D;
  use_node_num_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;
}

void Astar2D::reset()
{
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();

  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = nullptr;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
}

void Astar2D::setResolution(const double& res)
{
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

int Astar2D::astarSearch(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
    double success_dist, double max_time, int safety_mode)
{
  Node2DPtr cur_node = path_node_pool_[0];
  cur_node->parent = nullptr;
  cur_node->position = start_pt;
  posToIndex(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector2i end_index;
  posToIndex(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();

  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reach_end =
        abs(cur_node->index(0) - end_index(0)) <= 1 && abs(cur_node->index(1) - end_index(1)) <= 1;
    if ((cur_node->position - end_pt).norm() < success_dist)
      reach_end = true;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > max_time) {
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      // ROS_WARN("Astar Long Time");
      return NO_PATH;
    }

    open_set_.pop();
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector2d cur_pos = cur_node->position;
    Eigen::Vector2d nbr_pos;

    std::vector<Eigen::Vector2d> steps = generateSteps(cur_pos);
    for (auto step : steps) {
      nbr_pos = cur_pos + step;

      // Skip safety raycast if still near start to avoid immediate termination
      if ((nbr_pos - start_pt).norm() > 0.25) {
        // Check safety
        if (!checkPointSafety(nbr_pos, safety_mode))
          continue;

        bool safe = true;
        Vector2d dir = nbr_pos - cur_pos;
        double len = dir.norm();
        dir.normalize();
        for (double l = 0.025; l < len; l += 0.025) {
          Vector2d ckpt = cur_pos + l * dir;
          if (!checkPointSafety(ckpt, safety_mode)) {
            safe = false;
            break;
          }
        }
        if (!safe)
          continue;
      }

      // Check not in close set
      Eigen::Vector2i nbr_idx;
      posToIndex(nbr_pos, nbr_idx);
      if (close_set_map_.find(nbr_idx) != close_set_map_.end())
        continue;

      Node2DPtr neighbor;
      double tmp_g_score = step.norm() + cur_node->g_score;
      auto node_iter = open_set_map_.find(nbr_idx);
      if (node_iter == open_set_map_.end()) {
        neighbor = path_node_pool_[use_node_num_];
        use_node_num_ += 1;
        if (use_node_num_ == allocate_num_) {
          cout << "run out of node pool." << endl;
          return NO_PATH;
        }
        neighbor->index = nbr_idx;
        neighbor->position = nbr_pos;
      }
      else if (tmp_g_score < node_iter->second->g_score) {
        neighbor = node_iter->second;
      }
      else
        continue;

      neighbor->parent = cur_node;
      neighbor->g_score = tmp_g_score;
      neighbor->f_score = tmp_g_score + lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
      open_set_.push(neighbor);
      open_set_map_[nbr_idx] = neighbor;
    }
  }
  return NO_PATH;
}

std::vector<Eigen::Vector2d> Astar2D::generateSteps(Eigen::Vector2d pos)
{
  vector<Eigen::Vector2d> steps;

  // Normal Astar Step
  // for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
  //   for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_) {
  //     Eigen::Vector2d step;
  //     step << dx, dy;
  //     if (step.norm() < 1e-3)
  //       continue;
  //     steps.push_back(step);
  //   }

  // Habitat-like stepping (12 directions)
  const double step_length = 0.25;
  const double angle_increment = M_PI / 6;

  for (int i = 0; i < 12; ++i) {
    double angle = i * angle_increment;
    Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
    steps.push_back(step);
  }
  return steps;
}

double Astar2D::getDiagHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Diagonal distance heuristic for 2D
  return tie_breaker * (sqrt(2.0) * min(dx, dy) + abs(dx - dy));
}

double Astar2D::getManhHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Manhattan distance heuristic for 2D
  return tie_breaker * (dx + dy);
}

double Astar2D::getEuclHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Euclidean distance heuristic for 2D
  return tie_breaker * (x2 - x1).norm();
}

void Astar2D::backtrack(const Node2DPtr& end_node, const Eigen::Vector2d& end)
{
  path_nodes_.push_back(end);
  path_nodes_.push_back(end_node->position);
  Node2DPtr cur_node = end_node;
  while (cur_node->parent != nullptr) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node->position);
  }
  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector2d> Astar2D::getVisited()
{
  vector<Eigen::Vector2d> visited;
  for (int i = 0; i < use_node_num_; ++i) visited.push_back(path_node_pool_[i]->position);
  return visited;
}

void Astar2D::posToIndex(const Eigen::Vector2d& pt, Eigen::Vector2i& idx)
{
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

std::vector<Eigen::Vector2d> Astar2D::getPath()
{
  return path_nodes_;
}

double Astar2D::pathLength(const vector<Eigen::Vector2d>& path)
{
  double length = 0.0;
  if (path.size() < 2)
    return length;
  for (int i = 0; i < (int)path.size() - 1; ++i) length += (path[i + 1] - path[i]).norm();
  return length;
}

bool Astar2D::checkPointSafety(const Eigen::Vector2d& pos, int safety_mode)
{
  // Outside map bounds is always unsafe
  if (!sdf_map_->isInMap(pos))
    return false;

  // EXTREME: allow any position inside the map (bypass occupancy checks)
  if (safety_mode == SAFETY_MODE::EXTREME)
    return true;

  // Occupancy checks
  const auto occ = sdf_map_->getOccupancy(pos);
  // If inflated occupancy marks collision, or cell is definitely occupied -> unsafe
  if (sdf_map_->getInflateOccupancy(pos) == 1 || occ == SDFMap2D::OCCUPIED)
    return false;

  // In NORMAL mode, treat unknown as unsafe. In OPTIMISTIC, unknown is allowed.
  if (occ == SDFMap2D::UNKNOWN && safety_mode == SAFETY_MODE::NORMAL)
    return false;

  return true;
}
}  // namespace apexnav_planner
