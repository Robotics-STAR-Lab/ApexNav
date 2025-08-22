#ifndef _ASTAR2D_H
#define _ASTAR2D_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <ros/console.h>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include <plan_env/sdf_map2d.h>
#include <boost/functional/hash.hpp>
#include <queue>
#include <path_searching/matrix_hash.h>

namespace apexnav_planner {
class Node2D {
public:
  Eigen::Vector2i index;
  Eigen::Vector2d position;
  double g_score, f_score;
  Node2D* parent;

  Node2D() : index(0, 0), position(0.0, 0.0), g_score(0.0), f_score(0.0), parent(nullptr)
  {
  }
  ~Node2D() = default;
};
typedef Node2D* Node2DPtr;

class NodeComparator2D {
public:
  bool operator()(Node2DPtr node1, Node2DPtr node2)
  {
    return node1->f_score > node2->f_score;
  }
};

class Astar2D {
public:
  Astar2D() = default;
  ~Astar2D();
  enum { REACH_END = 1, NO_PATH = 2 };
  enum SAFETY_MODE { NORMAL = 0, OPTIMISTIC = 1, EXTREME = 2 };
  void init(ros::NodeHandle& nh, const SDFMap2D::Ptr& sdf_map);
  void reset();
  int astarSearch(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
      double success_dist = 0.1, double max_time = 0.01, int safety_mode = SAFETY_MODE::NORMAL);
  void setResolution(const double& res);
  static double pathLength(const std::vector<Eigen::Vector2d>& path);
  std::vector<Eigen::Vector2d> generateSteps(Eigen::Vector2d pos);

  std::vector<Eigen::Vector2d> getPath();
  std::vector<Eigen::Vector2d> getVisited();
  double getEarlyTerminateCost();

  double lambda_heu_;

private:
  void backtrack(const Node2DPtr& end_node, const Eigen::Vector2d& end);
  void posToIndex(const Eigen::Vector2d& pt, Eigen::Vector2i& idx);
  double getDiagHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);
  double getManhHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);
  double getEuclHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);
  bool checkPointSafety(const Eigen::Vector2d& pos, int safety_mode);

  // main data structure
  std::vector<Node2DPtr> path_node_pool_;
  int use_node_num_, iter_num_;
  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> open_set_;
  std::unordered_map<Eigen::Vector2i, Node2DPtr, matrix_hash<Eigen::Vector2i>> open_set_map_;
  std::unordered_map<Eigen::Vector2i, int, matrix_hash<Eigen::Vector2i>> close_set_map_;
  std::vector<Eigen::Vector2d> path_nodes_;
  double early_terminate_cost_;

  SDFMap2D::Ptr sdf_map_;

  // parameter
  double margin_;
  int allocate_num_;
  double resolution_, inv_resolution_;
  Eigen::Vector2d map_size_2d_, origin_;
};

}  // namespace apexnav_planner

#endif
