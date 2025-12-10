#ifndef _KINO_ASTAR_H_
#define _KINO_ASTAR_H_

#include "plan_env/sdf_map2d.h"
#include <path_searching/traj_representation.h>
#include <path_searching/matrix_hash.h>

#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <queue>
#include <algorithm>

#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include "tf/transform_datatypes.h"

#define inf 1 >> 30

// For comparing f
class NodeComparator {
public:
  template <class NodePtr>
  bool operator()(NodePtr node1, NodePtr node2)
  {
    return node1->f_score > node2->f_score;
  }
};

template <class NodePtr>
class NodeHashTable {
private:
  /* data */

  // std::unordered_map<Eigen::Vector2i, NodePtr, matrix_hash<Eigen::Vector2i>> data_2d_;
  // std::unordered_map<Eigen::Vector3i, NodePtr, matrix_hash<Eigen::Vector3i>> data_3d_;
  std::unordered_map<Eigen::Vector4i, NodePtr, matrix_hash<Eigen::Vector4i>> data_4d_;

public:
  NodeHashTable(/* args */)
  {
  }
  ~NodeHashTable()
  {
  }
  // : for 2d vehicle planning
  // void insert(Eigen::Vector2i idx, NodePtr node) {
  //   data_2d_.insert(std::make_pair(idx, node));
  // }
  // //for 3d vehicle planning
  // void insert(Eigen::Vector2i idx, int yaw_idx, NodePtr node) {
  //   data_3d_.insert(std::make_pair(Eigen::Vector3i(idx(0), idx(1), yaw_idx), node));
  // }
  // void insert(Eigen::Vector3i idx,NodePtr node ){
  //   data_3d_.insert(std::make_pair(idx,node));
  // }
  void insert(Eigen::Vector2i idx, int yaw_idx, int mani_angle_idx, NodePtr node)
  {
    data_4d_.insert(std::make_pair(Eigen::Vector4i(idx(0), idx(1), yaw_idx, mani_angle_idx), node));
  }

  // NodePtr find(Eigen::Vector2i idx) {
  //   auto iter = data_2d_.find(idx);
  //   return iter == data_2d_.end() ? NULL : iter->second;
  // }
  // NodePtr find(Eigen::Vector2i idx, int yaw_idx) {
  //   auto iter = data_3d_.find(Eigen::Vector3i(idx(0), idx(1), yaw_idx));
  //   return iter == data_3d_.end() ? NULL : iter->second;
  // }
  NodePtr find(Eigen::Vector2i idx, int yaw_idx, int mani_angle_idx)
  {
    auto iter = data_4d_.find(Eigen::Vector4i(idx[0], idx[1], yaw_idx, mani_angle_idx));
    return iter == data_4d_.end() ? NULL : iter->second;
  }

  void clear()
  {
    // data_2d_.clear();
    // data_3d_.clear();
    data_4d_.clear();
  }
};

namespace apexnav_planner {
class KinoAstar {
public:
  enum { REACH_HORIZON = 1, REACH_END = 2, NO_PATH = 3, REACH_END_BUT_SHOT_FAILS = 4 };
  KinoAstar(ros::NodeHandle nh, const SDFMap2D::Ptr& map)
  {
    nh_ = nh;
    this->map_ = map;
    start_state_.resize(5);
    end_state_.resize(5);
  }

  ~KinoAstar()
  {
    for (int i = 0; i < allocate_num_; i++) delete path_node_pool_[i];
  }
  void init();
  void reset();
  int search(
      const Eigen::VectorXd& end_state, Eigen::VectorXd& start_state, Eigen::Vector3d& init_ctrl);
  void getKinoNode();
  Eigen::Vector4d evaluatePos(const double& t);
  void kinoastarFlatPathPub(const std::vector<FlatTrajData> flat_trajs);
  bool isCollisionPosYaw(const Eigen::Vector2d& pos, const double& yaw);

  double length_;
  double width_;
  double height_;
  double wheel_base_;
  SDFMap2D::Ptr map_;
  // Terminal path
  std::vector<PathNodePtr> path_nodes_;
  std::vector<FlatTrajData> flat_trajs_;
  std::vector<Eigen::Vector4d> sample_traj_;
  Eigen::VectorXd start_state_;  // x, y, yaw, steering_angle, velocity
  Eigen::Vector3d start_ctrl_;
  Eigen::VectorXd end_state_;
  double totalTrajTime;
  bool has_path_ = false;

private:
  bool checkCollision(double x, double y, double z);
  int yawToIndex(const double& yaw);
  int getSingularity(const double& vel);
  double getHeu(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2);
  bool isShotSuccess(const Eigen::Vector4d& state1, const Eigen::Vector4d& state2);
  double computeShotTraj(const Eigen::Vector3d& state1, const Eigen::Vector3d& state2,
      const int shotptrind, std::vector<Eigen::Vector3d>& path_list, double& len);
  void retrievePath(const PathNodePtr& end_node);
  void stateTransit(
      const Eigen::Vector4d& state0, const Eigen::Vector3d& ctrl_input, Eigen::Vector4d& state1);

  void getSampleTraj();
  void getTrajsWithTime();
  double evaluateDistance(const Eigen::Vector2d& state1, const Eigen::Vector2d& state2);
  double evaluateDuration(const double& length, const double& startV, const double& endV);
  double evaluateLength(const double& curt, const double& locallength, const double& localtime,
      const double& startV, const double& endV);
  void getFlatState(const Eigen::Vector4d& state, const Eigen::Vector2d& control_input,
      const int& singul, Eigen::MatrixXd& flat_state);
  void getFlatState(const double& x, const double& y, const double& angle, const double& v,
      const double& maniangle, const Eigen::Vector3d& control_input, const int& singul,
      Eigen::MatrixXd& flat_state);

  void nodeVis(const Eigen::Vector3d& state);
  double normalizeAngle(const double& angle);

  ros::NodeHandle nh_;

  ros::Publisher expandNodes_pub_;
  ros::Publisher kinoastarFlatPathPub_, kinoastarFlatTrajPub_;

  std::vector<PathNodePtr> path_node_pool_;
  NodeHashTable<PathNodePtr> expanded_nodes_;
  std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> open_set_;

  double yaw_resolution_;  // Yaw angle discretization interval
  double inv_yaw_resolution_;
  double lambda_heu_;       // Greedy search coefficient
  int allocate_num_;        // Total number of grids
  double max_search_time_;  // Maximum search time
  double step_arc_;         // Forward search displacement
  double checkl_;           // Oneshot check intermediate point collision discretization interval
  double grid_interval_;
  int check_num_;  // Used to check collision when A* expands states
  double oneshot_range_;
  double sampletime_;  // Sampling time given to backend

  int use_node_num_;  // number of nodes used
  int iter_num_;

  double yaw_origin_;          // used to convert yaw to index
  double non_siguav_;          // minimum speed considered non-zero
  double collision_interval_;  // collision checking interval

  // Commented parameters exist in the original but are unused here
  double max_vel_;
  double max_acc_;
  double max_cur_;
  double max_steer_;

  // OMPL: this implementation explicitly computes all 48 Reeds-Shepp paths and returns
  // the shortest valid solution. (Radius values are shown in parentheses)
  std::vector<ompl::base::StateSpacePtr> shotptr_s;
  int shotptrindex;
  int shotptrsize;

  // runtime flags / process variables
  bool is_shot_succ_ = false;

  std::vector<double> shot_lengthList;  // lengths of the segments
  std::vector<double> shot_timeList;    // times for trapezoidal velocity calculation
  std::vector<int> shotindex;           // indices of turning points in arrays
  std::vector<int> shot_SList;          // store direction for each segment
};

inline double KinoAstar::normalizeAngle(const double& angle)
{
  double nor_angle = angle;
  nor_angle -= (angle >= M_PI) * 2 * M_PI;
  nor_angle += (angle <= -M_PI) * 2 * M_PI;
  return nor_angle;
}
}  // namespace apexnav_planner
#endif