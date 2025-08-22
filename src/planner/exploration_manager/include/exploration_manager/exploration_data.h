#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::pair;
using std::vector;

namespace apexnav_planner {
struct FSMData {
  FSMData()
  {
    trigger_ = false;
    have_odom_ = false;
    have_confidence_ = false;
    have_finished_ = false;
    state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_ACTION", "WAIT_ACTION_FINISH", "PUB_ACTION",
      "FINISH" };

    odom_pos_ = Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Vector3d::Zero();
    start_yaw_ = Vector3d::Zero();
    last_start_pos_ = Vector3d(-100, -100, -100);
    last_next_pos_ = Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    local_pos_ = Vector2d(0, 0);
  }
  // FSM data
  bool trigger_, have_odom_, have_confidence_;
  bool have_finished_;
  vector<string> state_str_;
  vector<Vector2d> traveled_path_;

  // odometry state
  Eigen::Vector3d odom_pos_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_yaw_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;

  int final_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  vector<Vector3d> stucking_points_;

  Vector2d local_pos_;
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  vector<Vector2d> action_steps_;
};

struct ExplorationData {
  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
  }
  vector<vector<Vector2d>> frontiers_, dormant_frontiers_;
  vector<Vector2d> frontier_averages_, dormant_frontier_averages_;
  vector<vector<Vector2d>> objects_;
  vector<Vector2d> object_averages_;
  vector<int> object_labels_;
  Vector2d next_pos_;
  vector<Vector2d> next_best_path_;
  vector<Vector2d> tsp_tour_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST };
  // params
  int policy_mode_;
  double sigma_threshold_, max_to_mean_threshold_, max_to_mean_percentage_;
  std::string tsp_dir_;
};

}  // namespace apexnav_planner

#endif