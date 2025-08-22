#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// ROS message types
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <visualization_msgs/Marker.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
// Centralized constants for ExplorationFSM (mirrors the style of FSMConstants in fsm2.h)
namespace FSMConstants {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 0.25;

// Robot Action
constexpr double ACTION_DISTANCE = 0.25;
constexpr double ACTION_ANGLE = M_PI / 6.0;

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;       // consider stuck if movement < this
constexpr double REACH_DISTANCE = 0.20;          // reach object distance
constexpr double SOFT_REACH_DISTANCE = 0.45;     // soft reach distance for object
constexpr double LOCAL_DISTANCE = 0.80;          // local target lookahead
constexpr double FORWARD_DISTANCE = 0.15;        // min clearance for marking obstacles
constexpr double FORCE_DORMANT_DISTANCE = 0.35;  // force dormant frontier if very close
constexpr double MIN_SAFE_DISTANCE = 0.15;       // min safe distance to obstacles

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;           // max consecutive stuck actions -> stop
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;  // times next_pos unchanged while stuck

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;  // penalize moving away
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;   // encourage moving closer
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;  // samples along a step for safety cost

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;  // multiply by map resolution
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;
}  // namespace FSMConstants

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum ROS_STATE { INIT, WAIT_TRIGGER, PLAN_ACTION, WAIT_ACTION_FINISH, PUB_ACTION, FINISH };
enum ACTION { STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_DOWN, TURN_UP };
enum HABITAT_STATE { READY, ACTION_EXEC, ACTION_FINISH, EPISODE_FINISH };
enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };
class ExplorationFSM {
private:
  /* Planning Utils */
  ros::NodeHandle nh_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  ROS_STATE state_;

  /* ROS Utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, habitat_state_sub_, confidence_threshold_sub_;
  ros::Publisher action_pub_, ros_state_pub_, expl_state_pub_, expl_result_pub_;
  ros::Publisher robot_marker_pub_;

  /* Action Planner */
  int callActionPlanner();
  int planNextBestAction(Vector2d current_pos, double current_yaw, const vector<Vector2d>& path,
      bool need_safety = true);
  Vector2d selectLocalTarget(
      const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance);
  int decideNextAction(double current_yaw, double target_yaw);
  Vector2d computeBestStep(
      const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos);
  double computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step);
  double computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
      const Vector2d& target_pos, const Vector2d& step);

  /* Helper functions */
  bool updateFrontierAndObject();
  void transitState(ROS_STATE new_state, string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS callbacks */
  void FSMCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32ConstPtr& msg);
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);

public:
  ExplorationFSM() = default;
  ~ExplorationFSM() = default;

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSM::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}
}  // namespace apexnav_planner

#endif