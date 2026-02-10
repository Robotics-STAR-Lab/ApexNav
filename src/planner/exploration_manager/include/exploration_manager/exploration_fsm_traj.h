#ifndef _EXPLORATION_FSM_REAL_H_
#define _EXPLORATION_FSM_REAL_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <memory>
#include <string>
#include <vector>
#include <chrono>

// ROS2 core
#include <rclcpp/rclcpp.hpp>

// ROS2 message types
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/empty.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <trajectory_manager/msg/poly_traj.hpp>

namespace apexnav_planner {

// Forward declarations
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;
struct LocalTrajectory;

namespace FSMConstantsReal {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 0.25;
constexpr double REPLAN_CHECK_DURATION = 0.1;  // Check if replan needed

// Trajectory execution
constexpr double TRAJECTORY_EXECUTION_TIMEOUT = 10.0;  // Max time to execute trajectory
constexpr double GOAL_REACH_THRESHOLD = 0.15;         // Distance to consider goal reached
constexpr double REPLAN_DISTANCE_THRESHOLD = 0.5;     // Trigger replan if deviate too much

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;
constexpr double REACH_DISTANCE = 0.20;
constexpr double SOFT_REACH_DISTANCE = 0.45;
constexpr double LOCAL_DISTANCE = 0.80;
constexpr double FORWARD_DISTANCE = 0.15;
constexpr double FORCE_DORMANT_DISTANCE = 0.35;
constexpr double MIN_SAFE_DISTANCE = 0.15;

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;
constexpr int MAX_REPLAN_FAILURES = 200;  // Max consecutive replan failures (~2s at 10ms interval)

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;
}  // namespace FSMConstantsReal

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

// Real-world FSM states (using class enum to avoid conflicts)
namespace RealFSM {
  enum class State {
    INIT,
    WAIT_TRIGGER,
    PLAN_TRAJ,           // Plan continuous trajectory
    EXEC_TRAJ,           // Executing trajectory
    REPLAN,              // Replanning during execution
    FINISH
  };

  enum class Result {
    EXPLORE,
    SEARCH_OBJECT,
    STUCKING,
    NO_FRONTIER,
    REACH_OBJECT,
    MANUAL_STOP
  };
}

// Trajectory planner return codes
enum class TrajPlannerResult {
  FAILED = 0,        // Trajectory planning failed
  SUCCESS = 1,       // Trajectory planned successfully
  MISSION_COMPLETE = 2  // Mission completed (no frontier or reached object)
};

// Real-world exploration FSM for continuous trajectory execution
class ExplorationFSMReal {
private:
  /* Planning Utils */
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<FastPlannerManager> planner_manager_;
  std::shared_ptr<ExplorationManager> expl_manager_;
  std::shared_ptr<PlanningVisualization> visualization_;

  std::shared_ptr<FSMParam> fp_;
  std::shared_ptr<FSMData> fd_;
  RealFSM::State state_;

  /* ROS2 Utils */
  rclcpp::TimerBase::SharedPtr exec_timer_, frontier_timer_, safety_timer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr trigger_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr confidence_threshold_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr traj_finish_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr habitat_state_sub_;

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr ros_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr expl_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr expl_result_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr action_pub_;

  // Real-world specific: trajectory control publishers
  rclcpp::Publisher<trajectory_manager::msg::PolyTraj>::SharedPtr poly_traj_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr stop_pub_;

  /* Trajectory execution status */
  // Trajectory state is tracked in fd_->static_state_
  int replan_fail_count_ = 0;

  /* Exploration Planner */
  TrajPlannerResult callTrajectoryPlanner();
  void polyTraj2ROSMsg(const LocalTrajectory& local_traj, trajectory_manager::msg::PolyTraj& poly_msg);
  void selectLocalTarget(const Eigen::Vector2d& current_pos, const std::vector<Eigen::Vector2d>& path,
      const double& local_distance, Eigen::Vector2d& target_pos, double& target_yaw);

  // Safety and stuck detection
  void emergencyStop();
  bool checkNeedReplan();
  bool checkStuckCondition();
  double computePathCost(const std::vector<Eigen::Vector2d>& path);

  /* Helper functions */
  bool updateFrontierAndObject();
  void transitState(RealFSM::State new_state, std::string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS2 callbacks */
  void FSMCallback();
  void safetyCallback();
  void frontierCallback();
  void triggerCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void goalCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void confidenceThresholdCallback(const std_msgs::msg::Float64::SharedPtr msg);
  void trajectoryFinishCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void habitatStateCallback(const std_msgs::msg::Int32::SharedPtr msg);

public:
  ExplorationFSMReal() = default;
  ~ExplorationFSMReal() = default;

  void init(rclcpp::Node::SharedPtr node);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSMReal::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}

}  // namespace apexnav_planner

#endif
