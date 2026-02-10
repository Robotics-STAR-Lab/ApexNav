#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

using namespace std::chrono_literals;

namespace apexnav_planner {

void ExplorationFSMReal::init(rclcpp::Node::SharedPtr node)
{
  node_ = node;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(node_);
  visualization_.reset(new PlanningVisualization(node_));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstantsReal::VIS_SCALE_FACTOR;

  state_ = RealFSM::State::INIT;

  // Load real-world specific parameters
  if (!node_->has_parameter("fsm/replan_time")) {
    node_->declare_parameter("fsm/replan_time", 0.2);
  }
  if (!node_->has_parameter("fsm/replan_traj_end_threshold")) {
    node_->declare_parameter("fsm/replan_traj_end_threshold", 1.0);
  }
  if (!node_->has_parameter("fsm/replan_frontier_change_delay")) {
    node_->declare_parameter("fsm/replan_frontier_change_delay", 0.5);
  }
  if (!node_->has_parameter("fsm/replan_timeout")) {
    node_->declare_parameter("fsm/replan_timeout", 2.0);
  }

  fp_->replan_time_ = node_->get_parameter("fsm/replan_time").as_double();
  fp_->replan_traj_end_threshold_ = node_->get_parameter("fsm/replan_traj_end_threshold").as_double();
  fp_->replan_frontier_change_delay_ = node_->get_parameter("fsm/replan_frontier_change_delay").as_double();
  fp_->replan_timeout_ = node_->get_parameter("fsm/replan_timeout").as_double();

  /* ROS2 Timer */
  exec_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstantsReal::EXEC_TIMER_DURATION),
      std::bind(&ExplorationFSMReal::FSMCallback, this));
  frontier_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstantsReal::FRONTIER_TIMER_DURATION),
      std::bind(&ExplorationFSMReal::frontierCallback, this));
  safety_timer_ = node_->create_wall_timer(
      50ms, std::bind(&ExplorationFSMReal::safetyCallback, this));

  /* ROS2 Subscriber */
  trigger_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/move_base_simple/goal", 10,
      std::bind(&ExplorationFSMReal::triggerCallback, this, std::placeholders::_1));
  goal_sub_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 10,
      std::bind(&ExplorationFSMReal::goalCallback, this, std::placeholders::_1));
  odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      "/odom_world", 10,
      std::bind(&ExplorationFSMReal::odometryCallback, this, std::placeholders::_1));
  confidence_threshold_sub_ = node_->create_subscription<std_msgs::msg::Float64>(
      "/detector/confidence_threshold", 10,
      std::bind(&ExplorationFSMReal::confidenceThresholdCallback, this, std::placeholders::_1));
  habitat_state_sub_ = node_->create_subscription<std_msgs::msg::Int32>(
      "/habitat/state", 10,
      std::bind(&ExplorationFSMReal::habitatStateCallback, this, std::placeholders::_1));

  /* ROS2 Publisher */
  ros_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/state", 10);
  expl_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_state", 10);
  expl_result_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/robot", 10);
  action_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/habitat/plan_action", 10);

  // Real-world trajectory publishers
  poly_traj_pub_ = node_->create_publisher<trajectory_manager::msg::PolyTraj>("/planning/trajectory", 10);
  stop_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/traj_server/stop", 10);

  RCLCPP_INFO(node_->get_logger(), "[ExplorationFSMReal] Initialization complete.");
}

// Main FSM callback for real-world exploration
void ExplorationFSMReal::FSMCallback()
{
  exec_timer_->cancel();

  // Publish current state
  std_msgs::msg::Int32 ros_state_msg;
  ros_state_msg.data = static_cast<int>(state_);
  ros_state_pub_->publish(ros_state_msg);

  switch (state_) {
    case RealFSM::State::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
            "[Real] No odom || No target confidence threshold.");
        exec_timer_->reset();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(RealFSM::State::WAIT_TRIGGER, "FSM");
      break;
    }

    case RealFSM::State::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
          "[Real] Waiting for trigger...");
      break;
    }

    case RealFSM::State::FINISH: {
      fd_->static_state_ = true;
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        // Signal habitat to stop
        std_msgs::msg::Int32 action_msg;
        action_msg.data = 0;  // ACTION::STOP
        action_pub_->publish(action_msg);
      }
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
          "[Real] Finish exploration!");
      break;
    }

    case RealFSM::State::PLAN_TRAJ: {
      // Plan trajectory based on current state
      if (fd_->static_state_) {
        // Robot is static, use current odometry
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      }
      else {
        // Robot is moving, predict future state for smooth replanning
        LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
        double t_plan = (node_->get_clock()->now() - info->start_time).seconds() + fp_->replan_time_;
        t_plan = std::min(t_plan, info->duration);

        Eigen::Vector3d cur_pos = info->traj.getPos(t_plan);
        Eigen::Vector3d cur_vel = info->traj.getVel(t_plan);
        Eigen::Vector3d cur_acc = info->traj.getAcc(t_plan);
        double cur_yaw = atan2(cur_vel(1), cur_vel(0));

        // Calculate yaw rate from acceleration
        Eigen::Matrix2d B_h;
        B_h << 0, -1.0, 1.0, 0;
        Eigen::Vector2d cur_vel_2d = cur_vel.head(2);
        Eigen::Vector2d cur_acc_2d = cur_acc.head(2);
        double norm_vel = cur_vel_2d.norm();
        double help1 = 1.0 / (norm_vel * norm_vel + 1e-2);
        double omega = help1 * cur_acc_2d.transpose() * B_h * cur_vel_2d;

        fd_->start_pt_ = cur_pos;
        fd_->start_vel_ = cur_vel;
        fd_->start_yaw_(0) = cur_yaw;
        fd_->start_yaw_(1) = omega;
      }

      TrajPlannerResult res = callTrajectoryPlanner();

      if (res == TrajPlannerResult::FAILED) {
        ++replan_fail_count_;
        fd_->static_state_ = true;
        if (replan_fail_count_ >= FSMConstantsReal::MAX_REPLAN_FAILURES) {
          RCLCPP_ERROR(node_->get_logger(),
              "[Real] Plan trajectory failed %d times, stopping and waiting for new trigger",
              replan_fail_count_);
          emergencyStop();
          replan_fail_count_ = 0;
          transitState(RealFSM::State::WAIT_TRIGGER, "FSM");
        } else {
          RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 500,
              "[Real] Plan trajectory failed (%d/%d), retrying...",
              replan_fail_count_, FSMConstantsReal::MAX_REPLAN_FAILURES);
        }
      }
      else if (res == TrajPlannerResult::SUCCESS) {
        replan_fail_count_ = 0;
        transitState(RealFSM::State::EXEC_TRAJ, "FSM");
      }
      else {  // TrajPlannerResult::MISSION_COMPLETE
        // If planning from predicted trajectory endpoint failed,
        // retry from actual odom position before giving up
        if (!fd_->static_state_) {
          RCLCPP_WARN(node_->get_logger(),
              "[Real] No frontier from predicted pos, retrying from odom...");
          fd_->static_state_ = true;
          // Don't transition - stay in PLAN_TRAJ to retry on next FSM tick
        }
        else {
          transitState(RealFSM::State::FINISH, "FSM");
        }
      }

      visualize();
      break;
    }

    case RealFSM::State::EXEC_TRAJ: {
      // Publish trajectory and transition to execution monitoring
      double dt = (node_->get_clock()->now() - fd_->newest_traj_.start_time).seconds();
      if (dt > 0) {
        trajectory_manager::msg::PolyTraj poly_msg;
        polyTraj2ROSMsg(fd_->newest_traj_, poly_msg);
        poly_traj_pub_->publish(poly_msg);
        fd_->static_state_ = false;
        transitState(RealFSM::State::REPLAN, "FSM");
      }
      break;
    }

    case RealFSM::State::REPLAN: {
      // Monitor trajectory execution and decide when to replan
      LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
      double t_cur = (node_->get_clock()->now() - info->start_time).seconds();
      double time_to_end = info->duration - t_cur;

      // Replan if trajectory is almost finished
      if (time_to_end < fp_->replan_traj_end_threshold_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        RCLCPP_WARN(node_->get_logger(), "[Real] Replan: traj fully executed");
        exec_timer_->reset();
        return;
      }

      // Replan if frontier changed during exploration
      if (t_cur > fp_->replan_frontier_change_delay_ &&
          fd_->final_result_ == FINAL_RESULT::EXPLORE &&
          expl_manager_->frontier_map2d_->isAnyFrontierChanged()) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        RCLCPP_WARN(node_->get_logger(), "[Real] Replan: frontier changed");
        exec_timer_->reset();
        return;
      }

      // Replan if trajectory timeout
      if (t_cur > fp_->replan_timeout_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        RCLCPP_WARN(node_->get_logger(), "[Real] Replan: time out");
        exec_timer_->reset();
        return;
      }
      break;
    }
  }

  exec_timer_->reset();
}

TrajPlannerResult ExplorationFSMReal::callTrajectoryPlanner()
{
  rclcpp::Time time_r = node_->get_clock()->now() + rclcpp::Duration::from_seconds(fp_->replan_time_);
  updateFrontierAndObject();

  // Call exploration manager to find next best point
  int expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

  // Determine final result based on exploration result
  if (expl_res == EXPL_RESULT::EXPLORATION)
    fd_->final_result_ = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    fd_->final_result_ = FINAL_RESULT::NO_FRONTIER;
  else
    fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;

  // Publish exploration result
  std_msgs::msg::Int32 expl_result_msg;
  expl_result_msg.data = fd_->final_result_;
  expl_result_pub_->publish(expl_result_msg);

  if (fd_->final_result_ == FINAL_RESULT::NO_FRONTIER) {
    RCLCPP_WARN(node_->get_logger(), "[Real] No (passable) frontier");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Select local target from global path
  Eigen::Vector2d goal_pos = expl_manager_->ed_->next_pos_;
  double goal_yaw = 0.0;
  auto path = expl_manager_->ed_->next_best_path_;
  selectLocalTarget(fd_->start_pt_.head(2), path, 4.0, goal_pos, goal_yaw);

  // Check if reached object
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (fd_->start_pt_.head(2) - goal_pos).norm() < 0.25) {
    RCLCPP_ERROR(node_->get_logger(), "[Real] Reach the object successfully!");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Prepare state for trajectory planning
  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control(0.0, 0.0, 0.0);
  double start_vel = Eigen::Vector2d(fd_->start_vel_(0), fd_->start_vel_(1)).norm();
  current_state << fd_->start_pt_(0), fd_->start_pt_(1), fd_->start_yaw_(0), 0.0, start_vel;
  goal_state << goal_pos(0), goal_pos(1), goal_yaw, 0.0, 0.0;

  // Plan trajectory using GCopter
  bool traj_res = expl_manager_->planTrajectory(current_state, goal_state, current_control);
  if (traj_res) {
    auto info = &expl_manager_->gcopter_->local_trajectory_;
    info->start_time = (node_->get_clock()->now() - time_r).seconds() > 0 ?
        node_->get_clock()->now() : time_r;
    fd_->newest_traj_ = expl_manager_->gcopter_->local_trajectory_;
    return TrajPlannerResult::SUCCESS;
  }

  return TrajPlannerResult::FAILED;
}

void ExplorationFSMReal::polyTraj2ROSMsg(
    const LocalTrajectory& local_traj, trajectory_manager::msg::PolyTraj& poly_msg)
{
  auto data = &local_traj;
  Eigen::VectorXd durs = data->traj.getDurations();
  int piece_num = data->traj.getPieceNum();

  poly_msg.drone_id = 0;
  poly_msg.traj_id = data->traj_id;
  poly_msg.start_time = data->start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);

  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);

    auto cMat = data->traj.operator[](i).getCoeffMat();
    int i8 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i8 + j] = cMat(0, j);
      poly_msg.coef_y[i8 + j] = cMat(1, j);
      poly_msg.coef_z[i8 + j] = cMat(2, j);
    }
  }
}

void ExplorationFSMReal::selectLocalTarget(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path, const double& local_distance,
    Eigen::Vector2d& target_pos, double& target_yaw)
{
  // First, try to find a collision-free target from the end of path
  for (int i = path.size() - 2; i >= 0; i--) {
    target_yaw = atan2(path.back()(1) - path[i](1), path.back()(0) - path[i](0));
    if (!expl_manager_->kinoastar_->isCollisionPosYaw(path[i], target_yaw)) {
      target_pos = path[i];
      break;
    }
  }

  // Find closest path point to current position
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select local target within local_distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      target_yaw = atan2(path[i](1) - path[i - 1](1), path[i](0) - path[i - 1](0));
      break;
    }
  }

  // Gradient-based safety adjustment
  double step_size = 0.05;
  double tolerance = 1e-3;
  int max_iterations = 30;

  for (int i = 0; i < max_iterations; ++i) {
    Eigen::Vector2d prev_pos = target_pos;

    // Get gradient from SDF map
    Eigen::Vector2d grad;
    double dist = expl_manager_->sdf_map_->getDistWithGrad(target_pos, grad);

    if (dist > 0.26)
      break;

    // Move along gradient to safer position
    if (grad.norm() > 1e-6) {
      target_pos += step_size * grad.normalized();
    }

    // Check convergence
    if ((target_pos - prev_pos).norm() < tolerance) {
      break;
    }
  }

  // Store selected local target
  expl_manager_->ed_->next_local_pos_ = target_pos;
}

void ExplorationFSMReal::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  auto vec2dTo3d = [](const std::vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    std::vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Eigen::Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontiers
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontiers
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw objects
  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Eigen::Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  std::vector<Eigen::Vector2d> local_points;
  local_points.push_back(ed_ptr->next_local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Eigen::Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
}

void ExplorationFSMReal::clearVisMarker()
{
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  visualization_->drawLines({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSMReal::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d sensor_pos = Eigen::Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

void ExplorationFSMReal::frontierCallback()
{
  // Update frontiers and visualize in idle states
  if (state_ != RealFSM::State::WAIT_TRIGGER && state_ != RealFSM::State::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

void ExplorationFSMReal::triggerCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (state_ != RealFSM::State::WAIT_TRIGGER)
    return;

  fd_->trigger_ = true;
  replan_fail_count_ = 0;
  RCLCPP_INFO(node_->get_logger(), "[Real] Exploration triggered!");
  transitState(RealFSM::State::PLAN_TRAJ, "triggerCallback");
}

void ExplorationFSMReal::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  // Extract linear velocity
  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  // Extract angular velocity
  fd_->odom_omega_(0) = msg->twist.twist.angular.x;
  fd_->odom_omega_(1) = msg->twist.twist.angular.y;
  fd_->odom_omega_(2) = msg->twist.twist.angular.z;

  fd_->have_odom_ = true;

  // Publish robot marker for visualization
  publishRobotMarker();
}

void ExplorationFSMReal::confidenceThresholdCallback(const std_msgs::msg::Float64::SharedPtr msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
  RCLCPP_INFO(node_->get_logger(), "[Real] Confidence threshold set to: %.2f", msg->data);
}

void ExplorationFSMReal::goalCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;

  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control;
  current_state << fd_->odom_pos_(0), fd_->odom_pos_(1), fd_->odom_yaw_, 0.0, fd_->odom_vel_(0);
  goal_state << x, y, yaw, 0.0, 0.0;
  if ((current_state.head(2) - goal_state.head(2)).norm() > 0.2) {
    current_control << 0.0, 0.0, 0.0;
    expl_manager_->planTrajectory(current_state, goal_state, current_control);
    trajectory_manager::msg::PolyTraj poly_msg;
    polyTraj2ROSMsg(expl_manager_->gcopter_->local_trajectory_, poly_msg);
    poly_traj_pub_->publish(poly_msg);
  }
  RCLCPP_INFO(node_->get_logger(), "[Real] Received goal pose: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
}

void ExplorationFSMReal::trajectoryFinishCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
  // Handle trajectory finish notification
  (void)msg;
}

void ExplorationFSMReal::emergencyStop()
{
  fd_->static_state_ = true;
  stop_pub_->publish(std_msgs::msg::Empty());
}

void ExplorationFSMReal::safetyCallback()
{
  if (state_ != RealFSM::State::REPLAN)
    return;

  // Check if robot deviates from planned trajectory
  double t_cur = (node_->get_clock()->now() - expl_manager_->gcopter_->local_trajectory_.start_time).seconds();
  t_cur = std::min(t_cur, expl_manager_->gcopter_->local_trajectory_.duration);
  Eigen::Vector3d cur_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_cur);

  if ((cur_pos.head(2) - fd_->odom_pos_.head(2)).norm() > 0.3) {
    RCLCPP_ERROR(node_->get_logger(), "[Real] Odom far from traj (%.2f, %.2f), Stop!!!",
        cur_pos(0), cur_pos(1));
    emergencyStop();
    transitState(RealFSM::State::PLAN_TRAJ, "Odom Far From Trajectory");
    return;
  }

  // Time-sampled safety check - use inflated map to detect obstacles
  double time_horizon = 2.5;  // Check trajectory for next 2.5 seconds
  double sample_dt = 0.1;     // Sample every 0.1 seconds

  for (double t_check = t_cur;
      t_check <= std::min(t_cur + time_horizon, expl_manager_->gcopter_->local_trajectory_.duration);
      t_check += sample_dt) {
    Eigen::Vector3d check_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_check);
    Eigen::Vector2d check_pos_2d = check_pos.head(2);

    // Skip positions too close to origin
    if ((check_pos_2d - Eigen::Vector2d(0.0, 0.0)).norm() < 1.5)
      continue;

    if (expl_manager_->sdf_map_->getInflateOccupancy(check_pos_2d)) {
      RCLCPP_ERROR(node_->get_logger(), "[Real] Safety Stop!!! Obstacle detected (%.2f, %.2f) at time %.2f",
          check_pos_2d(0), check_pos_2d(1), t_check);
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Trajectory Safety Stop");
      break;
    }
  }
}

void ExplorationFSMReal::publishRobotMarker()
{
  const double robot_height = FSMConstantsReal::ROBOT_HEIGHT;
  const double robot_radius = FSMConstantsReal::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::msg::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = node_->get_clock()->now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::msg::Marker::ADD;

  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  robot_marker.scale.x = robot_radius * 2;
  robot_marker.scale.y = robot_radius * 2;
  robot_marker.scale.z = robot_height;

  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::msg::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = node_->get_clock()->now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
  arrow_marker.action = visualization_msgs::msg::Marker::ADD;

  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  arrow_marker.scale.x = robot_radius + 0.13;
  arrow_marker.scale.y = 0.08;
  arrow_marker.scale.z = 0.08;

  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  robot_marker_pub_->publish(robot_marker);
  robot_marker_pub_->publish(arrow_marker);
}

void ExplorationFSMReal::transitState(RealFSM::State new_state, std::string pos_call)
{
  std::string state_str[] = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "EXEC_TRAJ", "REPLAN",
    "FINISH" };
  RCLCPP_INFO(node_->get_logger(), "[Real FSM]: %s -> from %s to %s", pos_call.c_str(),
      state_str[static_cast<int>(state_)].c_str(), state_str[static_cast<int>(new_state)].c_str());
  state_ = new_state;
}

void ExplorationFSMReal::habitatStateCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
  // HABITAT_STATE::EPISODE_FINISH = 3
  if (msg->data == 3) {
    RCLCPP_INFO(node_->get_logger(), "[Real] Episode finished, reinitializing...");
    init(node_);
  }
}

}  // namespace apexnav_planner
