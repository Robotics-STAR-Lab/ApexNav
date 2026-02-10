#include <rclcpp/rclcpp.hpp>
#include <gcopter/trajectory.hpp>
#include <trajectory_manager/msg/poly_traj.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include "controller/mpc.h"

#include <chrono>
#include <functional>

using namespace std;
using namespace Eigen;
using namespace std::chrono_literals;

class TrajectoryServer : public rclcpp::Node {
public:
  TrajectoryServer() : Node("trajectory_server_node")
  {
    receive_traj_ = false;
    have_odom_ = false;
    has_target_angle_ = false;
    target_yaw_ = 0.0;

    // Declare parameters
    if (!this->has_parameter("need_init")) {
      this->declare_parameter("need_init", false);
    }
    if (!this->has_parameter("max_correction_vel")) {
      this->declare_parameter("max_correction_vel", 0.6);
    }
    if (!this->has_parameter("max_correction_omega")) {
      this->declare_parameter("max_correction_omega", 1.2);
    }
    if (!this->has_parameter("mpc.predict_steps")) {
      this->declare_parameter("mpc.predict_steps", -1);
    }
    if (!this->has_parameter("mpc.dt")) {
      this->declare_parameter("mpc.dt", -1.0);
    }

    bool need_init = this->get_parameter("need_init").as_bool();
    max_correction_vel_ = this->get_parameter("max_correction_vel").as_double();
    max_correction_omega_ = this->get_parameter("max_correction_omega").as_double();
    mpc_N_ = this->get_parameter("mpc.predict_steps").as_int();
    mpc_dt_ = this->get_parameter("mpc.dt").as_double();

    // Create subscriptions
    traj_sub_ = this->create_subscription<trajectory_manager::msg::PolyTraj>(
        "trajectory", 10,
        std::bind(&TrajectoryServer::polyTrajCallback, this, std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odometry", 10,
        std::bind(&TrajectoryServer::odometryCallback, this, std::placeholders::_1));
    stop_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "/traj_server/stop", 10,
        std::bind(&TrajectoryServer::stopCallback, this, std::placeholders::_1));
    target_angle_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/traj_server/target_angle", 10,
        std::bind(&TrajectoryServer::targetAngleCallback, this, std::placeholders::_1));

    // Create publishers
    robot_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/robot", 10);
    vel_cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
    traj_vis_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/travel_traj", 10);
    current_desire_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/current_desire", 10);

    // Create timers
    vis_timer_ = this->create_wall_timer(200ms, std::bind(&TrajectoryServer::visCallback, this));
    cmd_timer_ = this->create_wall_timer(20ms, std::bind(&TrajectoryServer::cmdCallBack, this));

    RCLCPP_INFO(this->get_logger(), "[traj_server] TrajectoryServer initialized, waiting for messages...");

    need_init_ = need_init;
  }

  void initMPC()
  {
    if (mpc_N_ <= 0 || mpc_dt_ <= 0.0) {
      RCLCPP_ERROR(this->get_logger(), "[traj_server] Wrong MPC parameters!");
      return;
    }
    mpc_controller_.reset(new MPC);
    mpc_controller_->init(this->shared_from_this());
    xref_.resize(mpc_N_);

    if (need_init_) {
      init_state_ = 0;
      init_rotation_started_ = false;
      rotation_accum_ = 0.0;
      last_odom_yaw_ = 0.0;
      init_cmd_timer_ = this->create_wall_timer(100ms,
          std::bind(&TrajectoryServer::initCmdCallback, this));
    }
  }

  void initCmdCallback()
  {
    geometry_msgs::msg::Twist twist_msg;
    switch (init_state_) {
      case 0: {
        // Prefer odom-based rotation stop: accumulate yaw change from odom
        if (have_odom_) {
          if (!init_rotation_started_) {
            last_odom_yaw_ = odom_yaw_;
            rotation_accum_ = 0.0;
            init_rotation_started_ = true;
          }

          // publish rotation command
          twist_msg.angular.z = M_PI / 6;  // rotation speed
          vel_cmd_pub_->publish(twist_msg);

          // accumulate yaw change using shortest-angle difference
          double delta = atan2(sin(odom_yaw_ - last_odom_yaw_), cos(odom_yaw_ - last_odom_yaw_));
          rotation_accum_ += fabs(delta);
          last_odom_yaw_ = odom_yaw_;

          // Stop after approximately one full rotation
          if (rotation_accum_ >= 2.0 * M_PI - 0.05) {
            twist_msg.angular.z = 0.0;
            vel_cmd_pub_->publish(twist_msg);
            init_state_++;
          }
        }
        else {
          // Waiting for odom: do not start rotation until odom is available.
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
              "[traj_server] Waiting for odom to start init rotation.");
          return;
        }
      } break;

      case 1: {
        twist_msg.linear.x = 0.0;
        vel_cmd_pub_->publish(twist_msg);
        init_cmd_timer_->cancel();
      } break;

      default:
        break;
    }
  }

  void polyTrajCallback(const trajectory_manager::msg::PolyTraj::SharedPtr msg)
  {
    if (msg->order != 7) {
      RCLCPP_ERROR(this->get_logger(), "[traj_server] Only support trajectory order equals 7 now!");
      return;
    }

    if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
      RCLCPP_ERROR(this->get_logger(), "[traj_server] WRONG trajectory parameters, ");
      return;
    }

    int piece_nums = msg->duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<Eigen::Matrix<double, 3, 8>> cMats(piece_nums);

    for (int i = 0; i < piece_nums; ++i) {
      int i8 = i * 8;
      cMats[i].row(0) << msg->coef_x[i8 + 0], msg->coef_x[i8 + 1], msg->coef_x[i8 + 2],
          msg->coef_x[i8 + 3], msg->coef_x[i8 + 4], msg->coef_x[i8 + 5], msg->coef_x[i8 + 6],
          msg->coef_x[i8 + 7];
      cMats[i].row(1) << msg->coef_y[i8 + 0], msg->coef_y[i8 + 1], msg->coef_y[i8 + 2],
          msg->coef_y[i8 + 3], msg->coef_y[i8 + 4], msg->coef_y[i8 + 5], msg->coef_y[i8 + 6],
          msg->coef_y[i8 + 7];
      cMats[i].row(2) << msg->coef_z[i8 + 0], msg->coef_z[i8 + 1], msg->coef_z[i8 + 2],
          msg->coef_z[i8 + 3], msg->coef_z[i8 + 4], msg->coef_z[i8 + 5], msg->coef_z[i8 + 6],
          msg->coef_z[i8 + 7];
      dura[i] = msg->duration[i];
    }

    traj_.reset(new Trajectory<7, 3>(dura, cMats));
    start_time_ = rclcpp::Time(msg->start_time);
    traj_duration_ = traj_->getTotalDuration();
    traj_id_ = msg->traj_id;
    receive_traj_ = true;

    RCLCPP_INFO(this->get_logger(),
        "[traj_server] Received trajectory ID %d, total duration: %.3f, start_time: %.3f",
        traj_id_, traj_duration_, start_time_.seconds());
  }

  void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;

    odom_linear_vel_(0) = msg->twist.twist.linear.x;
    odom_linear_vel_(1) = msg->twist.twist.linear.y;
    odom_linear_vel_(2) = msg->twist.twist.linear.z;

    Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
    odom_yaw_ = atan2(rot_x(1), rot_x(0));
    have_odom_ = true;
    traj_real_.push_back(Eigen::Vector3d(odom_pos_(0), odom_pos_(1), 0.15));
    if (traj_real_.size() > 50000)
      traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 10000);
  }

  void stopCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    (void)msg;
    // Immediate emergency stop
    rclcpp::Time time_now = this->get_clock()->now();
    double t_stop = (time_now - start_time_).seconds();
    traj_duration_ = min(t_stop, traj_duration_);
  }

  void targetAngleCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    target_yaw_ = msg->data;
    has_target_angle_ = true;
    rotation_start_time_ = this->get_clock()->now();

    RCLCPP_INFO(this->get_logger(), "Received target angle: %.3f radians (%.1f degrees)",
        target_yaw_, target_yaw_ * 180.0 / M_PI);
  }

  void visCallback()
  {
    displayTrajWithColor(
        traj_real_, 0.10, Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), 0);
  }

  void cmdCallBack()
  {
    // Check for rotate-to-target-angle task
    if (has_target_angle_) {
      executeRotationToTarget();
      return;
    }

    if (!receive_traj_) {
      return;
    }

    rclcpp::Time current_time = this->get_clock()->now();
    double elapsed_time = (current_time - start_time_).seconds();

    if (elapsed_time < 0)
      return;  // Wait for start time to pass

    if (elapsed_time > traj_duration_) {
      // Trajectory finished, stop publishing
      geometry_msgs::msg::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_->publish(twist_msg);  // Publish zero velocity
      receive_traj_ = false;             // Reset flag so that no more commands are published
      return;
    }

    // MPC control
    Eigen::Vector3d pos = traj_->getPos(elapsed_time);
    Eigen::Vector3d vel = traj_->getVel(elapsed_time);

    Eigen::Vector3d ref;
    ref(0) = pos(0);
    ref(1) = pos(1);
    ref(2) = atan2(vel(1), vel(0));
    for (int i = 0; i < mpc_N_; ++i) {
      double temp_t = elapsed_time + i * mpc_dt_;
      if (temp_t <= traj_duration_) {
        pos = traj_->getPos(temp_t);
        vel = traj_->getVel(temp_t);
        ref(0) = pos(0);
        ref(1) = pos(1);
        ref(2) = atan2(vel(1), vel(0));
      }
      xref_[i] = ref;
    }
    Eigen::Vector2d cmd;
    mpc_controller_->setOdom(
        Eigen::Vector4d(odom_pos_(0), odom_pos_(1), odom_yaw_, odom_linear_vel_.head(2).norm()));
    cmd = mpc_controller_->calCmd(xref_);
    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = cmd(0);
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = cmd(1);
    vel_cmd_pub_->publish(twist_msg);

    // Publish current desired pose
    geometry_msgs::msg::Pose desire_pose;
    Eigen::Vector3d current_desire_pos = traj_->getPos(elapsed_time);
    Eigen::Vector3d current_desire_vel = traj_->getVel(elapsed_time);
    double current_desire_yaw = atan2(current_desire_vel(1), current_desire_vel(0));

    desire_pose.position.x = current_desire_pos(0);
    desire_pose.position.y = current_desire_pos(1);
    desire_pose.position.z = current_desire_pos(2);

    // Convert yaw to quaternion
    Eigen::Quaterniond q(Eigen::AngleAxisd(current_desire_yaw, Eigen::Vector3d::UnitZ()));
    desire_pose.orientation.x = q.x();
    desire_pose.orientation.y = q.y();
    desire_pose.orientation.z = q.z();
    desire_pose.orientation.w = q.w();

    current_desire_pub_->publish(desire_pose);
  }

  void executeRotationToTarget()
  {
    if (!have_odom_) {
      return;
    }

    // Compute yaw error
    double yaw_error =
        std::atan2(std::sin(target_yaw_ - odom_yaw_), std::cos(target_yaw_ - odom_yaw_));

    // Angle threshold: consider target reached if below this
    const double angle_threshold = 0.02;  // ~0.6 degrees

    if (std::abs(yaw_error) < angle_threshold) {
      // Target angle reached: stop rotating
      geometry_msgs::msg::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_->publish(twist_msg);

      has_target_angle_ = false;
      RCLCPP_INFO(this->get_logger(), "Reached target angle: %.3f radians (%.1f degrees)",
          target_yaw_, target_yaw_ * 180.0 / M_PI);
      return;
    }

    // Compute angular velocity: use a simple P controller instead of PID
    const double Kp_rotation = 2.0;  // proportional gain; adjust if needed
    double angular_velocity = Kp_rotation * yaw_error;

    // Limit maximum angular velocity
    const double max_angular_velocity = max_correction_omega_;  // rad/s
    angular_velocity = std::max(-max_angular_velocity, std::min(max_angular_velocity, angular_velocity));

    // Send rotation command
    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = angular_velocity;
    vel_cmd_pub_->publish(twist_msg);

    // Log debug info
    double elapsed_rotation_time = (this->get_clock()->now() - rotation_start_time_).seconds();
    if (static_cast<int>(elapsed_rotation_time * 10) % 10 == 0) {  // print every 0.1s
      RCLCPP_INFO(this->get_logger(),
          "Rotating to target: current=%.2f deg, target=%.2f deg, error=%.2f deg, vel=%.2f rad/s",
          odom_yaw_ * 180.0 / M_PI, target_yaw_ * 180.0 / M_PI, yaw_error * 180.0 / M_PI,
          angular_velocity);
    }
  }

  void displayTrajWithColor(
      vector<Eigen::Vector3d> path, double resolution, Eigen::Vector4d color, int id)
  {
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = this->get_clock()->now();
    mk.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    mk.action = visualization_msgs::msg::Marker::DELETE;
    mk.id = id;
    traj_vis_pub_->publish(mk);

    mk.action = visualization_msgs::msg::Marker::ADD;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);
    mk.scale.x = resolution;
    mk.scale.y = resolution;
    mk.scale.z = resolution;
    geometry_msgs::msg::Point pt;
    for (int i = 0; i < int(path.size()); i++) {
      pt.x = path[i](0);
      pt.y = path[i](1);
      pt.z = path[i](2);
      mk.points.push_back(pt);
    }
    traj_vis_pub_->publish(mk);
  }

private:
  rclcpp::Subscription<trajectory_manager::msg::PolyTraj>::SharedPtr traj_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr target_angle_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_cmd_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_vis_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr current_desire_pub_;

  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr vis_timer_;
  rclcpp::TimerBase::SharedPtr init_cmd_timer_;

  // Trajectory Data
  std::unique_ptr<Trajectory<7, 3>> traj_;
  rclcpp::Time start_time_;
  double traj_duration_;
  int traj_id_;
  bool receive_traj_;

  bool use_mpc_ = true;
  MPC::Ptr mpc_controller_;
  std::vector<Eigen::Vector3d> xref_;
  int mpc_N_;
  double mpc_dt_;

  // Target Angle Data
  double target_yaw_;
  bool has_target_angle_;
  rclcpp::Time rotation_start_time_;

  // Data
  Vector3d odom_pos_, odom_linear_vel_;
  Quaterniond odom_orient_;
  double odom_yaw_;
  bool have_odom_;
  double replan_time_ = 0.5;
  vector<Eigen::Vector3d> traj_real_;
  int init_state_;
  // init rotation: prefer odom-based stopping (accumulate yaw change);
  bool init_rotation_started_;
  double rotation_accum_;  // accumulated absolute yaw change (rad)
  double last_odom_yaw_;   // last odom yaw used for accumulation
  double max_correction_vel_, max_correction_omega_;
  bool need_init_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryServer>();
  node->initMPC();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
