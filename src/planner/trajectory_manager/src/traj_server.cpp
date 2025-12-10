#include <ros/ros.h>
#include <gcopter/trajectory.hpp>
#include <trajectory_manager/PolyTraj.h>
#include <Eigen/Dense>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include "controller/mpc.h"

using namespace std;
using namespace Eigen;
// PIDController removed — using MPC (and simple P for small-angle rotation) instead

class TrajectoryServer {
public:
  TrajectoryServer(ros::NodeHandle& nh)
  {
    nh_ = nh;
    receive_traj_ = false;
    have_odom_ = false;
    has_target_angle_ = false;
    target_yaw_ = 0.0;
    bool need_init;
    nh.param("need_init", need_init, false);
    nh.param("max_correction_vel", max_correction_vel_, 0.6);
    nh.param("max_correction_omega", max_correction_omega_, 1.2);
    traj_sub_ = nh_.subscribe("trajectory", 10, &TrajectoryServer::polyTrajCallback, this);
    odom_sub_ = nh_.subscribe("odometry", 10, &TrajectoryServer::odometryCallback, this);
    stop_sub_ = nh_.subscribe("/traj_server/stop", 10, &TrajectoryServer::stopCallback, this);
    target_angle_sub_ = nh_.subscribe(
        "/traj_server/target_angle", 10, &TrajectoryServer::targetAngleCallback, this);
    robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
    vel_cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    traj_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/travel_traj", 10);
    current_desire_pub_ = nh_.advertise<geometry_msgs::Pose>("/current_desire", 10);
    vis_timer_ = nh_.createTimer(ros::Duration(0.20), &TrajectoryServer::visCallback, this);
    cmd_timer_ = nh_.createTimer(ros::Duration(0.02), &TrajectoryServer::cmdCallBack, this);
    init_cmd_timer_ =
        nh_.createTimer(ros::Duration(0.1), &TrajectoryServer::initCmdCallback, this, false, false);
    std::cout << "[traj_server] TrajectoryServer initialized, waiting for messages..." << std::endl;

    nh.param("mpc/predict_steps", mpc_N_, -1);
    nh.param("mpc/dt", mpc_dt_, -1.0);
    if (mpc_N_ <= 0 || mpc_dt_ <= 0.0) {
      ROS_ERROR("[traj_server] Wrong MPC parameters!");
      return;
    }
    mpc_controller_.reset(new MPC);
    mpc_controller_->init(nh_);
    xref_.resize(mpc_N_);

    if (need_init) {
      init_state_ = 0;
      init_rotation_started_ = false;
      rotation_accum_ = 0.0;
      last_odom_yaw_ = 0.0;
      init_cmd_timer_.start();
    }
  }

  void initCmdCallback(const ros::TimerEvent& event)
  {
    geometry_msgs::Twist twist_msg;
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
          vel_cmd_pub_.publish(twist_msg);

          // accumulate yaw change using shortest-angle difference
          double delta = atan2(sin(odom_yaw_ - last_odom_yaw_), cos(odom_yaw_ - last_odom_yaw_));
          rotation_accum_ += fabs(delta);
          last_odom_yaw_ = odom_yaw_;

          // Stop after approximately one full rotation
          if (rotation_accum_ >= 2.0 * M_PI - 0.05) {
            twist_msg.angular.z = 0.0;
            vel_cmd_pub_.publish(twist_msg);
            init_state_++;
          }
        }
        else {
          // Waiting for odom: do not start rotation until odom is available.
          ROS_WARN_THROTTLE(5, "[traj_server] Waiting for odom to start init rotation.");
          return;
        }
      } break;

      case 1: {
        twist_msg.linear.x = 0.0;
        vel_cmd_pub_.publish(twist_msg);
        init_cmd_timer_.stop();
      } break;

      default:
        break;
    }
  }

  void polyTrajCallback(const trajectory_manager::PolyTrajConstPtr& msg)
  {
    if (msg->order != 7) {
      ROS_ERROR("[traj_server] Only support trajectory order equals 7 now!");
      return;
    }

    if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
      ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
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
    start_time_ = msg->start_time;
    traj_duration_ = traj_->getTotalDuration();
    traj_id_ = msg->traj_id;
    receive_traj_ = true;

    std::cout << "[traj_server] Received trajectory ID " << traj_id_
              << ", total duration: " << traj_duration_ << ", start_time: " << start_time_.toSec()
              << std::endl;
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg)
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
    // publishRobotMarker();
    traj_real_.push_back(Eigen::Vector3d(odom_pos_(0), odom_pos_(1), 0.15));
    if (traj_real_.size() > 50000)
      traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 10000);
  }

  void stopCallback(const std_msgs::EmptyConstPtr& msg)
  {
    // Immediate emergency stop
    ros::Time time_now = ros::Time::now();
    double t_stop = (time_now - start_time_).toSec();
    traj_duration_ = min(t_stop, traj_duration_);
  }

  void targetAngleCallback(const std_msgs::Float32ConstPtr& msg)
  {
    target_yaw_ = msg->data;
    has_target_angle_ = true;
    rotation_start_time_ = ros::Time::now();

    ROS_INFO("Received target angle: %.3f radians (%.1f degrees)", target_yaw_,
        target_yaw_ * 180.0 / M_PI);
  }

  void visCallback(const ros::TimerEvent& e)
  {
    displayTrajWithColor(
        traj_real_, 0.10, Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), 0);
  }

  void cmdCallBack(const ros::TimerEvent& event)
  {
    // Check for rotate-to-target-angle task
    if (has_target_angle_) {
      executeRotationToTarget();
      return;
    }

    if (!receive_traj_) {
      return;
    }

    ros::Time current_time = ros::Time::now();
    double elapsed_time = (current_time - start_time_).toSec();

    if (elapsed_time < 0)
      return;  // Wait for start time to pass

    if (elapsed_time > traj_duration_) {
      // Trajectory finished, stop publishing
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_.publish(twist_msg);  // Publish zero velocity
      receive_traj_ = false;            // Reset flag so that no more commands are published
      return;
    }

    if (use_mpc_) {
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
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = cmd(0);
      twist_msg.linear.y = 0.0;
      twist_msg.linear.z = 0.0;
      twist_msg.angular.x = 0.0;
      twist_msg.angular.y = 0.0;
      twist_msg.angular.z = cmd(1);
      vel_cmd_pub_.publish(twist_msg);

      // Publish current desired pose
      geometry_msgs::Pose desire_pose;
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

      current_desire_pub_.publish(desire_pose);
      // ROS_ERROR("mpc cmd: [%f, %f]", cmd(0), cmd(1));
    }
    else {
      // Non-MPC branch removed: use MPC controller to compute commands here as well.
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
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = cmd(0);
      twist_msg.linear.y = 0.0;
      twist_msg.linear.z = 0.0;
      twist_msg.angular.x = 0.0;
      twist_msg.angular.y = 0.0;
      twist_msg.angular.z = cmd(1);
      vel_cmd_pub_.publish(twist_msg);

      // Publish current desired pose
      geometry_msgs::Pose desire_pose;
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

      current_desire_pub_.publish(desire_pose);
    }
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
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_.publish(twist_msg);

      has_target_angle_ = false;
      ROS_INFO("Reached target angle: %.3f radians (%.1f degrees)", target_yaw_,
          target_yaw_ * 180.0 / M_PI);
      return;
    }

    // Compute angular velocity: use a simple P controller instead of PID
    const double Kp_rotation = 2.0;  // proportional gain; adjust if needed
    double angular_velocity = Kp_rotation * yaw_error;

    // Limit maximum angular velocity
    const double max_angular_velocity = max_correction_omega_;  // rad/s
    angular_velocity = std::max(-max_angular_velocity, std::min(max_angular_velocity, angular_velocity));

    // Send rotation command
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = angular_velocity;
    vel_cmd_pub_.publish(twist_msg);

    // Log debug info
    double elapsed_rotation_time = (ros::Time::now() - rotation_start_time_).toSec();
    if (static_cast<int>(elapsed_rotation_time * 10) % 10 == 0) {  // print every 0.1s
      ROS_INFO("Rotating to target: current=%.2f°, target=%.2f°, error=%.2f°, vel=%.2f rad/s",
          odom_yaw_ * 180.0 / M_PI, target_yaw_ * 180.0 / M_PI, yaw_error * 180.0 / M_PI,
          angular_velocity);
    }
  }

  void publishRobotMarker()
  {
    const double robot_height = 0.15;
    const double robot_radius = 0.18;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";  // Set reference frame
    marker.header.stamp = ros::Time::now();
    marker.ns = "robot_position";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;  // Set to CYLINDER
    marker.action = visualization_msgs::Marker::ADD;

    // Set cylinder position
    marker.pose.position.x = odom_pos_(0);
    marker.pose.position.y = odom_pos_(1);
    marker.pose.position.z = odom_pos_(2) + robot_height / 2.0;

    // Set cylinder orientation (quaternion)
    marker.pose.orientation.x = odom_orient_.x();
    marker.pose.orientation.y = odom_orient_.y();
    marker.pose.orientation.z = odom_orient_.z();
    marker.pose.orientation.w = odom_orient_.w();

    // Set cylinder size
    marker.scale.x = robot_radius * 2;  // diameter
    marker.scale.y = robot_radius * 2;  // diameter
    marker.scale.z = robot_height;      // height

    marker.color.r = 50.0 / 255.0;
    marker.color.g = 50.0 / 255.0;
    marker.color.b = 255.0 / 255.0;
    marker.color.a = 1.0;  // opaque

    // Create and publish arrow (direction)
    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = "world";
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = "robot_direction";
    arrow_marker.id = 1;
    arrow_marker.type = visualization_msgs::Marker::ARROW;  // Set to ARROW
    arrow_marker.action = visualization_msgs::Marker::ADD;

    // Set arrow position (start)
    arrow_marker.pose.position.x = odom_pos_(0);
    arrow_marker.pose.position.y = odom_pos_(1);
    arrow_marker.pose.position.z = odom_pos_(2) + robot_height;

    // Set arrow orientation (from quaternion)
    arrow_marker.pose.orientation.x = odom_orient_.x();
    arrow_marker.pose.orientation.y = odom_orient_.y();
    arrow_marker.pose.orientation.z = odom_orient_.z();
    arrow_marker.pose.orientation.w = odom_orient_.w();

    // Set arrow size
    arrow_marker.scale.x = robot_radius + 0.13;  // arrow length
    arrow_marker.scale.y = 0.08;                 // arrow width
    arrow_marker.scale.z = 0.08;                 // arrow thickness

    arrow_marker.color.r = 10.0 / 255.0;
    arrow_marker.color.g = 255.0 / 255.0;
    arrow_marker.color.b = 10.0 / 255.0;
    arrow_marker.color.a = 1.0;  // opaque

    robot_marker_pub_.publish(marker);
    robot_marker_pub_.publish(arrow_marker);
  }

  void displayTrajWithColor(
      vector<Eigen::Vector3d> path, double resolution, Eigen::Vector4d color, int id)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::SPHERE_LIST;
    mk.action = visualization_msgs::Marker::DELETE;
    mk.id = id;
    traj_vis_pub_.publish(mk);

    mk.action = visualization_msgs::Marker::ADD;
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
    geometry_msgs::Point pt;
    for (int i = 0; i < int(path.size()); i++) {
      pt.x = path[i](0);
      pt.y = path[i](1);
      pt.z = path[i](2);
      mk.points.push_back(pt);
    }
    traj_vis_pub_.publish(mk);
    ros::Duration(0.0001).sleep();
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber traj_sub_, odom_sub_, stop_sub_, target_angle_sub_;
  ros::Publisher vel_cmd_pub_, robot_marker_pub_, traj_vis_pub_, current_desire_pub_;
  ros::Timer cmd_timer_, vis_timer_, init_cmd_timer_;

  // Trajectory Data
  std::unique_ptr<Trajectory<7, 3>> traj_;
  ros::Time start_time_;
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
  ros::Time rotation_start_time_;

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
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_server_node");
  ros::NodeHandle nh("~");
  TrajectoryServer traj_server(nh);
  ros::spin();
  return 0;
}