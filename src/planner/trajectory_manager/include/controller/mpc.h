#pragma once

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm>

#include <OsqpEigen/OsqpEigen.h>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>

using namespace std;
using namespace Eigen;

class MPCState {
public:
  double x = 0;
  double y = 0;
  double v = 0;
  double yaw = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class MPC {
public:
  Eigen::Vector2d calCmd(const std::vector<Eigen::Vector3d>& _xref);
  void setOdom(const Eigen::Vector4d& car_state);

private:
  // parameters
  /// algorithm param
  double du_th = 0.1;
  double dt = 0.2;
  int T = 5;
  int delay_num;
  int max_iter = 3;
  vector<double> Q = { 10, 10, 2.5, 0.5 };
  vector<double> R = { 0.01, 0.01 };
  vector<double> Rd = { 0.01, 1.0 };
  /// constraints
  double max_omega = M_PI / 4;
  double max_comega = M_PI / 6 * 0.2;
  double min_speed = -55.0 / 3.6;
  double max_speed = 55.0 / 3.6;
  double max_cv = 0.2;
  double max_accel = 1.0;

  // MPC dataset
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::VectorXd C;
  MPCState xbar[500];
  Eigen::MatrixXd xref;
  Eigen::MatrixXd dref;
  Eigen::MatrixXd output;
  Eigen::MatrixXd last_output;
  std::vector<Eigen::Vector2d> output_buff;

  // control data
  bool has_odom;
  bool receive_traj_ = false;
  double tolerance = 0.1;
  double traj_duration_;
  double t_track = 0.0;
  ros::Time start_time_;
  MPCState now_state;

  // ros interface
  ros::NodeHandle node_;
  ros::Timer cmd_timer_;
  ros::Publisher predict_pub, ref_pub, err_pub;

  // MPC function
  void getLinearModel(const MPCState& s);
  void stateTrans(MPCState& s, double v, double yaw_dot);
  void predictMotion(void);
  void predictMotion(MPCState* b);
  void solveMPCV();
  void getCmd();

  // utils
  MPCState xopt[500];
  void normlize_theta(double& th)
  {
    while (th > M_PI) th -= M_PI * 2;
    while (th < -M_PI) th += M_PI * 2;
  }

  void smooth_yaw(void)
  {
    double dyaw = xref(3, 0) - now_state.yaw;

    while (dyaw >= M_PI / 2) {
      xref(3, 0) -= M_PI * 2;
      dyaw = xref(3, 0) - now_state.yaw;
    }
    while (dyaw <= -M_PI / 2) {
      xref(3, 0) += M_PI * 2;
      dyaw = xref(3, 0) - now_state.yaw;
    }

    for (int i = 0; i < T - 1; i++) {
      dyaw = xref(3, i + 1) - xref(3, i);
      while (dyaw >= M_PI / 2) {
        xref(3, i + 1) -= M_PI * 2;
        dyaw = xref(3, i + 1) - xref(3, i);
      }
      while (dyaw <= -M_PI / 2) {
        xref(3, i + 1) += M_PI * 2;
        dyaw = xref(3, i + 1) - xref(3, i);
      }
    }
  }

  void drawPredictPath(MPCState* b)
  {
    int id = 0;
    double sc = 0.2;
    visualization_msgs::Marker sphere, line_strip;
    sphere.header.frame_id = line_strip.header.frame_id = "world";
    sphere.header.stamp = line_strip.header.stamp = ros::Time::now();
    sphere.type = visualization_msgs::Marker::SPHERE_LIST;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    sphere.action = line_strip.action = visualization_msgs::Marker::ADD;
    sphere.id = id;
    line_strip.id = id + 1000;

    sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
    sphere.color.r = line_strip.color.r = 1;
    sphere.color.g = line_strip.color.g = 0;
    sphere.color.b = line_strip.color.b = 1;
    sphere.color.a = line_strip.color.a = 1;
    sphere.scale.x = sc;
    sphere.scale.y = sc;
    sphere.scale.z = sc;
    line_strip.scale.x = sc / 2;
    geometry_msgs::Point pt;

    for (int i = 0; i < T; i++) {
      pt.x = b[i].x;
      pt.y = b[i].y;
      pt.z = 0.0;
      line_strip.points.push_back(pt);
    }
    predict_pub.publish(line_strip);
  }

  void drawRefPath(void)
  {
    int id = 0;
    double sc = 0.2;
    visualization_msgs::Marker sphere, line_strip;
    sphere.header.frame_id = line_strip.header.frame_id = "world";
    sphere.header.stamp = line_strip.header.stamp = ros::Time::now();
    sphere.type = visualization_msgs::Marker::SPHERE_LIST;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    sphere.action = line_strip.action = visualization_msgs::Marker::ADD;
    sphere.id = id;
    line_strip.id = id + 1000;

    sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
    sphere.color.r = line_strip.color.r = 0;
    sphere.color.g = line_strip.color.g = 0;
    sphere.color.b = line_strip.color.b = 1;
    sphere.color.a = line_strip.color.a = 1;
    sphere.scale.x = sc;
    sphere.scale.y = sc;
    sphere.scale.z = sc;
    line_strip.scale.x = sc / 2;
    geometry_msgs::Point pt;

    for (int i = 0; i < T; i++) {
      pt.x = xref(0, i);
      pt.y = xref(1, i);
      pt.z = 0.0;
      line_strip.points.push_back(pt);
    }
    ref_pub.publish(line_strip);
  }

public:
  MPC()
  {
  }
  void init(ros::NodeHandle& nh);
  ~MPC()
  {
  }

  typedef shared_ptr<MPC> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};