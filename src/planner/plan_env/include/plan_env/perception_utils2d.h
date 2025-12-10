#ifndef _PERCEPTION_UTILS_2D_H_
#define _PERCEPTION_UTILS_2D_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Vector2d;
using std::vector;

namespace apexnav_planner {

class PerceptionUtils2D {
public:
  PerceptionUtils2D(ros::NodeHandle& nh);
  ~PerceptionUtils2D()
  {
  }

  // Set position and yaw angle
  void setPose(const Vector2d& pos, const double& yaw);

  // Get FOV information
  void getFOV(vector<Vector2d>& list1, vector<Vector2d>& list2);
  bool insideFOV(const Vector2d& point);
  void getFOVBoundingBox(Vector2d& bmin, Vector2d& bmax);

private:
  // Current position and yaw angle
  Vector2d pos_;
  double yaw_;

  // Plane normal vectors of camera FOV
  vector<Vector2d> normals_;

  // Parameters
  double left_angle_, right_angle_, max_dist_, vis_dist_;
  Vector2d n_left_, n_right_;

  // FOV vertices
  vector<Vector2d> cam_vertices1_, cam_vertices2_;
};

}  // namespace apexnav_planner

#endif
