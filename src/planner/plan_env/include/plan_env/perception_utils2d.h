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

  // 设置位置和偏航角
  void setPose(const Vector2d& pos, const double& yaw);

  // 获取 FOV 信息
  void getFOV(vector<Vector2d>& list1, vector<Vector2d>& list2);
  bool insideFOV(const Vector2d& point);
  void getFOVBoundingBox(Vector2d& bmin, Vector2d& bmax);

private:
  // 当前位置和偏航角
  Vector2d pos_;
  double yaw_;

  // 相机 FOV 的平面法向量
  vector<Vector2d> normals_;

  // 参数
  double left_angle_, right_angle_, max_dist_, vis_dist_;
  Vector2d n_left_, n_right_;

  // FOV 顶点
  vector<Vector2d> cam_vertices1_, cam_vertices2_;
};

}  // namespace apexnav_planner

#endif
