#include <plan_env/perception_utils2d.h>

namespace apexnav_planner {

PerceptionUtils2D::PerceptionUtils2D(ros::NodeHandle& nh)
{
  nh.param("perception_utils/left_angle", left_angle_, -1.0);
  nh.param("perception_utils/right_angle", right_angle_, -1.0);
  nh.param("perception_utils/max_dist", max_dist_, -1.0);
  nh.param("perception_utils/vis_dist", vis_dist_, -1.0);

  // 初始化 FOV 法向量（改为基于 X 轴正方向）
  n_left_ << cos(M_PI_2 - left_angle_), sin(M_PI_2 - left_angle_);
  n_right_ << cos(-M_PI_2 + right_angle_), sin(-M_PI_2 + right_angle_);

  // 初始化 FOV 顶点（改为基于 X 轴正方向）
  double vert = vis_dist_ * tan(left_angle_);  // 上下高度
  Vector2d origin(0, 0);                       // 原点
  Vector2d left(vis_dist_, vert);              // 左侧顶点
  Vector2d right(vis_dist_, -vert);            // 右侧顶点

  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(left);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(right);

  cam_vertices1_.push_back(left);
  cam_vertices2_.push_back(right);
}

void PerceptionUtils2D::setPose(const Vector2d& pos, const double& yaw)
{
  pos_ = pos;
  yaw_ = yaw;

  // 计算当前 FOV 法向量
  Matrix2d R_wb;
  R_wb << cos(yaw_), -sin(yaw_), sin(yaw_), cos(yaw_);

  normals_ = { n_left_, n_right_ };
  for (auto& n : normals_) {
    n = R_wb * n;
  }
}

void PerceptionUtils2D::getFOV(vector<Vector2d>& list1, vector<Vector2d>& list2)
{
  list1.clear();
  list2.clear();

  // 获取当前 FOV 的顶点
  Matrix2d Rwb;
  Rwb << cos(yaw_), -sin(yaw_), sin(yaw_), cos(yaw_);
  for (int i = 0; i < (int)cam_vertices1_.size(); ++i) {
    auto p1 = Rwb * cam_vertices1_[i] + pos_;
    auto p2 = Rwb * cam_vertices2_[i] + pos_;
    list1.push_back(p1);
    list2.push_back(p2);
  }
}

bool PerceptionUtils2D::insideFOV(const Vector2d& point)
{
  Vector2d dir = point - pos_;
  if (dir.norm() > max_dist_)  // 仅考虑 x, y 平面的距离
    return false;

  dir.normalize();  // 标准化
  for (auto n : normals_) {
    if (dir.dot(n) < 0.0)  // 与法向量的点积判断是否在 FOV 内
      return false;
  }
  return true;
}

void PerceptionUtils2D::getFOVBoundingBox(Vector2d& bmin, Vector2d& bmax)
{
  double left = yaw_ + left_angle_;
  double right = yaw_ - right_angle_;
  Vector2d left_pt = pos_ + max_dist_ * Vector2d(cos(left), sin(left));
  Vector2d right_pt = pos_ + max_dist_ * Vector2d(cos(right), sin(right));
  vector<Vector2d> points = { left_pt, right_pt };

  bmax = bmin = pos_;
  for (auto p : points) {
    bmax = bmax.array().max(p.array());
    bmin = bmin.array().min(p.array());
  }
}

}  // namespace apexnav_planner
