#ifndef RAYCAST2D_H_
#define RAYCAST2D_H_

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <cmath>
class RayCaster2D {
private:
  /* data */
  Eigen::Vector2d start_;
  Eigen::Vector2d end_;
  Eigen::Vector2d direction_;
  Eigen::Vector2d min_;
  Eigen::Vector2d max_;
  int x_;
  int y_;
  int endX_;
  int endY_;
  double maxDist_;
  double dx_;
  double dy_;
  int stepX_;
  int stepY_;
  double tMaxX_;
  double tMaxY_;
  double tDeltaX_;
  double tDeltaY_;
  double dist_;

  int step_num_;

  double resolution_;
  Eigen::Vector2d offset_;
  Eigen::Vector2d half_;
  double signum(double x);
  double mod(double value, double modulus);
  double intbound(double s, double ds);

public:
  RayCaster2D(/* args */)
  {
  }
  ~RayCaster2D()
  {
  }

  void setParams(const double& res, const Eigen::Vector2d& origin);
  bool input(const Eigen::Vector2d& start, const Eigen::Vector2d& end);
  bool nextId(Eigen::Vector2i& idx);
  bool nextPos(Eigen::Vector2d& pos);
};

inline double RayCaster2D::signum(double x)
{
  return x == 0 ? 0 : x < 0 ? -1 : 1;
}

inline double RayCaster2D::mod(double value, double modulus)
{
  return fmod(fmod(value, modulus) + modulus, modulus);
}

inline double RayCaster2D::intbound(double s, double ds)
{
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0) {
    return intbound(-s, -ds);
  }
  else {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

#endif  // RAYCAST_H_