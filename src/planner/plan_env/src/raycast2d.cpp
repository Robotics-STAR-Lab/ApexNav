#include <plan_env/raycast2d.h>

void RayCaster2D::setParams(const double& res, const Eigen::Vector2d& origin)
{
  resolution_ = res;
  half_ = Eigen::Vector2d(0.5, 0.5);
  offset_ = half_ - origin / resolution_;
}

bool RayCaster2D::input(const Eigen::Vector2d& start, const Eigen::Vector2d& end)
{
  start_ = start / resolution_;
  end_ = end / resolution_;

  x_ = (int)std::floor(start_.x());
  y_ = (int)std::floor(start_.y());
  endX_ = (int)std::floor(end_.x());
  endY_ = (int)std::floor(end_.y());
  direction_ = (end_ - start_);
  maxDist_ = direction_.squaredNorm();

  // Break out direction vector.
  dx_ = endX_ - x_;
  dy_ = endY_ - y_;

  // Direction to increment x,y,z when stepping.
  stepX_ = (int)signum((int)dx_);
  stepY_ = (int)signum((int)dy_);

  // See description above. The initial values depend on the fractional
  // part of the origin.
  tMaxX_ = intbound(start_.x(), dx_);
  tMaxY_ = intbound(start_.y(), dy_);

  // The change in t when taking a step (always positive).
  tDeltaX_ = ((double)stepX_) / dx_;
  tDeltaY_ = ((double)stepY_) / dy_;

  dist_ = 0;

  step_num_ = 0;

  // Avoids an infinite loop.
  if (stepX_ == 0 && stepY_ == 0)
    return false;
  else
    return true;
}

bool RayCaster2D::nextId(Eigen::Vector2i& idx)
{
  auto tmp = Eigen::Vector2d(x_, y_);
  idx = (tmp + offset_).cast<int>();

  if (x_ == endX_ && y_ == endY_) {
    return false;
  }

  // tMaxX stores the t-value at which we cross a cube boundary along the
  // X axis, and similarly for Y. Therefore, choosing the least tMax
  // chooses the closest cube boundary.
  if (tMaxX_ < tMaxY_) {
    // Update which grid we are now in along the x-axis.
    x_ += stepX_;
    // Adjust tMaxX to the next X-oriented boundary crossing.
    tMaxX_ += tDeltaX_;
  }
  else {
    // Update which grid we are now in along the y-axis.
    y_ += stepY_;
    tMaxY_ += tDeltaY_;
  }

  return true;
}

bool RayCaster2D::nextPos(Eigen::Vector2d& pos)
{
  auto tmp = Eigen::Vector2d(x_, y_);
  pos = (tmp + half_) * resolution_;

  if (x_ == endX_ && y_ == endY_) {
    return false;
  }

  // tMaxX stores the t-value at which we cross a grid boundary along the
  // X axis, and similarly for Y. Choosing the least tMax chooses the closest
  // grid boundary.
  if (tMaxX_ < tMaxY_) {
    // Update which grid we are now in along the x-axis.
    x_ += stepX_;
    // Adjust tMaxX to the next X-oriented boundary crossing.
    tMaxX_ += tDeltaX_;
  }
  else {
    // Update which grid we are now in along the y-axis.
    y_ += stepY_;
    tMaxY_ += tDeltaY_;
  }

  return true;
}