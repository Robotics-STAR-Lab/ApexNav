/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef SFC_GEN_HPP
#define SFC_GEN_HPP

#include "geo_utils.hpp"
#include "firi.hpp"

#include <ompl/util/Console.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/DiscreteMotionValidator.h>

#include <deque>
#include <memory>
#include <Eigen/Eigen>

namespace sfc_gen {
// 输入：起点  终点  Map最小点  Map最大点  地图
template <typename Map>
inline double planPath(const Eigen::Vector3d& s, const Eigen::Vector3d& g,
    const Eigen::Vector3d& lb, const Eigen::Vector3d& hb, const Map* mapPtr, const double& timeout,
    std::vector<Eigen::Vector3d>& p)
{
  // ompl搜索路径 表示 R^n 的状态空间。 距离函数是 L2 范数。
  auto space(std::make_shared<ompl::base::RealVectorStateSpace>(3));

  ompl::base::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, hb(0) - lb(0));
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, hb(1) - lb(1));
  bounds.setLow(2, 0.0);
  bounds.setHigh(2, hb(2) - lb(2));
  space->setBounds(bounds);

  //空间信息的基类。 这包含有关空间规划的所有信息。在使用之前还需要调用 setup()
  auto si(std::make_shared<ompl::base::SpaceInformation>(space));

  // 设置ompl的检查  这里用匿名函数计算碰撞
  si->setStateValidityChecker([&](const ompl::base::State* state) {
    const auto* pos = state->as<ompl::base::RealVectorStateSpace::StateType>();
    const Eigen::Vector3d position(lb(0) + (*pos)[0], lb(1) + (*pos)[1], lb(2) + (*pos)[2]);
    return mapPtr->query(position) == 0;
  });
  si->setup();
  //设置要输出的记录数据的最低级别。 日志级别较低的消息将不会被记录。？？
  ompl::msg::setLogLevel(ompl::msg::LOG_NONE);

  ompl::base::ScopedState<> start(space), goal(space);
  start[0] = s(0) - lb(0);
  start[1] = s(1) - lb(1);
  start[2] = s(2) - lb(2);
  goal[0] = g(0) - lb(0);
  goal[1] = g(1) - lb(1);
  goal[2] = g(2) - lb(2);

  auto pdef(std::make_shared<ompl::base::ProblemDefinition>(si));
  pdef->setStartAndGoalStates(start, goal);
  pdef->setOptimizationObjective(std::make_shared<ompl::base::PathLengthOptimizationObjective>(si));
  auto planner(std::make_shared<ompl::geometric::InformedRRTstar>(si));
  planner->setProblemDefinition(pdef);
  planner->setup();

  ompl::base::PlannerStatus solved;
  solved = planner->ompl::base::Planner::solve(timeout);

  double cost = INFINITY;
  if (solved) {
    p.clear();
    const ompl::geometric::PathGeometric path_ = ompl::geometric::PathGeometric(
        dynamic_cast<const ompl::geometric::PathGeometric&>(*pdef->getSolutionPath()));
    for (size_t i = 0; i < path_.getStateCount(); i++) {
      const auto state =
          path_.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>()->values;
      p.emplace_back(lb(0) + state[0], lb(1) + state[1], lb(2) + state[2]);
    }
    cost = pdef->getSolutionPath()->cost(pdef->getOptimizationObjective()).value();
  }

  return cost;
}

// 输入：
inline void convexCover(const std::vector<Eigen::Vector3d>& path,  // 前端路径
    const std::vector<Eigen::Vector3d>& points,                    // surf实际坐标
    const Eigen::Vector3d& lowCorner,                              //  三维坐标最小值
    const Eigen::Vector3d& highCorner,                             //  三维坐标最大值
    const double& progress,                                        // 7
    const double& range,                                           // 3
    std::vector<Eigen::MatrixX4d>& hpolys,                         //  输出的超平面们
    const double eps = 1.0e-6)
{
  hpolys.clear();
  const int n = path.size();
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  bd(0, 0) = 1.0;
  bd(1, 0) = -1.0;
  bd(2, 1) = 1.0;
  bd(3, 1) = -1.0;
  bd(4, 2) = 1.0;
  bd(5, 2) = -1.0;

  Eigen::MatrixX4d hp, gap;
  Eigen::Vector3d a, b = path[0];
  std::vector<Eigen::Vector3d> valid_pc;
  std::vector<Eigen::Vector3d> bs;
  valid_pc.reserve(points.size());
  for (int i = 1; i < n;) {
    a = b;
    // 如果两点的距离小于progress就采用该点，否则在a与当前点之间截取progress长度
    if ((a - path[i]).norm() > progress) {
      b = (path[i] - a).normalized() * progress + a;
    }
    else {
      b = path[i];
      i++;
    }
    bs.emplace_back(b);
    // 组成一个立方体
    bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
    bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
    bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
    bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
    bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
    bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

    valid_pc.clear();
    for (const Eigen::Vector3d& p : points) {
      // p在a b最大坐标+range 和 最小坐标-range内，则将这个点放进valid_pc
      if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0) {
        valid_pc.emplace_back(p);
      }
    }
    // 将valid_pc变成一个3*n的矩阵pc
    Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(
        valid_pc[0].data(), 3, valid_pc.size());

    firi::firi(bd, pc, a, b, hp);

    if (hpolys.size() != 0) {
      const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
      if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                   ((hpolys.back() * ah).array() > -eps).cast<int>().sum()) {
        firi::firi(bd, pc, a, a, gap, 1);
        hpolys.emplace_back(gap);
      }
    }

    hpolys.emplace_back(hp);
  }
}

inline void shortCut(std::vector<Eigen::MatrixX4d>& hpolys)
{
  std::vector<Eigen::MatrixX4d> htemp = hpolys;
  if (htemp.size() == 1) {
    Eigen::MatrixX4d headPoly = htemp.front();
    htemp.insert(htemp.begin(), headPoly);
  }
  hpolys.clear();

  int M = htemp.size();
  Eigen::MatrixX4d hPoly;
  bool overlap;
  std::deque<int> idices;
  idices.push_front(M - 1);
  for (int i = M - 1; i >= 0; i--) {
    for (int j = 0; j < i; j++) {
      if (j < i - 1) {
        overlap = geo_utils::overlap(htemp[i], htemp[j], 0.01);
      }
      else {
        overlap = true;
      }
      if (overlap) {
        idices.push_front(j);
        i = j + 1;
        break;
      }
    }
  }
  for (const auto& ele : idices) {
    hpolys.push_back(htemp[ele]);
  }
}

}  // namespace sfc_gen

#endif
