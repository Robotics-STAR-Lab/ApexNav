#ifndef _GCOPTER_HPP_
#define _GCOPTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

#include "plan_env/sdf_map2d.h"
#include "path_searching/kino_astar.h"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "gcopter/trajectory.hpp"
#include "gcopter/minco.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/sfc_gen.hpp"

#define uint unsigned int
namespace apexnav_planner {
struct Config {
  // Vehicle parameters
  double max_vel_;
  double max_acc_;
  double max_domega_;
  double wheel_base_;
  double non_siguav_;
  double zoom_omega_;

  // Corridor parameters
  int denseResolution_;
  int sparseResolution_;
  double timeResolution_;

  Config(rclcpp::Node::SharedPtr node)
  {
    if (!node->has_parameter("optimizer.max_vel")) {
      node->declare_parameter("optimizer.max_vel", 5.0);
    }
    if (!node->has_parameter("optimizer.max_acc")) {
      node->declare_parameter("optimizer.max_acc", 5.0);
    }
    if (!node->has_parameter("optimizer.max_domega")) {
      node->declare_parameter("optimizer.max_domega", 50.0);
    }
    if (!node->has_parameter("optimizer.wheel_base")) {
      node->declare_parameter("optimizer.wheel_base", 0.8);
    }
    if (!node->has_parameter("optimizer.non_siguav")) {
      node->declare_parameter("optimizer.non_siguav", 1.0);
    }
    if (!node->has_parameter("optimizer.zoom_omega")) {
      node->declare_parameter("optimizer.zoom_omega", 1.0);
    }
    if (!node->has_parameter("optimizer.denseResolution")) {
      node->declare_parameter("optimizer.denseResolution", 20);
    }
    if (!node->has_parameter("optimizer.sparseResolution")) {
      node->declare_parameter("optimizer.sparseResolution", 8);
    }
    if (!node->has_parameter("optimizer.timeResolution")) {
      node->declare_parameter("optimizer.timeResolution", 1.0);
    }

    node->get_parameter("optimizer.max_vel", max_vel_);
    node->get_parameter("optimizer.max_acc", max_acc_);
    node->get_parameter("optimizer.max_domega", max_domega_);
    node->get_parameter("optimizer.wheel_base", wheel_base_);
    node->get_parameter("optimizer.non_siguav", non_siguav_);
    node->get_parameter("optimizer.zoom_omega", zoom_omega_);
    node->get_parameter("optimizer.denseResolution", denseResolution_);
    node->get_parameter("optimizer.sparseResolution", sparseResolution_);
    node->get_parameter("optimizer.timeResolution", timeResolution_);
  }
};

struct LocalTrajectory {
  Trajectory<7, 3> traj;
  int traj_id;
  rclcpp::Time start_time;
  double duration;

  LocalTrajectory()
  {
    traj_id = 0;
    start_time = rclcpp::Time(0);
    duration = 0;
  }
};

class Gcopter {
private:
  Config config_;
  rclcpp::Node::SharedPtr node_;
  SDFMap2D::Ptr map_;
  std::shared_ptr<KinoAstar> kinoastar_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr inner_point_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr inner_init_point_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr minco_init_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr minco_init_path_alpha_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr minco_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr minco_opt_path_alpha_pub_;

  // Optimization parameters
  double rho_;
  double v_weight_, a_weight_, omega_weight_, colli_weight_, domega_weight_;
  double safe_dist_;

  int piece_singul_num_;
  std::vector<Eigen::VectorXd> pieceTimes;
  std::vector<Eigen::MatrixXd> innerPointses;
  Eigen::VectorXi singuls;

  std::vector<Eigen::MatrixXd> iniStates;
  std::vector<Eigen::MatrixXd> finStates;
  Eigen::VectorXi eachTrajNums;

  std::vector<Eigen::MatrixXd> finalInnerpointses;
  std::vector<Eigen::VectorXd> finalpieceTimes;

  double Freedom_;

  std::vector<minco::MINCO_S4NU> mincos;

  // statelists -> statelist -> state
  std::vector<std::vector<Eigen::Vector4d>> statelists;

  // Optimization process parameters
  int iter_num_;
  // Store gradients: for q, T with c added, c, T without c added
  Eigen::Matrix3Xd gradByPoints;  // joint_piece_*(piece_num_-1)
  Eigen::VectorXd gradByTimes;
  Eigen::MatrixX3d partialGradByCoeffs;  //(2s*piece_num_) * joint_piece_
  Eigen::VectorXd partialGradByTimes;

  // Auxiliary for derivatives
  Eigen::Matrix2d B_h;

  bool ifprint = false;

  // Helper function for quaternion from yaw
  inline geometry_msgs::msg::Quaternion createQuaternionMsgFromYaw(double yaw)
  {
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    geometry_msgs::msg::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
  }

public:
  // Results
  Trajectory<7, 3> final_traj;
  std::vector<Trajectory<7, 3>> final_trajes;
  std::vector<Trajectory<7, 3>> init_final_trajes;
  Eigen::VectorXi final_singuls;
  LocalTrajectory local_trajectory_;

  Gcopter(const Config& conf, rclcpp::Node::SharedPtr node, const SDFMap2D::Ptr& map,
      std::shared_ptr<KinoAstar> kinoastar)
    : config_(conf)
  {
    Freedom_ = 3;
    node_ = node;
    map_ = map;
    kinoastar_ = kinoastar;
    minco_init_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("/trajectory/minco_init_path", 10);
    minco_init_path_alpha_pub_ =
        node_->create_publisher<nav_msgs::msg::Path>("/trajectory/minco_init_path_alpha_pub_", 10);
    minco_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("/trajectory/mincoPath", 10);
    minco_opt_path_alpha_pub_ =
        node_->create_publisher<visualization_msgs::msg::MarkerArray>("/trajectory/minco_opt_path_alpha_pub_", 10);
    inner_point_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/trajectory/innerpoint", 10);
    inner_init_point_pub_ =
        node_->create_publisher<visualization_msgs::msg::MarkerArray>("/trajectory/initinnerpoint", 10);

    // Read parameters
    if (!node_->has_parameter("optimizer.time_weight")) {
      node_->declare_parameter("optimizer.time_weight", 1.0);
    }
    if (!node_->has_parameter("optimizer.safe_dist")) {
      node_->declare_parameter("optimizer.safe_dist", 0.0);
    }
    if (!node_->has_parameter("optimizer.penaltyWeights")) {
      node_->declare_parameter("optimizer.penaltyWeights", std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0});
    }

    node_->get_parameter("optimizer.time_weight", rho_);
    node_->get_parameter("optimizer.safe_dist", safe_dist_);
    std::vector<double> penaWt;
    node_->get_parameter("optimizer.penaltyWeights", penaWt);
    if (penaWt.size() >= 5) {
      v_weight_ = penaWt[0];
      a_weight_ = penaWt[1];
      omega_weight_ = penaWt[2];
      domega_weight_ = penaWt[3];
      colli_weight_ = penaWt[4];
    }

    B_h << 0, -1, 1, 0;
  }

  inline void minco_plan()
  {
    if (!(kinoastar_->has_path_)) {
      RCLCPP_ERROR(node_->get_logger(), "There is no kinoastar path!!!!!!!!!!!!");
      return;
    }
    getState();
    visInnerPoints();

    optimizer();
    visFinalInnerPoints();
    local_trajectory_.traj_id++;
    local_trajectory_.start_time = node_->get_clock()->now();
    local_trajectory_.duration = final_trajes[0].getTotalDuration();
    local_trajectory_.traj = final_trajes[0];
  }

  inline void getState()
  {
    innerPointses.clear();
    iniStates.clear();
    finStates.clear();
    finalInnerpointses.clear();
    finalpieceTimes.clear();
    statelists.clear();

    double basetime = 0.0;
    piece_singul_num_ = kinoastar_->flat_trajs_.size();
    pieceTimes.resize(piece_singul_num_);
    singuls.resize(piece_singul_num_);
    eachTrajNums.resize(piece_singul_num_);

    // Store all intermediate points for each segment: 3 * piece_nums-1
    Eigen::MatrixXd ego_innerPs;

    for (int i = 0; i < piece_singul_num_; i++) {
      FlatTrajData kino_traj = kinoastar_->flat_trajs_.at(i);
      singuls[i] = kino_traj.singul;

      // Trajectory points for segment i
      std::vector<Eigen::Vector4d> pts = kino_traj.traj_pts;
      // Time for segment i trajectory (from frontend)
      double initTotalduration = 0.0;
      for (const auto pt : pts) initTotalduration += pt[2];

      // Resegment: round according to time resolution, but at least mintrajNum_ segments
      int piece_nums = std::max(int(initTotalduration / config_.timeResolution_ + 0.5), 2);
      // Evenly divide into smaller segments
      double timePerPiece = initTotalduration / piece_nums;
      // Store total time for each large segment
      Eigen::VectorXd piecetime;
      piecetime.resize(piece_nums);
      piecetime.setConstant(timePerPiece);
      pieceTimes[i] = piecetime;

      ego_innerPs.resize(3, piece_nums - 1);
      // States after dividing large segments into small ones and then evenly subdividing
      std::vector<Eigen::Vector4d> statelist;
      double res_time = 0;
      // Loop through small segments
      for (int j = 0; j < piece_nums; j++) {
        // Uniform density sampling
        int resolution = config_.sparseResolution_;
        // Get positions at time nodes after uniform subdivision and store in statelist, put segment endpoints into ego_innerPs
        for (int k = 0; k <= resolution; k++) {
          // Time for k-th small segment after sampling: basetime is total time, res_time is timing for loop segments
          double t = basetime + res_time + 1.0 * k / resolution * timePerPiece;
          // Get state coordinates (x,y,yaw,maniangle) at time t through interpolation
          Eigen::Vector4d pos = kinoastar_->evaluatePos(t);
          statelist.push_back(pos);
          if (k == resolution && j != piece_nums - 1)
            ego_innerPs.col(j) = Eigen::Vector3d(pos[0], pos[1], pos[3]);
        }
        res_time += timePerPiece;
      }

      statelists.push_back(statelist);       // Store complete trajectory point set (for subsequent optimization)
      innerPointses.push_back(ego_innerPs);  // Store intermediate points after segmentation
      iniStates.push_back(kino_traj.start_state);  // Store initial state
      finStates.push_back(kino_traj.final_state);  // Store final state
      eachTrajNums[i] = piece_nums;                // Record number of segments for each trajectory
      basetime += initTotalduration;
    }
  }

  inline void optimizer()
  {
    if ((int)innerPointses.size() != piece_singul_num_ ||
        (int)singuls.size() != piece_singul_num_ || (int)iniStates.size() != piece_singul_num_ ||
        (int)finStates.size() != piece_singul_num_ ||
        (int)eachTrajNums.size() != piece_singul_num_ ||
        (int)pieceTimes.size() != piece_singul_num_) {
      RCLCPP_ERROR(node_->get_logger(), "[Optimizer ERROR]");
      RCLCPP_ERROR(node_->get_logger(), "piece_singul_num_: %d", piece_singul_num_);
      RCLCPP_ERROR(node_->get_logger(), "innerPointses.size(): %ld", innerPointses.size());
      RCLCPP_ERROR(node_->get_logger(), "singuls.size(): %ld", singuls.size());
      RCLCPP_ERROR(node_->get_logger(), "iniStates.size(): %ld", iniStates.size());
      RCLCPP_ERROR(node_->get_logger(), "finStates.size(): %ld", finStates.size());
      RCLCPP_ERROR(node_->get_logger(), "eachTrajNums.size(): %ld", eachTrajNums.size());
      RCLCPP_ERROR(node_->get_logger(), "pieceTimes.size(): %ld", pieceTimes.size());
      return;
    }

    int variable_num_ = 0;
    mincos.clear();
    mincos.resize(piece_singul_num_);
    for (int i = 0; i < piece_singul_num_; i++) {
      if (innerPointses[i].cols() == 0) {
        RCLCPP_ERROR(node_->get_logger(), "[optimizer ERROR] no Innerpoint!");
        return;
      }
      int piece_num = eachTrajNums[i];

      if (iniStates[i].col(1).norm() >= config_.max_vel_)
        iniStates[i].col(1) = iniStates[i].col(1).normalized() * (config_.max_vel_ - 1.0e-2);
      if (iniStates[i].col(2).norm() >= config_.max_acc_)
        iniStates[i].col(2) = iniStates[i].col(2).normalized() * (config_.max_acc_ - 1.0e-2);
      if (finStates[i].col(1).norm() >= config_.max_vel_)
        finStates[i].col(1) = finStates[i].col(1).normalized() * (config_.max_vel_ - 1.0e-2);
      if (finStates[i].col(2).norm() >= config_.max_acc_)
        finStates[i].col(2) = finStates[i].col(2).normalized() * (config_.max_acc_ - 1.0e-2);

      variable_num_ += (Freedom_ + 1) * (piece_num - 1) + 1;

      mincos[i].setConditions(iniStates[i], finStates[i], piece_num);
    }

    // Initial trajectory
    init_final_trajes.clear();
    for (int i = 0; i < piece_singul_num_; i++) {
      mincos[i].setParameters(innerPointses[i], pieceTimes[i]);

      Trajectory<7, 3> traj;
      mincos[i].getTrajectory(traj);
      init_final_trajes.push_back(traj);
    }
    mincoInitTrajPub(init_final_trajes, singuls);

    // minco
    Eigen::VectorXd x;
    x.resize(variable_num_);
    int offset = 0;
    for (int i = 0; i < piece_singul_num_; i++) {
      memcpy(x.data() + offset, innerPointses[i].data(), innerPointses[i].size() * sizeof(x[0]));
      offset += innerPointses[i].size();
    }
    for (int i = 0; i < piece_singul_num_; i++) {
      Eigen::Map<Eigen::VectorXd> Vt(x.data() + offset, pieceTimes[i].size());
      offset += pieceTimes[i].size();
      RealT2VirtualT(pieceTimes[i], Vt);
    }

    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs_params.mem_size = 256;
    lbfgs_params.past = 5;
    lbfgs_params.g_epsilon = 0.0;
    lbfgs_params.min_step = 1.0e-32;
    lbfgs_params.delta = 1.0e-6;
    lbfgs_params.max_iterations = 5000;

    Eigen::VectorXd g;
    g.resize(x.size());

    ifprint = false;
    iter_num_ = 0;
    double cost;
    int result = lbfgs::lbfgs_optimize(
        x, cost, Gcopter::costFunctionCallback, NULL, NULL, this, lbfgs_params);

    ifprint = false;
    costFunctionCallback(this, x, g);

    offset = 0;
    final_trajes.clear();

    // Compute minimum snap trajectory with optimized waypoints and time
    for (int i = 0; i < piece_singul_num_; i++) {
      Eigen::Map<Eigen::MatrixXd> P(x.data() + offset, 3, eachTrajNums[i] - 1);
      offset += 3 * (eachTrajNums[i] - 1);
      finalInnerpointses.emplace_back(P);
    }

    for (int i = 0; i < piece_singul_num_; i++) {
      Eigen::Map<const Eigen::VectorXd> t(x.data() + offset, eachTrajNums[i]);
      Eigen::VectorXd T;
      offset += eachTrajNums[i];
      VirtualT2RealT(t, T);
      finalpieceTimes.emplace_back(T);
      mincos[i].setParameters(finalInnerpointses[i], T);
      mincos[i].getTrajectory(final_traj);
      final_trajes.push_back(final_traj);
    }
    final_singuls = singuls;
  }

  template <typename EIGENVEC>
  inline void RealT2VirtualT(const Eigen::VectorXd& RT, EIGENVEC& VT)
  {
    const int sizeT = RT.size();
    VT.resize(sizeT);
    for (int i = 0; i < sizeT; ++i) {
      VT(i) = RT(i) > 1.0 ? (sqrt(2.0 * RT(i) - 1.0) - 1.0) : (1.0 - sqrt(2.0 / RT(i) - 1.0));
    }
  }

  template <typename EIGENVEC>
  inline void VirtualT2RealT(const EIGENVEC& VT, Eigen::VectorXd& RT)
  {
    const int sizeTau = VT.size();
    RT.resize(sizeTau);
    for (int i = 0; i < sizeTau; ++i) {
      RT(i) = VT(i) > 0.0 ? ((0.5 * VT(i) + 1.0) * VT(i) + 1.0) :
                            1.0 / ((0.5 * VT(i) - 1.0) * VT(i) + 1.0);
    }
  }

  static inline int earlyExit(void* instance, const Eigen::VectorXd& x, const Eigen::VectorXd& g,
      const double fx, const double step, const int k, const int ls)
  {
    if (!rclcpp::ok()) {
      return 1;
    }

    Gcopter& obj = *(Gcopter*)instance;
    obj.innerPointses.clear();
    std::vector<Eigen::VectorXd> t_container;
    obj.pieceTimes.clear();

    // Map input variables to variable matrices
    int offset = 0;
    Eigen::Map<const Eigen::MatrixXd> P(x.data() + offset, 3, obj.eachTrajNums[0] - 1);
    offset += 3 * (obj.eachTrajNums[0] - 1);
    obj.innerPointses.emplace_back(P);

    Eigen::VectorXd T;
    Eigen::Map<const Eigen::VectorXd> t(x.data() + offset, obj.eachTrajNums[0]);
    offset += obj.eachTrajNums[0];
    obj.VirtualT2RealT(t, T);
    obj.pieceTimes.push_back(T);
    t_container.emplace_back(t);

    int traj_id = 0;
    obj.mincos[traj_id].setParameters(obj.innerPointses[traj_id], obj.pieceTimes[traj_id]);
    obj.init_final_trajes.clear();
    obj.mincos[traj_id].setParameters(obj.innerPointses[traj_id], obj.pieceTimes[traj_id]);
    Trajectory<7, 3> traj;
    obj.mincos[traj_id].getTrajectory(traj);
    obj.init_final_trajes.push_back(traj);
    obj.mincoInitTrajPub(obj.init_final_trajes, obj.singuls);
    obj.mincoInitPathPubwithAlpha(obj.init_final_trajes, obj.singuls, k);
    std::cout << "iter num:" << k << std::endl;
    return 0;
  }

  static double costFunctionCallback(void* ptr, const Eigen::VectorXd& x, Eigen::VectorXd& g)
  {
    if (x.norm() > 1e4)
      return inf;

    Gcopter& obj = *(Gcopter*)ptr;
    obj.iter_num_ += 1;

    obj.innerPointses.clear();
    std::vector<Eigen::Map<Eigen::MatrixXd>> gradP_container;
    std::vector<Eigen::VectorXd> t_container;
    obj.pieceTimes.clear();
    std::vector<Eigen::Map<Eigen::VectorXd>> gradt_container;

    g.setZero();
    // Map input variables to variable matrices
    int offset = 0;
    Eigen::Map<const Eigen::MatrixXd> P(x.data() + offset, 3, obj.eachTrajNums[0] - 1);
    Eigen::Map<Eigen::MatrixXd> gradP(g.data() + offset, 3, obj.eachTrajNums[0] - 1);
    offset += 3 * (obj.eachTrajNums[0] - 1);
    gradP.setZero();
    obj.innerPointses.emplace_back(P);
    gradP_container.push_back(gradP);

    Eigen::VectorXd T;
    Eigen::Map<const Eigen::VectorXd> t(x.data() + offset, obj.eachTrajNums[0]);
    Eigen::Map<Eigen::VectorXd> gradt(g.data() + offset, obj.eachTrajNums[0]);
    offset += obj.eachTrajNums[0];
    obj.VirtualT2RealT(t, T);
    gradt.setZero();
    obj.pieceTimes.push_back(T);
    t_container.emplace_back(t);
    gradt_container.push_back(gradt);

    double cost_of_all = 0;

    // Since only optimizing single trajectory, set to 0
    int traj_id = 0;
    double cost;
    obj.mincos[traj_id].setParameters(obj.innerPointses[traj_id], obj.pieceTimes[traj_id]);
    obj.mincos[traj_id].getEnergy(cost);

    // Calculate trajectory gradient with respect to control points and time variables
    obj.mincos[traj_id].getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
    obj.mincos[traj_id].getEnergyPartialGradByTimes(obj.partialGradByTimes);
    if (obj.ifprint)
      std::cout << "Energy cost:" << cost << std::endl;

    // Add additional constraints or penalty terms
    obj.attachPenaltyFunctional(traj_id, cost);
    if (obj.ifprint)
      std::cout << "attachPenaltyFunctional cost:" << cost << std::endl;

    // Calculate gradient
    obj.mincos[traj_id].propogateGrad(
        obj.partialGradByCoeffs, obj.partialGradByTimes, obj.gradByPoints, obj.gradByTimes);

    // Add regularization for time duration
    cost += obj.rho_ * obj.pieceTimes[traj_id].sum();
    if (obj.ifprint)
      std::cout << "T cost:" << obj.rho_ * obj.pieceTimes[traj_id].sum() << std::endl;

    Eigen::VectorXd rho_times;
    rho_times.resize(obj.gradByTimes.size());
    obj.gradByTimes += obj.rho_ * rho_times.setOnes();

    gradP_container[traj_id] = obj.gradByPoints;
    backwardGradT(t_container[traj_id], obj.gradByTimes, gradt_container[traj_id]);
    cost_of_all += cost;

    obj.ifprint = false;

    return cost_of_all;
  }

  // Gradients for partialGradByCoeffs and partialGradByTimes
  void attachPenaltyFunctional(const int& traj_id, double& cost);

  inline void positiveSmoothedL1(const double& x, double& f, double& df)
  {
    const double pe = 1e-3;
    const double half = 0.5 * pe;
    const double f3c = 1.0 / (pe * pe);
    const double f4c = -0.5 * f3c / pe;
    const double d2c = 3.0 * f3c;
    const double d3c = 4.0 * f4c;

    if (x < pe) {
      f = (f4c * x + f3c) * x * x * x;
      df = (d3c * x + d2c) * x * x;
    }
    else {
      f = x - half;
      df = 1.0;
    }
  }

  inline void activationSmoothed(const double& x, double& f, double& df)
  {
    double mu = 0.01;
    double mu4_1 = 1.0 / (mu * mu * mu * mu);
    if (x < -mu) {
      df = 0;
      f = 0;
    }
    else if (x < 0) {
      double y = x + mu;
      double y2 = y * y;
      df = y2 * (mu - 2 * x) * mu4_1;
      f = 0.5 * y2 * y * (mu - x) * mu4_1;
    }
    else if (x < mu) {
      double y = x - mu;
      double y2 = y * y;
      df = y2 * (mu + 2 * x) * mu4_1;
      f = 0.5 * y2 * y * (mu + x) * mu4_1 + 1;
    }
    else {
      df = 0;
      f = 1;
    }
  }

  template <typename EIGENVEC>
  static inline void backwardGradT(
      const Eigen::VectorXd& tau, const Eigen::VectorXd& gradT, EIGENVEC& gradTau)
  {
    const int sizetau = tau.size();
    gradTau.resize(sizetau);
    double gradrt2vt;
    for (int i = 0; i < sizetau; i++) {
      if (tau(i) > 0) {
        gradrt2vt = tau(i) + 1.0;
      }
      else {
        double denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
        gradrt2vt = (1.0 - tau(i)) / (denSqrt * denSqrt);
      }
      gradTau(i) = gradT(i) * gradrt2vt;
    }
    return;
  }

  void visInnerPoints();
  void visFinalInnerPoints();
  void mincoInitTrajPub(const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls);
  void mincoInitPathPubwithAlpha(const std::vector<Trajectory<7, 3>>& final_trajes,
      const Eigen::VectorXi& final_singuls, const int& k);
  void mincoPathPub(const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls);
};
}  // namespace apexnav_planner
#endif
