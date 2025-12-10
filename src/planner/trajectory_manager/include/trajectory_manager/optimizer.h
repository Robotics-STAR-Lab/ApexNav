#ifndef _GCOPTER_HPP_
#define _GCOPTER_HPP_

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include "tf/transform_datatypes.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

#include "plan_env/sdf_map2d.h"
#include "path_searching/kino_astar.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

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

  Config(const ros::NodeHandle& nh_)
  {
    nh_.param<double>(ros::this_node::getName() + "/max_vel", max_vel_, 5);
    nh_.param<double>(ros::this_node::getName() + "/max_acc", max_acc_, 5);
    nh_.param<double>(ros::this_node::getName() + "/max_domega", max_domega_, 50);
    nh_.param<double>(ros::this_node::getName() + "/wheel_base", wheel_base_, 0.8);
    nh_.param<double>(ros::this_node::getName() + "/non_siguav", non_siguav_, 1);
    nh_.param<double>(ros::this_node::getName() + "/zoom_omega", zoom_omega_, 1);

    nh_.param<int>(ros::this_node::getName() + "/denseResolution", denseResolution_, 20);
    nh_.param<int>(ros::this_node::getName() + "/sparseResolution", sparseResolution_, 8);
    nh_.param<double>(ros::this_node::getName() + "/timeResolution", timeResolution_, 1);
  }
};

struct LocalTrajectory {
  Trajectory<7, 3> traj;
  int traj_id;
  ros::Time start_time;
  double duration;

  LocalTrajectory()
  {
    traj_id = 0;
    start_time = ros::Time(0);
    duration = 0;
  }
};

class Gcopter {
private:
  Config config_;
  ros::NodeHandle nh_;
  SDFMap2D::Ptr map_;
  std::shared_ptr<KinoAstar> kinoastar_;

  ros::Publisher inner_point_pub_;
  ros::Publisher inner_init_point_pub_;
  ros::Publisher minco_init_path_pub_;
  ros::Publisher minco_init_path_alpha_pub_;
  ros::Publisher minco_path_pub_;
  ros::Publisher minco_opt_path_alpha_pub_;

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

public:
  // Results
  Trajectory<7, 3> final_traj;
  std::vector<Trajectory<7, 3>> final_trajes;
  std::vector<Trajectory<7, 3>> init_final_trajes;
  Eigen::VectorXi final_singuls;
  LocalTrajectory local_trajectory_;

  Gcopter(const Config& conf, ros::NodeHandle& nh, const SDFMap2D::Ptr& map,
      std::shared_ptr<KinoAstar> kinoastar)
    : config_(conf)
  {
    Freedom_ = 3;
    nh_ = nh;
    map_ = map;
    kinoastar_ = kinoastar;
    minco_init_path_pub_ = nh_.advertise<nav_msgs::Path>("/trajectory/minco_init_path", 10);
    minco_init_path_alpha_pub_ =
        nh_.advertise<nav_msgs::Path>("/trajectory/minco_init_path_alpha_pub_", 10);
    minco_path_pub_ = nh_.advertise<nav_msgs::Path>("/trajectory/mincoPath", 10);
    minco_opt_path_alpha_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/trajectory/minco_opt_path_alpha_pub_", 10);
    inner_point_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/trajectory/innerpoint", 10);
    inner_init_point_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/trajectory/initinnerpoint", 10);

    // Read parameters
    nh_.param<double>(ros::this_node::getName() + "/time_weight", rho_, 1);
    nh_.param<double>(ros::this_node::getName() + "/safe_dist", safe_dist_, 0);
    std::vector<double> penaWt;
    nh_.getParam(ros::this_node::getName() + "/penaltyWeights", penaWt);
    v_weight_ = penaWt[0];
    a_weight_ = penaWt[1];
    omega_weight_ = penaWt[2];
    domega_weight_ = penaWt[3];
    colli_weight_ = penaWt[4];

    B_h << 0, -1, 1, 0;
  }

  inline void minco_plan()
  {
    if (!(kinoastar_->has_path_)) {
      ROS_ERROR("There is no kinoastar path!!!!!!!!!!!!");
      return;
    }
    ros::Time current = ros::Time::now();
    getState();
    visInnerPoints();

    current = ros::Time::now();
    optimizer();
    visFinalInnerPoints();
    local_trajectory_.traj_id++;
    local_trajectory_.start_time = ros::Time::now();
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
    // ROS_INFO("start optimizer!");
    if ((int)innerPointses.size() != piece_singul_num_ ||
        (int)singuls.size() != piece_singul_num_ || (int)iniStates.size() != piece_singul_num_ ||
        (int)finStates.size() != piece_singul_num_ ||
        (int)eachTrajNums.size() != piece_singul_num_ ||
        (int)pieceTimes.size() != piece_singul_num_) {
      ROS_ERROR("[Optimizer ERROR]");
      ROS_ERROR("piece_singul_num_: %d", piece_singul_num_);
      ROS_ERROR("innerPointses.size(): %ld", innerPointses.size());
      ROS_ERROR("singuls.size(): %ld", singuls.size());
      ROS_ERROR("iniStates.size(): %ld", iniStates.size());
      ROS_ERROR("finStates.size(): %ld", finStates.size());
      ROS_ERROR("eachTrajNums.size(): %ld", eachTrajNums.size());
      ROS_ERROR("pieceTimes.size(): %ld", pieceTimes.size());
      return;
    }

    int variable_num_ = 0;
    mincos.clear();
    mincos.resize(piece_singul_num_);
    // ROS_INFO("minco::MINCO_S4NU Minco;");
    for (int i = 0; i < piece_singul_num_; i++) {
      if (innerPointses[i].cols() == 0) {
        ROS_ERROR("[optimizer ERROR] no Innerpoint!");
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

      // ROS_INFO("setConditions");
      // std::cout << "iniStates[i]:  " << iniStates[i] << std::endl;
      // std::cout << "finStates[i]:  " << finStates[i] << std::endl;
      // std::cout << "piece_num:  " << piece_num << std::endl;
      // std::cout << "Freedom_:  " << Freedom_ << std::endl;
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
    // std::cout << "-------------------------optimize-------------------------" << std::endl;
    iter_num_ = 0;
    double cost;
    // ROS_INFO("go in lbfgs optimize!");
    // std::cout << "x:  " << x.transpose() << std::endl;
    ros::Time current = ros::Time::now();
    // int result = lbfgs::lbfgs_optimize(x,
    //                                    cost,
    //                                    Gcopter::costFunctionCallback,
    //                                    NULL,
    //                                    Gcopter::earlyExit,
    //                                    this,
    //                                    lbfgs_params);
    int result = lbfgs::lbfgs_optimize(
        x, cost, Gcopter::costFunctionCallback, NULL, NULL, this, lbfgs_params);
    double mincotime = (ros::Time::now() - current).toSec();
    // ROS_INFO("\033[40;36m only minco optimizer time:%f s\033[0m", mincotime);
    // std::cout << "-------------------------final-------------------------" << std::endl;

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

    // std::cout << "T:    ";
    for (int i = 0; i < piece_singul_num_; i++) {
      Eigen::Map<const Eigen::VectorXd> t(x.data() + offset, eachTrajNums[i]);
      Eigen::VectorXd T;
      offset += eachTrajNums[i];
      VirtualT2RealT(t, T);
      finalpieceTimes.emplace_back(T);
      mincos[i].setParameters(finalInnerpointses[i], T);
      mincos[i].getTrajectory(final_traj);
      final_trajes.push_back(final_traj);
      // std::cout << "  ||||  " << T.transpose();
    }
    final_singuls = singuls;
    // std::cout << std::endl;
    // std::cout << "x: " << x.transpose() << std::endl;
    // std::cout << "g: " << g.transpose() << std::endl;
    // std::cout << "eachTrajNums: " << eachTrajNums.transpose() << std::endl;
    // ROS_INFO("\n\n optimizer finish! result:%d   finalcost:%f   iter_num_:%d\n\n ", result, cost,
    //     iter_num_);
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
    if (!ros::ok()) {
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
  void attachPenaltyFunctional(const int& traj_id, double& cost)
  {
    int N = eachTrajNums[traj_id];

    Eigen::Vector3d gradESDF;
    Eigen::Vector2d gradESDF2d;

    Eigen::Vector2d sigma, dsigma, ddsigma, dddsigma, ddddsigma;
    double vel2_reci, acc2;
    Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
    double s1, s2, s3, s4, s5, s6, s7;
    double step, alpha;
    Eigen::Matrix<double, 8, 2> gradViolaPc, gradViolaVc, gradViolaAc, gradViolaKLc, gradViolaKRc;
    double gradViolaPt, gradViolaVt, gradViolaAt, gradViolaKLt, gradViolaKRt;
    double violaPos, violaVel, violaAcc;
    double violaPosPenaD, violaVelPenaD, violaAccPenaD;
    double violaPosPena, violaVelPena, violaAccPena;

    // omega and domega
    double omega, max_omega_, domega;
    double zoom = config_.zoom_omega_;
    double violaOmegaL, violaOmegaR, violadOmegaL, violadOmegaR;

    double v_min = config_.non_siguav_;
    double violavmin;
    double violavminPena, violavminPenaD;
    double violaOmegaPenaL, violaOmegaPenaDL, violaOmegaPenaR, violaOmegaPenaDR;
    double violadOmegaPenaL, violadOmegaPenaDL, violadOmegaPenaR, violadOmegaPenaDR;

    double omg;

    double z_h0, z_h1, z_h2, z_h3, z_h4, z_h41, z_h5, z_h6;
    Eigen::Matrix2d ego_R;
    Eigen::Matrix2d help_L;

    int singul = singuls[traj_id];

    ////////////////////Continuous dense sampling
    double DenseMinV = 0.15;
    double violaDenseMinV;
    double violaDenseMinVPena, violaDenseMinVPenaD;
    Eigen::Matrix<double, 8, 2> gradDenseViolaC, gradDenseC;
    double gradDenseViolaT, gradDenseT;
    double Densealpha;
    Eigen::Matrix<double, 8, 1> Densebeta0, Densebeta1, Densebeta2, Densebeta3, Densebeta4;
    Eigen::Vector2d Densesigma, Densedsigma, Denseddsigma, Densedddsigma, Denseddddsigma;

    Eigen::VectorXd T = pieceTimes[traj_id];

    double cost_safe = 0, cost_v = 0, cost_a = 0, cost_omega = 0, cost_domega = 0;
    double cost_dense_v = 0, cost_dense_omega = 0, cost_dense_domega = 0;
    double dense_cost = 0, cost_dense_all = 0;
    double cost_mean_t = 0;

    for (int i = 0; i < N; ++i) {
      int K;
      K = config_.sparseResolution_;
      const Eigen::Matrix<double, 8, 3>& c = mincos[traj_id].getCoeffs().block<8, 3>(8 * i, 0);
      step = T[i] / K;

      s1 = 0.0;

      for (int j = 0; j <= K; ++j) {
        s2 = s1 * s1;
        s3 = s2 * s1;
        s4 = s2 * s2;
        s5 = s4 * s1;
        s6 = s3 * s3;
        s7 = s4 * s3;
        beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
        beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
        beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
        beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
        beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
        alpha = 1.0 / K * j;

        // update s1 for the next iteration
        s1 += step;

        sigma = c.block<8, 2>(0, 0).transpose() * beta0;
        dsigma = c.block<8, 2>(0, 0).transpose() * beta1;
        ddsigma = c.block<8, 2>(0, 0).transpose() * beta2;
        dddsigma = c.block<8, 2>(0, 0).transpose() * beta3;
        ddddsigma = c.block<8, 2>(0, 0).transpose() * beta4;

        omg = (j == 0 || j == K) ? 0.5 : 1.0;

        // some help values
        z_h0 = dsigma.norm();
        z_h1 = ddsigma.transpose() * dsigma;
        z_h2 = dddsigma.transpose() * dsigma;
        z_h3 = ddsigma.transpose() * B_h * dsigma;

        // avoid siguality
        vel2_reci = 1.0 / (z_h0 * z_h0);
        z_h0 = 1.0 / z_h0;

        z_h4 = z_h1 * vel2_reci;
        violaVel = 1.0 / vel2_reci - config_.max_vel_ * config_.max_vel_;
        acc2 = z_h1 * z_h1 * vel2_reci;
        violaAcc = acc2 - config_.max_acc_ * config_.max_acc_;

        // zmk
        omega = z_h3 * vel2_reci;
        max_omega_ = 2.0 * zoom * (config_.max_vel_ - dsigma.norm()) / config_.wheel_base_;
        violaOmegaL = omega - max_omega_;
        violaOmegaR = -omega - max_omega_;

        z_h41 = dddsigma.transpose() * B_h * dsigma;
        z_h5 = dddsigma.transpose() * B_h * ddsigma;
        z_h6 = ddddsigma.transpose() * B_h * dsigma;
        domega = z_h41 * vel2_reci - 2.0 * z_h3 * z_h1 * vel2_reci * vel2_reci;
        violadOmegaL = domega - config_.max_domega_;
        violadOmegaR = -domega - config_.max_domega_;

        ego_R << dsigma(0), -dsigma(1), dsigma(1), dsigma(0);
        ego_R = ego_R * z_h0 * singul;

        Eigen::Vector2d bpt = sigma;
        violaPos = -map_->getDistWithGrad(bpt, gradESDF2d) + safe_dist_;
        gradESDF2d = -gradESDF2d;

        if (violaPos > 0.0) {
          positiveSmoothedL1(violaPos, violaPosPena, violaPosPenaD);

          gradViolaPc = beta0 * gradESDF2d.transpose();
          gradViolaPt = alpha * gradESDF2d.transpose() * dsigma;

          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * colli_weight_ * violaPosPenaD * gradViolaPc;
          partialGradByTimes(i) +=
              omg * colli_weight_ * (violaPosPenaD * gradViolaPt * step + violaPosPena / K);
          cost += omg * step * colli_weight_ * violaPosPena;  // cost is the same
          cost_safe += omg * step * colli_weight_ * violaPosPena;
        }

        if (violaVel > 0.0) {
          positiveSmoothedL1(violaVel, violaVelPena, violaVelPenaD);

          gradViolaVc = 2.0 * beta1 * dsigma.transpose();
          gradViolaVt = 2.0 * alpha * z_h1;
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * v_weight_ * violaVelPenaD * gradViolaVc;
          partialGradByTimes(i) +=
              omg * v_weight_ * (violaVelPenaD * gradViolaVt * step + violaVelPena / K);
          cost += omg * step * v_weight_ * violaVelPena;
          cost_v += omg * step * v_weight_ * violaVelPena;
        }

        if (violaAcc > 0.0) {
          positiveSmoothedL1(violaAcc, violaAccPena, violaAccPenaD);
          gradViolaAc =
              2.0 * beta1 * (z_h4 * ddsigma.transpose() - z_h4 * z_h4 * dsigma.transpose()) +
              2.0 * beta2 * z_h4 * dsigma.transpose();
          gradViolaAt = 2.0 * alpha * (z_h4 * (ddsigma.squaredNorm() + z_h2) - z_h4 * z_h4 * z_h1);
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * a_weight_ * violaAccPenaD * gradViolaAc;
          partialGradByTimes(i) +=
              omg * a_weight_ * (violaAccPenaD * gradViolaAt * step + violaAccPena / K);
          cost += omg * step * a_weight_ * violaAccPena;
          cost_a += omg * step * a_weight_ * violaAccPena;
        }

        if (violaOmegaL > 0.0) {
          positiveSmoothedL1(violaOmegaL, violaOmegaPenaL, violaOmegaPenaDL);
          gradViolaKLc =
              beta1 * (vel2_reci * ddsigma.transpose() * B_h -
                          2 * vel2_reci * vel2_reci * z_h3 * dsigma.transpose() +
                          2.0 * zoom * dsigma.transpose() * sqrt(vel2_reci) / config_.wheel_base_) +
              beta2 * vel2_reci * dsigma.transpose() * B_h.transpose();
          gradViolaKLt =
              alpha *
              (vel2_reci * (dddsigma.transpose() * B_h * dsigma - 2 * vel2_reci * z_h3 * z_h1) +
                  2.0 * zoom * z_h1 * sqrt(vel2_reci) / config_.wheel_base_);
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * omega_weight_ * violaOmegaPenaDL * gradViolaKLc;
          partialGradByTimes(i) +=
              omg * omega_weight_ * (violaOmegaPenaDL * gradViolaKLt * step + violaOmegaPenaL / K);
          cost += omg * step * omega_weight_ * violaOmegaPenaL;
          cost_omega += omg * step * omega_weight_ * violaOmegaPenaL;
        }

        if (violaOmegaR > 0.0) {
          positiveSmoothedL1(violaOmegaR, violaOmegaPenaR, violaOmegaPenaDR);
          gradViolaKRc = -(
              beta1 * (vel2_reci * ddsigma.transpose() * B_h -
                          2 * vel2_reci * vel2_reci * z_h3 * dsigma.transpose() -
                          2.0 * zoom * dsigma.transpose() * sqrt(vel2_reci) / config_.wheel_base_) +
              beta2 * vel2_reci * dsigma.transpose() * B_h.transpose());
          gradViolaKRt =
              -alpha *
              (vel2_reci * (dddsigma.transpose() * B_h * dsigma - 2 * vel2_reci * z_h3 * z_h1) -
                  2.0 * zoom * z_h1 * sqrt(vel2_reci) / config_.wheel_base_);
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * omega_weight_ * violaOmegaPenaDR * gradViolaKRc;
          partialGradByTimes(i) +=
              omg * omega_weight_ * (violaOmegaPenaDR * gradViolaKRt * step + violaOmegaPenaR / K);
          cost += omg * step * omega_weight_ * violaOmegaPenaR;
          cost_omega += omg * step * omega_weight_ * violaOmegaPenaR;
        }

        if (violadOmegaL > 0.0) {
          positiveSmoothedL1(violadOmegaL, violadOmegaPenaL, violadOmegaPenaDL);
          gradViolaKLc =
              beta3 * (dsigma.transpose() * B_h.transpose() * vel2_reci) +
              -beta2 * 2.0 * vel2_reci * vel2_reci *
                  (z_h1 * dsigma.transpose() * B_h.transpose() + z_h3 * dsigma.transpose()) +
              beta1 *
                  ((dddsigma.transpose() * B_h * vel2_reci) -
                      vel2_reci * vel2_reci * 2.0 *
                          (z_h41 * dsigma.transpose() + z_h1 * ddsigma.transpose() * B_h +
                              z_h3 * ddsigma.transpose()) +
                      8.0 * z_h3 * z_h1 * dsigma.transpose() * vel2_reci * vel2_reci * vel2_reci);
          gradViolaKLt =
              alpha * ((z_h5 + z_h6) * vel2_reci +
                          -(ddsigma.squaredNorm() * z_h3 + z_h2 * z_h3 + z_h41 * 2.0 * z_h1) * 2.0 *
                              vel2_reci * vel2_reci +
                          8.0 * z_h3 * z_h1 * z_h1 * vel2_reci * vel2_reci * vel2_reci);
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * domega_weight_ * violadOmegaPenaDL * gradViolaKLc;
          partialGradByTimes(i) += omg * domega_weight_ *
                                   (violadOmegaPenaDL * gradViolaKLt * step + violadOmegaPenaL / K);
          cost += omg * step * domega_weight_ * violadOmegaPenaL;
          cost_domega += omg * step * domega_weight_ * violadOmegaPenaL;
        }

        if (violadOmegaR > 0.0) {
          positiveSmoothedL1(violadOmegaR, violadOmegaPenaR, violadOmegaPenaDR);
          gradViolaKRc =
              -beta3 * (dsigma.transpose() * B_h.transpose() * vel2_reci) +
              beta2 * 2.0 * vel2_reci * vel2_reci *
                  (z_h1 * dsigma.transpose() * B_h.transpose() + z_h3 * dsigma.transpose()) +
              -beta1 *
                  ((dddsigma.transpose() * B_h * vel2_reci) -
                      vel2_reci * vel2_reci * 2.0 *
                          (z_h41 * dsigma.transpose() + z_h1 * ddsigma.transpose() * B_h +
                              z_h3 * ddsigma.transpose()) +
                      8.0 * z_h3 * z_h1 * dsigma.transpose() * vel2_reci * vel2_reci * vel2_reci);
          gradViolaKRt =
              -alpha * ((z_h5 + z_h6) * vel2_reci +
                           -(ddsigma.squaredNorm() * z_h3 + z_h2 * z_h3 + z_h41 * 2.0 * z_h1) *
                               2.0 * vel2_reci * vel2_reci +
                           8.0 * z_h3 * z_h1 * z_h1 * vel2_reci * vel2_reci * vel2_reci);

          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              omg * step * domega_weight_ * violadOmegaPenaDR * gradViolaKRc;
          partialGradByTimes(i) += omg * domega_weight_ *
                                   (violadOmegaPenaDR * gradViolaKRt * step + violadOmegaPenaR / K);
          cost += omg * step * domega_weight_ * violadOmegaPenaR;
          cost_domega += omg * step * domega_weight_ * violadOmegaPenaR;
        }

        ////////////////////Continuous dense sampling

        violaDenseMinV = DenseMinV * DenseMinV - dsigma.squaredNorm();

        if (violaDenseMinV >= -0.01 && config_.denseResolution_ != 0) {

          gradDenseC.setZero();
          gradDenseT = 0.0;
          dense_cost = 0.0;

          activationSmoothed(violaDenseMinV, violaDenseMinVPena, violaDenseMinVPenaD);

          double special_step = step / config_.denseResolution_;
          double special_s1 = s1 - step;
          int disQuantity;
          if (j == 0) {
            disQuantity = config_.denseResolution_ / 2;
            Densealpha = 1.0 / K * j - 1.0 / K / config_.denseResolution_;
          }
          else if (j == K) {
            special_s1 = special_s1 - step / 2.0;
            disQuantity = config_.denseResolution_ / 2;
            Densealpha = 1.0 / K * j - 0.5 / K - 1.0 / K / config_.denseResolution_;
          }
          else {
            special_s1 = special_s1 - step / 2.0;
            disQuantity = config_.denseResolution_;
            Densealpha = 1.0 / K * j - 0.5 / K - 1.0 / K / config_.denseResolution_;
          }

          for (int l = 0; l <= disQuantity; l++) {

            s2 = special_s1 * special_s1;
            s3 = s2 * special_s1;
            s4 = s2 * s2;
            s5 = s4 * special_s1;
            s6 = s3 * s3;
            s7 = s4 * s3;

            // Densebeta0 << 1.0, special_s1, s2, s3, s4, s5, s6, s7;
            Densebeta1 << 0.0, 1.0, 2.0 * special_s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5,
                7.0 * s6;
            Densebeta2 << 0.0, 0.0, 2.0, 6.0 * special_s1, 12.0 * s2, 20.0 * s3, 30.0 * s4,
                42.0 * s5;
            Densebeta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * special_s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
            Densebeta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * special_s1, 360.0 * s2, 840 * s3;
            Densealpha += 1.0 / K / config_.denseResolution_;

            special_s1 += special_step;

            Densedsigma = c.block<8, 2>(0, 0).transpose() * Densebeta1;
            Denseddsigma = c.block<8, 2>(0, 0).transpose() * Densebeta2;
            Densedddsigma = c.block<8, 2>(0, 0).transpose() * Densebeta3;
            Denseddddsigma = c.block<8, 2>(0, 0).transpose() * Densebeta4;
            omg = (l == 0 || l == disQuantity) ? 0.5 : 1.0;

            z_h0 = Densedsigma.norm();
            z_h1 = Denseddsigma.transpose() * Densedsigma;
            z_h2 = Densedddsigma.transpose() * Densedsigma;
            z_h3 = Denseddsigma.transpose() * B_h * Densedsigma;
            vel2_reci = 1.0 / (z_h0 * z_h0);
            z_h41 = Densedddsigma.transpose() * B_h * Densedsigma;
            z_h5 = Densedddsigma.transpose() * B_h * Denseddsigma;
            z_h6 = Denseddddsigma.transpose() * B_h * Densedsigma;

            omega = z_h3 * vel2_reci;
            max_omega_ = 2.0 * zoom * (config_.max_vel_ - Densedsigma.norm()) / config_.wheel_base_;
            violaOmegaL = omega - max_omega_;
            violaOmegaR = -omega - max_omega_;

            domega = z_h41 * vel2_reci - 2.0 * z_h3 * z_h1 * vel2_reci * vel2_reci;
            violadOmegaL = domega - config_.max_domega_;
            violadOmegaR = -domega - config_.max_domega_;
            violavmin = v_min * v_min - Densedsigma.squaredNorm();

            if (violavmin > 0.0) {
              positiveSmoothedL1(violavmin, violavminPena, violavminPenaD);
              gradDenseViolaC = -2.0 * Densebeta1 * Densedsigma.transpose();
              gradDenseViolaT = -2.0 * Densealpha * z_h1;
              gradDenseC +=
                  omg * special_step * v_weight_ * 1e12 * violavminPenaD * gradDenseViolaC;
              gradDenseT += omg * v_weight_ * 1e12 *
                            (violavminPenaD * gradDenseViolaT * special_step +
                                violavminPena / K / config_.denseResolution_);
              dense_cost += omg * special_step * v_weight_ * 1e12 * violavminPena;
              cost_dense_v += omg * special_step * v_weight_ * 1e12 * violavminPena;
            }

            if (violaOmegaL > 0.0) {
              positiveSmoothedL1(violaOmegaL, violaOmegaPenaL, violaOmegaPenaDL);
              gradDenseViolaC =
                  Densebeta1 * (vel2_reci * Denseddsigma.transpose() * B_h -
                                   2 * vel2_reci * vel2_reci * z_h3 * Densedsigma.transpose() +
                                   2.0 * zoom * Densedsigma.transpose() * sqrt(vel2_reci) /
                                       config_.wheel_base_) +
                  Densebeta2 * vel2_reci * Densedsigma.transpose() * B_h.transpose();
              gradDenseViolaT =
                  Densealpha * (vel2_reci * (Densedddsigma.transpose() * B_h * Densedsigma -
                                                2 * vel2_reci * z_h3 * z_h1) +
                                   2.0 * zoom * z_h1 * sqrt(vel2_reci) / config_.wheel_base_);
              gradDenseC += omg * special_step * omega_weight_ * violaOmegaPenaDL * gradDenseViolaC;
              gradDenseT += omg * omega_weight_ *
                            (violaOmegaPenaDL * gradDenseViolaT * special_step +
                                violaOmegaPenaL / K / config_.denseResolution_);
              dense_cost += omg * special_step * omega_weight_ * violaOmegaPenaL;
              cost_dense_omega += omg * special_step * omega_weight_ * violaOmegaPenaL;
            }
            if (violaOmegaR > 0.0) {
              positiveSmoothedL1(violaOmegaR, violaOmegaPenaR, violaOmegaPenaDR);
              gradDenseViolaC =
                  -(Densebeta1 * (vel2_reci * Denseddsigma.transpose() * B_h -
                                     2 * vel2_reci * vel2_reci * z_h3 * Densedsigma.transpose() -
                                     2.0 * zoom * Densedsigma.transpose() * sqrt(vel2_reci) /
                                         config_.wheel_base_) +
                      Densebeta2 * vel2_reci * Densedsigma.transpose() * B_h.transpose());
              gradDenseViolaT =
                  -Densealpha * (vel2_reci * (Densedddsigma.transpose() * B_h * Densedsigma -
                                                 2 * vel2_reci * z_h3 * z_h1) -
                                    2.0 * zoom * z_h1 * sqrt(vel2_reci) / config_.wheel_base_);
              gradDenseC += omg * special_step * omega_weight_ * violaOmegaPenaDR * gradDenseViolaC;
              gradDenseT += omg * omega_weight_ *
                            (violaOmegaPenaDR * gradDenseViolaT * special_step +
                                violaOmegaPenaR / K / config_.denseResolution_);
              dense_cost += omg * special_step * omega_weight_ * violaOmegaPenaR;
              cost_dense_omega += omg * special_step * omega_weight_ * violaOmegaPenaR;
            }

            if (violadOmegaL > 0.0) {
              positiveSmoothedL1(violadOmegaL, violadOmegaPenaL, violadOmegaPenaDL);
              gradDenseViolaC =
                  Densebeta3 * (Densedsigma.transpose() * B_h.transpose() * vel2_reci) +
                  -Densebeta2 * 2.0 * vel2_reci * vel2_reci *
                      (z_h1 * Densedsigma.transpose() * B_h.transpose() +
                          z_h3 * Densedsigma.transpose()) +
                  Densebeta1 * ((Densedddsigma.transpose() * B_h * vel2_reci) -
                                   vel2_reci * vel2_reci * 2.0 *
                                       (z_h41 * Densedsigma.transpose() +
                                           z_h1 * Denseddsigma.transpose() * B_h +
                                           z_h3 * Denseddsigma.transpose()) +
                                   8.0 * z_h3 * z_h1 * Densedsigma.transpose() * vel2_reci *
                                       vel2_reci * vel2_reci);
              gradDenseViolaT =
                  Densealpha *
                  ((z_h5 + z_h6) * vel2_reci +
                      -(Denseddsigma.squaredNorm() * z_h3 + z_h2 * z_h3 + z_h41 * 2.0 * z_h1) *
                          2.0 * vel2_reci * vel2_reci +
                      8.0 * z_h3 * z_h1 * z_h1 * vel2_reci * vel2_reci * vel2_reci);
              gradDenseC +=
                  omg * special_step * domega_weight_ * violadOmegaPenaDL * gradDenseViolaC;
              gradDenseT += omg * domega_weight_ *
                            (violadOmegaPenaDL * gradDenseViolaT * special_step +
                                violadOmegaPenaL / K / config_.denseResolution_);
              dense_cost += omg * special_step * domega_weight_ * violadOmegaPenaL;
              cost_dense_domega += omg * special_step * domega_weight_ * violadOmegaPenaL;
            }

            if (violadOmegaR > 0.0) {
              positiveSmoothedL1(violadOmegaR, violadOmegaPenaR, violadOmegaPenaDR);
              gradDenseViolaC =
                  -Densebeta3 * (Densedsigma.transpose() * B_h.transpose() * vel2_reci) +
                  Densebeta2 * 2.0 * vel2_reci * vel2_reci *
                      (z_h1 * Densedsigma.transpose() * B_h.transpose() +
                          z_h3 * Densedsigma.transpose()) +
                  -Densebeta1 * ((Denseddsigma.transpose() * B_h * vel2_reci) -
                                    vel2_reci * vel2_reci * 2.0 *
                                        (z_h41 * Densedsigma.transpose() +
                                            z_h1 * Denseddsigma.transpose() * B_h +
                                            z_h3 * Denseddsigma.transpose()) +
                                    8.0 * z_h3 * z_h1 * Densedsigma.transpose() * vel2_reci *
                                        vel2_reci * vel2_reci);
              gradDenseViolaT =
                  -Densealpha *
                  ((z_h5 + z_h6) * vel2_reci +
                      -(Denseddsigma.squaredNorm() * z_h3 + z_h2 * z_h3 + z_h41 * 2.0 * z_h1) *
                          2.0 * vel2_reci * vel2_reci +
                      8.0 * z_h3 * z_h1 * z_h1 * vel2_reci * vel2_reci * vel2_reci);
              gradDenseC +=
                  omg * special_step * domega_weight_ * violadOmegaPenaDR * gradDenseViolaC;
              gradDenseT += omg * domega_weight_ *
                            (violadOmegaPenaDR * gradDenseViolaT * special_step +
                                violadOmegaPenaR / K / config_.denseResolution_);
              dense_cost += omg * special_step * domega_weight_ * violadOmegaPenaR;
              cost_dense_domega += omg * special_step * domega_weight_ * violadOmegaPenaR;
            }
          }

          cost_dense_all += violaDenseMinVPena * dense_cost;
          cost += violaDenseMinVPena * dense_cost;
          partialGradByCoeffs.block<8, 2>(i * 8, 0) +=
              -2.0 * beta1 * dsigma.transpose() * violaDenseMinVPenaD * dense_cost +
              violaDenseMinVPena * gradDenseC;
          partialGradByTimes(i) +=
              -2.0 * alpha * violaDenseMinVPenaD * dense_cost * ddsigma.transpose() * dsigma +
              violaDenseMinVPena * gradDenseT;
        }
      }
    }

    if (ifprint) {
      std::cout << "cost safe: " << cost_safe << std::endl;
      std::cout << "cost v: " << cost_v << std::endl;
      std::cout << "cost a: " << cost_a << std::endl;
      std::cout << "cost omega: " << cost_omega << std::endl;
      std::cout << "cost domega: " << cost_domega << std::endl;
      std::cout << "cost_dense_v: " << cost_dense_v << std::endl;
      std::cout << "cost_dense_omega: " << cost_dense_omega << std::endl;
      std::cout << "cost_dense_domega: " << cost_dense_domega << std::endl;
      std::cout << "cost_dense_all: " << cost_dense_all << std::endl;
      std::cout << "cost_mean_t: " << cost_mean_t << std::endl;
      std::cout << "cost: " << cost << std::endl;
    }
  }

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

  void visInnerPoints()
  {
    visualization_msgs::MarkerArray markerarraydelete;
    visualization_msgs::MarkerArray markerarray;
    visualization_msgs::Marker marker;

    marker.header.frame_id = "world";
    marker.ns = "initinnerPoint";
    marker.lifetime = ros::Duration();
    marker.type = visualization_msgs::Marker::CYLINDER;

    marker.action = visualization_msgs::Marker::DELETEALL;
    markerarraydelete.markers.push_back(marker);
    inner_init_point_pub_.publish(markerarraydelete);

    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.08;
    marker.scale.y = 0.08;
    marker.scale.z = 0.04;
    marker.color.a = 0.8;
    marker.color.r = 1.0 - 195.0 / 255;
    marker.color.g = 1.0 - 176.0 / 255;
    marker.color.b = 1.0 - 145.0 / 255;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.position.z = 0.15;

    for (uint i = 0; i < innerPointses.size(); i++) {
      for (uint j = 0; j < innerPointses[i].cols(); j++) {
        marker.scale.x = 0.08;
        marker.scale.y = 0.08;
        marker.scale.z = 0.04;
        marker.color.a = 0.8;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.header.stamp = ros::Time::now();
        marker.id = j * 10000 + i * 100;
        marker.pose.position.x = innerPointses[i].col(j).x();
        marker.pose.position.y = innerPointses[i].col(j).y();
        markerarray.markers.push_back(marker);

        marker.scale.z = 0.2;
        marker.color.a = 1.0;
        std::ostringstream str;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        double mani_angle =
            fabs(innerPointses[i].col(j).z()) > 1e-4 ? innerPointses[i].col(j).z() : 0.0;
        str << mani_angle;
        marker.text = str.str();
        marker.id = j * 10000 + i * 100 + 1;
        markerarray.markers.push_back(marker);
      }
    }
    inner_init_point_pub_.publish(markerarray);
  }

  void visFinalInnerPoints()
  {
    visualization_msgs::MarkerArray markerarraydelete;
    visualization_msgs::MarkerArray markerarray;
    visualization_msgs::Marker marker;

    marker.header.frame_id = "world";
    marker.ns = "innerPoint";
    marker.lifetime = ros::Duration();
    marker.type = visualization_msgs::Marker::CYLINDER;

    marker.action = visualization_msgs::Marker::DELETEALL;
    marker.scale.x = 0.12;
    marker.scale.y = 0.12;
    marker.scale.z = 0.04;
    marker.color.a = 0.8;
    marker.color.r = 95.0 / 255;
    marker.color.g = 76.0 / 255;
    marker.color.b = 45.0 / 255;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.position.x = 0.15;
    marker.pose.position.y = 0.15;
    marker.pose.position.z = 0.15;
    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    markerarray.markers.push_back(marker);
    inner_point_pub_.publish(markerarray);
    markerarray.markers.clear();
    marker.action = visualization_msgs::Marker::ADD;

    for (uint i = 0; i < finalInnerpointses.size(); i++) {
      for (uint j = 0; j < finalInnerpointses[i].cols(); j++) {
        marker.header.stamp = ros::Time::now();
        marker.id = j * 100 + i * 1;
        marker.pose.position.x = finalInnerpointses[i].col(j).x();
        marker.pose.position.y = finalInnerpointses[i].col(j).y();
        markerarray.markers.push_back(marker);
      }
    }
    inner_point_pub_.publish(markerarray);
  }

  void mincoInitTrajPub(
      const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls)
  {
    if (final_trajes.size() != final_singuls.size())
      ROS_ERROR("[mincoInitTrajPub] Input size ERROR !!!!");

    int traj_size = final_trajes.size();
    double total_time;
    Eigen::VectorXd traj_time;
    traj_time.resize(traj_size);
    for (int i = 0; i < traj_size; i++) {
      traj_time[i] = final_trajes[i].getTotalDuration();
    }
    total_time = traj_time.sum();

    int index = 0;
    Eigen::VectorXd currPos, currVel;

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();

    for (double time = 1e-5; time < total_time; time += 1e-4) {
      double index_time = 0;
      for (index = 0; index < traj_size; index++) {
        if (time > index_time && time < index_time + traj_time[index])
          break;
        index_time += traj_time[index];
      }
      currPos = final_trajes[index].getPos(time - index_time);
      currVel = final_trajes[index].getVel(time - index_time);
      double yaw = atan2(currVel.y(), currVel.x());
      int singuls = final_singuls[index];
      if (singuls < 0)
        yaw += M_PI;

      Eigen::Matrix2d R;
      R << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);

      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = ros::Time::now();
      pose.pose.position.x = (currPos).x();
      pose.pose.position.y = (currPos).y();
      pose.pose.position.z = 0.15;

      pose.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
      path.poses.push_back(pose);
    }

    minco_init_path_pub_.publish(path);
  }

  void mincoInitPathPubwithAlpha(const std::vector<Trajectory<7, 3>>& final_trajes,
      const Eigen::VectorXi& final_singuls, const int& k)
  {
    if (final_trajes.size() != final_singuls.size())
      ROS_ERROR("[mincoInitTrajPub] Input size ERROR !!!!");

    int traj_size = final_trajes.size();
    double total_time;
    Eigen::VectorXd traj_time;
    traj_time.resize(traj_size);
    for (int i = 0; i < traj_size; i++) {
      traj_time[i] = final_trajes[i].getTotalDuration();
    }
    total_time = traj_time.sum();

    int index = 0;
    Eigen::VectorXd currPos, currVel;

    visualization_msgs::MarkerArray markerarraydelete;
    visualization_msgs::MarkerArray markerarray;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = "minco_opt_path_alpha_pub_";
    marker.lifetime = ros::Duration();
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::DELETEALL;
    marker.scale.x = 0.12;
    marker.scale.y = 0.12;
    marker.scale.z = 0.02;
    marker.color.a = 1;
    marker.color.r = 1;
    marker.color.g = 0;
    marker.color.b = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    markerarraydelete.markers.push_back(marker);
    marker.action = visualization_msgs::Marker::ADD;

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();

    double traj_lenth = 0.0;
    double acc_cost = 0.0;
    double alpha_cost = 0.0;
    double mani_alpha_cost = 0.0;

    for (double time = 1e-5; time < total_time; time += 1e-4) {
      double index_time = 0;
      for (index = 0; index < traj_size; index++) {
        if (time > index_time && time < index_time + traj_time[index])
          break;
        index_time += traj_time[index];
      }
      currPos = final_trajes[index].getPos(time - index_time);
      currVel = final_trajes[index].getVel(time - index_time);

      double yaw = atan2(currVel.y(), currVel.x());
      int singuls = final_singuls[index];
      if (singuls < 0)
        yaw += M_PI;

      Eigen::Matrix2d R;
      R << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);

      Eigen::Matrix2d B_h;
      B_h << 0, -1.0, 1.0, 0;
      Eigen::VectorXd currAcc = final_trajes[index].getAcc(time - index_time).head(2);
      Eigen::VectorXd currJer = final_trajes[index].getJer(time - index_time).head(2);
      Eigen::VectorXd currSna = final_trajes[index].getSna(time - index_time).head(2);
      double normVel = currVel.head(2).norm();
      double help1 = 1 / (normVel * normVel);
      double help1_e = 1 / (normVel * normVel + 1e-4);
      double omega = help1 * currAcc.transpose() * B_h * currVel.head(2);
      double z_h0 = currVel.head(2).norm();
      double z_h1 = currAcc.transpose() * currVel.head(2);
      double z_h2 = currJer.transpose() * currVel.head(2);
      double z_h3 = currAcc.transpose() * B_h * currVel.head(2);
      double z_h4 = currJer.transpose() * B_h * currVel.head(2);
      double z_h5 = currJer.transpose() * B_h * currAcc;
      double z_h6 = currSna.transpose() * B_h * currVel.head(2);

      double help2 = currJer.transpose() * B_h * currVel.head(2);
      double domega = help2 * help1 - 2.0 * help1 * help1 * currAcc.transpose() * B_h *
                                          currVel.head(2) * currAcc.transpose() * currVel.head(2);
      double domega2 = help2 * help1 - 2.0 * help1 * help1 * z_h3 * z_h1;
      double domega_e = help2 * help1_e - 2.0 * help1_e * help1_e * currAcc.transpose() * B_h *
                                              currVel.head(2) * currAcc.transpose() *
                                              currVel.head(2);
      double ddomega =
          (z_h5 + z_h6) * help1 +
          -(currAcc.squaredNorm() * z_h3 + z_h2 * z_h3 + z_h4 * 2.0 * z_h1) * 2.0 * help1 * help1 +
          8.0 * z_h3 * z_h1 * z_h1 * help1 * help1 * help1;
      double cur = z_h3 * (help1 * sqrt(help1));
      double Accl = sqrt(z_h1 * z_h1 * help1);
      double maxomega = 2.0 * 0.2 * (2 - z_h0) / 0.3;

      double currmaniAcc = final_trajes[index].getAcc(time - index_time)[2];

      traj_lenth += normVel * 1e-4;
      acc_cost += abs(Accl) * 1e-4;
      alpha_cost += abs(domega) * 1e-4;
      mani_alpha_cost += abs(currmaniAcc) * 1e-4;

      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = ros::Time::now();
      pose.pose.position.x = 5.0 + time;
      pose.pose.position.y = domega / 2.0;
      pose.pose.position.z = 0;

      pose.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);
      path.poses.push_back(pose);

      double index_piece_time = 0;
      Eigen::VectorXd trajDurations = final_trajes[index].getDurations();
      for (double pieceindex = 0; pieceindex < final_trajes[index].getPieceNum(); pieceindex++) {
        if (time > index_piece_time && time < index_piece_time + trajDurations[pieceindex]) {
          if (std::fmod(time - index_piece_time, trajDurations[pieceindex] / 8.0) < 1.1e-4) {
            marker.header.stamp = ros::Time::now();
            marker.id = (int)(time * 1000);
            marker.pose.position.x = 5.0 + time;
            marker.pose.position.y = domega / 2.0;
            marker.pose.position.z = 0;
            markerarray.markers.push_back(marker);
          }
          break;
        }
        index_piece_time += trajDurations[pieceindex];
      }
    }
    if (k % 1 == 0) {
      minco_opt_path_alpha_pub_.publish(markerarraydelete);
      minco_opt_path_alpha_pub_.publish(markerarray);
    }
    minco_init_path_alpha_pub_.publish(path);
  }

  void mincoPathPub(
      const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls)
  {
    if (final_trajes.size() != final_singuls.size())
      ROS_ERROR("[mincoCarPathPub] Input size ERROR !!!!");

    int traj_size = final_trajes.size();
    double total_time;
    Eigen::VectorXd traj_time;
    traj_time.resize(traj_size);
    for (int i = 0; i < traj_size; i++) traj_time[i] = final_trajes[i].getTotalDuration();
    total_time = traj_time.sum();

    int index = 0;
    Eigen::VectorXd currPos, currVel;

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();

    for (double time = 1e-5; time < total_time; time += 4e-4) {
      double index_time = 0;
      for (index = 0; index < traj_size; index++) {
        if (time > index_time && time < index_time + traj_time[index])
          break;
        index_time += traj_time[index];
      }
      currPos = final_trajes[index].getPos(time - index_time);
      currVel = final_trajes[index].getVel(time - index_time);

      double yaw = atan2(currVel.y(), currVel.x());
      int singuls = final_singuls[index];
      if (singuls < 0) {
        yaw += M_PI;
      }
      Eigen::Matrix2d R;
      R << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);

      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = ros::Time::now();
      pose.pose.position.x = (currPos).x();
      pose.pose.position.y = (currPos).y();
      pose.pose.position.z = 0.15;

      pose.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
      path.poses.push_back(pose);
    }

    minco_path_pub_.publish(path);
  }
};
}  // namespace apexnav_planner
#endif