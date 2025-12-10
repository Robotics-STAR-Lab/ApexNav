#include <path_searching/kino_astar.h>

#define uint unsigned int

namespace apexnav_planner {
void KinoAstar::init()
{
  std::string node_name = ros::this_node::getName();
  nh_.param<double>(ros::this_node::getName() + "/lambda_heu", lambda_heu_, 5.0);
  nh_.param<double>(ros::this_node::getName() + "/max_seach_time", max_search_time_, 0.1);
  nh_.param<double>(ros::this_node::getName() + "/step_arc", step_arc_, 0.9);
  nh_.param<double>(ros::this_node::getName() + "/checkl", checkl_, 0.2);
  nh_.param<int>(ros::this_node::getName() + "/check_num", check_num_, 5);
  nh_.param<double>(ros::this_node::getName() + "/oneshot_range", oneshot_range_, 5);
  nh_.param<double>(ros::this_node::getName() + "/sampletime", sampletime_, 0.1);

  inv_yaw_resolution_ = 3.15;
  inv_yaw_resolution_ = 1.0 / yaw_resolution_;
  grid_interval_ = map_->getResolution();
  allocate_num_ = 1000000;

  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) path_node_pool_[i] = new PathNode;

  use_node_num_ = 0;
  iter_num_ = 0;
  yaw_origin_ = -M_PI;

  nh_.param<double>(ros::this_node::getName() + "/max_vel", max_vel_, 5);
  nh_.param<double>(ros::this_node::getName() + "/max_acc", max_acc_, 5);

  max_acc_ = max_acc_ * 0.6;
  max_vel_ = max_vel_ * 0.6;

  nh_.param<double>(ros::this_node::getName() + "/max_cur", max_cur_, 0.5);
  nh_.param<double>(ros::this_node::getName() + "/non_siguav", non_siguav_, 0.01);
  nh_.param<double>(ros::this_node::getName() + "/collision_interval", collision_interval_, 0.1);
  nh_.param<double>(ros::this_node::getName() + "/wheel_base", wheel_base_, 0.8);
  max_steer_ = std::atan(wheel_base_ * max_cur_);
  nh_.param<double>(ros::this_node::getName() + "/length", length_, 1);
  nh_.param<double>(ros::this_node::getName() + "/width", width_, 1);
  nh_.param<double>(ros::this_node::getName() + "/height", height_, 1);

  // SE(2) motion model (x,y,yaw), suitable for nonholonomic constraints with limited turning radius
  shotptr_s.push_back(std::make_shared<ompl::base::DubinsStateSpace>(0.2));
  shotptr_s.push_back(std::make_shared<ompl::base::DubinsStateSpace>(0.1));
  shotptr_s.push_back(std::make_shared<ompl::base::DubinsStateSpace>(0.05));

  shotptrindex = -1;
  shotptrsize = shotptr_s.size();

  expandNodes_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("kinoastar/expanded_nodes", 10);
  kinoastarFlatPathPub_ = nh_.advertise<visualization_msgs::MarkerArray>("/kinoastar/FlatPath", 10);
  kinoastarFlatTrajPub_ = nh_.advertise<nav_msgs::Path>("/kinoastar/FlatTraj", 10);
}

void KinoAstar::reset()
{
  expanded_nodes_.clear();
  path_nodes_.clear();
  flat_trajs_.clear();
  sample_traj_.clear();
  std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
  open_set_.swap(empty_queue);

  for (int i = 0; i < use_node_num_; i++) {
    PathNodePtr node = path_node_pool_[i];
    node->parent = NULL;
    node->node_state = NOT_EXPAND;
  }

  use_node_num_ = 0;
  iter_num_ = 0;
  is_shot_succ_ = false;
  has_path_ = false;
}

// Obtain path_nodes_ and expanded_nodes_
// state format: x y yaw mani v
int KinoAstar::search(
    const Eigen::VectorXd& end_state, Eigen::VectorXd& start_state, Eigen::Vector3d& init_ctrl)
{
  bool isocc = false;
  bool initsearch = false;  // 'initsearch' indicates whether to consider initial state's velocity;
                            // false means start can only search forward
  double start_time = ros::Time::now().toSec();

  Eigen::Vector2d start_pos2d = start_state.head(2);
  Eigen::Vector2d end_pos2d = end_state.head(2);
  Eigen::Vector2i start_idx;
  map_->posToIndex(start_pos2d, start_idx);

  // Check whether start/end states are in collision
  // isocc = isCollisionPosYaw(start_state.head(2), start_state[2]);
  // if (isocc) {
  //   ROS_ERROR("KinoAstar: start is not free!");
  //   return NO_PATH;
  // }
  isocc = isCollisionPosYaw(end_state.head(2), end_state[2]);
  if (isocc) {
    ROS_WARN("KinoAstar: end is not free!");
    return NO_PATH;
  }
  if (!map_->isInMap(end_pos2d)) {
    ROS_WARN("KinoAstar: end is out of map!");
    return NO_PATH;
  }

  start_state_ = start_state;
  start_ctrl_ = init_ctrl;
  end_state_ = end_state;

  // Initialize path_node_pool_ with the start node
  PathNodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->state = start_state.head(4);
  cur_node->index = start_idx;
  cur_node->yaw_idx = yawToIndex(start_state[2]);
  cur_node->g_score = 0.0;
  cur_node->input = Eigen::Vector3d(0.0, 0.0, 0.0);
  cur_node->singul = getSingularity(start_state[4]);
  cur_node->f_score = lambda_heu_ * getHeu(cur_node->state, end_state);
  cur_node->node_state = IN_OPEN_SET;

  open_set_.push(cur_node);
  use_node_num_ += 1;

  expanded_nodes_.insert(cur_node->index, yawToIndex(start_state[2]), 0, cur_node);
  PathNodePtr terminate_node = NULL;
  // If initial velocity is negative, do not consider direction; otherwise consider direction
  if (cur_node->singul == 0)
    initsearch = true;

  while (!open_set_.empty()) {
    cur_node = open_set_.top();

    if ((cur_node->state.head(2) - end_state_.head(2)).norm() < oneshot_range_ && initsearch)
      isShotSuccess(cur_node->state, end_state_.head(4));

    if (is_shot_succ_) {
      terminate_node = cur_node;
      start_state = end_state;
      init_ctrl = Eigen::Vector3d(0.0, 0.0, 0.0);
      retrievePath(terminate_node);
      has_path_ = true;
      ROS_WARN("one shot! iter num: %d", iter_num_);
      return REACH_END;
    }
    // If oneshot fails and runtime exceeds the limit, treat this node as the endpoint
    double runTime = ros::Time::now().toSec() - start_time;
    if (runTime > max_search_time_) {
      terminate_node = cur_node;
      retrievePath(terminate_node);
      has_path_ = true;
      start_state.head(4) = terminate_node->state;
      start_state[4] = terminate_node->singul;
      init_ctrl = terminate_node->input;
      if (terminate_node->parent == NULL) {
        std::cout << "[34mKino Astar]: terminate_node->parent == NULL" << std::endl;
        printf("\033[Kino Astar]: NO_PATH \n\033[0m");
        return NO_PATH;
      }
      else {
        ROS_WARN("KinoSearch: Reach the max seach time");
        return REACH_END;
      }
    }

    /* ---------- pop node and add to close set ---------- */
    open_set_.pop();
    cur_node->node_state = IN_CLOSE_SET;
    iter_num_ += 1;
    // Obtain control inputs by sampling
    Eigen::Vector4d cur_state = cur_node->state;
    Eigen::Vector4d pro_state;
    Eigen::Vector3d ctrl_input;
    std::vector<Eigen::Vector3d> inputs;
    double res = 0.2;

    // Initial search uses different sampling than subsequent searches
    // (initial state may have forward velocity). Obtain inputs.
    // input[0] is steering angle; input[1] is displacement
    if (!initsearch) {
      if (start_state_[4] > 0) {
        for (double arc = grid_interval_; arc <= 2 * grid_interval_ + 1e-3; arc += grid_interval_) {
          for (double steer = -max_steer_; steer <= max_steer_ + 1e-3;
              steer += res * max_steer_ * 1.0) {
            ctrl_input << steer, arc, 0.0;
            inputs.push_back(ctrl_input);
          }
        }
      }
      initsearch = true;
    }
    else {
      for (double arc = 0; arc <= step_arc_ + 1e-3; arc += 0.5 * step_arc_) {
        for (double steer = -max_steer_; steer <= max_steer_ + 1e-3;
            steer += res * max_steer_ * 1.0) {
          ctrl_input << steer, arc, 0.0;
          inputs.push_back(ctrl_input);
        }
      }
    }

    for (auto& input : inputs) {
      int singul;
      if (abs(input[1]) < 1e-3)
        singul = cur_node->singul;
      else
        singul = input[1] > 0 ? 1 : -1;

      // Compute the successor state `pro_state` after applying this sampled input
      stateTransit(cur_state, input, pro_state);
      /* inside map range */
      Eigen::Vector2d pro_pos2d = pro_state.head(2);
      if (!map_->isInMap(pro_pos2d)) {
        std::cout << "[Kino Astar]: out of map range" << pro_state.transpose() << std::endl;
        continue;
      }
      // Check whether pro_state is in the closed set; if so, skip
      // Determine the grid cell for pro_state
      Eigen::Vector2i pro_id;
      map_->posToIndex(pro_pos2d, pro_id);
      int pro_yaw_id = yawToIndex(pro_state[2]);
      PathNodePtr pro_node;
      pro_node = expanded_nodes_.find(pro_id, pro_yaw_id, 0);

      if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET)
        continue;

      /* collision checking */
      Eigen::Vector4d xt;

      // Divide trajectory length into 'check_num_' segments for collision checking
      for (int k = 1; k <= check_num_; ++k) {
        // Sample along trajectory length; steering angle remains unchanged
        double tmparc = input[1] * double(k) / double(check_num_);
        double tmpmani = input[2] * double(k) / double(check_num_);
        Eigen::Vector3d tmpctrl;
        tmpctrl << input[0], tmparc, tmpmani;
        // Compute the sampled point
        stateTransit(cur_state, tmpctrl, xt);
        // Check for collisions around the vehicle at sampled intervals
        if (tmparc < 1e-4)
          isocc = isCollisionPosYaw(xt.head(2), xt[2]);
        else
          isocc = isCollisionPosYaw(xt.head(2), xt[2]);
        if (isocc)
          break;
      }
      if (isocc)
        continue;
      nodeVis(pro_state.head(3));

      /* ---------- compute cost ---------- */
      double tmp_g_score = 0.0;
      double tmp_f_score = 0.0;

      tmp_g_score += std::fabs(input[1]);
      tmp_g_score += cur_node->g_score;
      tmp_f_score = tmp_g_score + lambda_heu_ * getHeu(pro_state, end_state);

      // New node not expanded before: add it
      if (pro_node == NULL) {
        pro_node = path_node_pool_[use_node_num_];
        pro_node->index = pro_id;
        pro_node->state = pro_state;
        pro_node->yaw_idx = pro_yaw_id;
        pro_node->f_score = tmp_f_score;
        pro_node->g_score = tmp_g_score;
        pro_node->input = input;
        pro_node->parent = cur_node;
        pro_node->node_state = IN_OPEN_SET;
        pro_node->singul = singul;
        open_set_.push(pro_node);

        expanded_nodes_.insert(pro_id, pro_yaw_id, 0, pro_node);
        use_node_num_ += 1;
        if (use_node_num_ == allocate_num_) {
          std::cout << "run out of memory." << std::endl;
          return NO_PATH;
        }
      }
      else if (pro_node->node_state == IN_OPEN_SET) {
        if (tmp_g_score < pro_node->g_score) {
          pro_node->index = pro_id;
          pro_node->state = pro_state;
          pro_node->yaw_idx = pro_yaw_id;
          pro_node->f_score = tmp_f_score;
          pro_node->g_score = tmp_g_score;
          pro_node->input = input;
          pro_node->parent = cur_node;
          pro_node->singul = singul;
        }
      }
      else {
        std::cout << "error type in searching: " << pro_node->node_state << std::endl;
      }
    }
  }
  std::cout << "open set empty, no path." << std::endl;
  return NO_PATH;
}

void KinoAstar::getKinoNode()
{
  if (!has_path_)
    return;
  ros::Time current = ros::Time::now();
  getSampleTraj();
  getTrajsWithTime();
}

// Obtain sample_traj_
void KinoAstar::getSampleTraj()
{
  double mani_angle;
  // Sample points
  std::vector<Eigen::Vector4d> roughSampleList;

  // Endpoint
  PathNodePtr node = path_nodes_.back();

  while (node->parent != NULL) {
    for (int k = check_num_; k > 0; k--) {
      double tmparc = node->input[1] * double(k) / double(check_num_);
      double tmpmani = node->input[2] * double(k) / double(check_num_);
      Eigen::Vector3d tmpctrl;
      tmpctrl << node->input[0], tmparc, tmpmani;
      Eigen::Vector4d state4d;
      stateTransit(node->parent->state, tmpctrl, state4d);
      state4d[2] = normalizeAngle(state4d[2]);
      roughSampleList.push_back(state4d);
    }
    node = node->parent;
  }

  // Also include the start point
  start_state_[2] = normalizeAngle(start_state_[2]);
  start_state_[3] = normalizeAngle(start_state_[3]);
  roughSampleList.emplace_back(start_state_[0], start_state_[1], start_state_[2], start_state_[3]);
  reverse(roughSampleList.begin(), roughSampleList.end());  // From start to goal

  // Add oneshot check points
  if (is_shot_succ_) {
    ompl::base::StateSpacePtr shotptr_ = shotptr_s[shotptrindex];
    ompl::base::ScopedState<> from(shotptr_), to(shotptr_), s(shotptr_);
    Eigen::Vector4d state1, state2;
    state1 = roughSampleList.back();                                       // front-end endpoint
    state2 << end_state_[0], end_state_[1], end_state_[2], end_state_[3];  // actual endpoint
    from[0] = state1[0];
    from[1] = state1[1];
    from[2] = state1[2];
    to[0] = state2[0];
    to[1] = state2[1];
    to[2] = state2[2];
    double shotLength = shotptr_->distance(from(), to());  // oneshot distance

    std::vector<double> reals;

    mani_angle = state1[3];
    for (double l = checkl_; l < shotLength; l += checkl_) {
      shotptr_->interpolate(from(), to(), l / shotLength,
          s());  // interpolate intermediate point at fraction l/shotLength
      reals = s.reals();
      roughSampleList.emplace_back(reals[0], reals[1], normalizeAngle(reals[2]), mani_angle);
    }
    end_state_[2] = normalizeAngle(end_state_[2]);
    roughSampleList.emplace_back(end_state_[0], end_state_[1], end_state_[2], end_state_[3]);
  }

  // Truncate the initial trajectory to limit its length
  double tmp_len = 0;  // cumulative distance from the start to the current point
  uint truncate_idx = 0;
  for (truncate_idx = 0; truncate_idx < roughSampleList.size() - 1; truncate_idx++) {
    tmp_len += evaluateDistance(
        roughSampleList[truncate_idx].head(2), roughSampleList[truncate_idx + 1].head(2));
  }
  roughSampleList.assign(roughSampleList.begin(), roughSampleList.begin() + truncate_idx + 1);
  sample_traj_ = roughSampleList;
}

// Obtain flat_trajs_
void KinoAstar::getTrajsWithTime()
{
  double start_vel = fabs(start_state_[4]);
  double end_vel = fabs(end_state_[4]);
  std::vector<Eigen::Vector4d> traj_pts;  // store uniformly-sampled points: x,y,t,mani_angle
  std::vector<double> thetas;             // store uniformly-sampled yaw values
  shot_lengthList.clear();
  shot_timeList.clear();
  shotindex.clear();
  shot_SList.clear();

  double tmp_length = 0;
  // Determine forward direction by comparing the vector between two points with the local heading
  int last_singul =
      (sample_traj_[1] - sample_traj_[0])
                  .head(2)
                  .dot(Eigen::Vector2d(cos(sample_traj_[0][2]), sin(sample_traj_[0][2]))) >= 0 ?
          1 :
          -1;
  shotindex.push_back(0);

  for (uint i = 0; i < sample_traj_.size() - 1; i++) {
    Eigen::Vector4d state1 = sample_traj_[i];
    Eigen::Vector4d state2 = sample_traj_[i + 1];
    int cur_singul;
    if ((state2 - state1).head(2).norm() < 1e-4)
      cur_singul = last_singul;
    else
      cur_singul =
          (state2 - state1).head(2).dot(Eigen::Vector2d(cos(state1[2]), sin(state1[2]))) >= 0 ?
              1 :
              -1;  // current forward direction
    // If forward direction unchanged, accumulate length; if it changes, record this turning point
    if (cur_singul * last_singul >= 0)
      tmp_length += evaluateDistance(state1.head(2), state2.head(2));
    else {
      ROS_ERROR("break break break");
      // store turning point index
      shotindex.push_back(i);
      // store direction before the turn
      shot_SList.push_back(last_singul);
      // store total distance before the turn
      shot_lengthList.push_back(tmp_length);
      if (tmp_length == inf)
        return;
      // store time before the turn
      shot_timeList.push_back(evaluateDuration(tmp_length, non_siguav_, non_siguav_));
      tmp_length =
          evaluateDistance(state1.head(2), state2.head(2));  // restart distance accumulation
    }
    last_singul = cur_singul;
  }
  // append the final segment to lists
  shot_SList.push_back(last_singul);
  shot_lengthList.push_back(tmp_length);
  shot_timeList.push_back(evaluateDuration(tmp_length, non_siguav_, non_siguav_));
  shotindex.push_back(sample_traj_.size() - 1);

  // adjust times considering start and end velocities
  if (shot_timeList.size() >= 2) {
    shot_timeList[0] = evaluateDuration(shot_lengthList[0], start_vel, non_siguav_);
    shot_timeList[shot_timeList.size() - 1] =
        evaluateDuration(shot_lengthList.back(), non_siguav_, end_vel);
  }
  else {
    shot_timeList[0] = evaluateDuration(shot_lengthList[0], start_vel, end_vel);
  }

  for (uint i = 0; i < shot_lengthList.size(); i++) {
    double initv = non_siguav_, finv = non_siguav_;
    Eigen::Vector3d init_ctrl_input, final_ctrl_input;
    init_ctrl_input << 0, 0, 0;
    final_ctrl_input << 0, 0, 0;
    // read initial and final velocities
    if (i == 0) {
      initv = start_vel;
      init_ctrl_input = start_ctrl_;
    }
    if (i == shot_lengthList.size() - 1)
      finv = end_vel;

    double locallength = shot_lengthList[i];  // i-th segment length
    int sig = shot_SList[i];                  // i-th segment direction
    // extract all points belonging to the i-th segment
    std::vector<Eigen::Vector4d> localTraj;
    localTraj.assign(
        sample_traj_.begin() + shotindex[i], sample_traj_.begin() + shotindex[i + 1] + 1);
    traj_pts.clear();
    thetas.clear();
    double samplet;
    double tmparc = 0;
    int index = 0;
    // // push the start point (commented out)
    // traj_pts.emplace_back(start_state_[0],start_state_[1],0);
    // thetas.push_back(start_state_[2]);
    // if this segment is too short, ensure at least two samples
    if (shot_timeList[i] <= sampletime_) {
      sampletime_ = shot_timeList[i] / 2.0;
    }
    // sample the trajectory uniformly by sample time
    for (samplet = sampletime_; samplet < shot_timeList[i]; samplet += sampletime_) {
      // sample at the current sample time
      double arc = evaluateLength(samplet, locallength, shot_timeList[i], initv, finv);
      // find the two points surrounding this arc distance and interpolate
      for (uint k = index; k < localTraj.size() - 1; k++) {
        // find the point where cumulative distance >= arc
        tmparc += evaluateDistance(localTraj[k].head(2), localTraj[k + 1].head(2));
        if (tmparc >= arc) {
          index = k;
          double l1 = tmparc - arc;
          double l = evaluateDistance(localTraj[k].head(2), localTraj[k + 1].head(2));
          double l2 = l - l1;
          double px = (l1 / l * localTraj[k] + l2 / l * localTraj[k + 1])[0];
          double py = (l1 / l * localTraj[k] + l2 / l * localTraj[k + 1])[1];
          double yaw = (l1 / l * localTraj[k] + l2 / l * localTraj[k + 1])[2];
          bool occ = isCollisionPosYaw(Eigen::Vector2d(px, py), yaw);
          if (occ)
            ROS_ERROR("isCollisionPosYaw occ!!!!!!!!  position: %f %f", px, py);

          if (fabs(localTraj[k + 1][2] - localTraj[k][2]) >= M_PI) {
            if (localTraj[k + 1][2] <= 0)
              yaw = l1 / l * localTraj[k][2] + l2 / l * (localTraj[k + 1][2] + 2 * M_PI);
            else if (localTraj[k][2] <= 0)
              yaw = l1 / l * (localTraj[k][2] + 2 * M_PI) + l2 / l * localTraj[k + 1][2];
          }

          traj_pts.emplace_back(px, py, sampletime_, 0.0);
          thetas.push_back(yaw);
          tmparc -= evaluateDistance(localTraj[k].head(2), localTraj[k + 1].head(2));
          break;
        }
      }
    }
    traj_pts.emplace_back(localTraj.back()[0], localTraj.back()[1],
        shot_timeList[i] - (samplet - sampletime_), localTraj.back()[3]);
    thetas.push_back(localTraj.back()[2]);

    Eigen::MatrixXd startS;
    Eigen::MatrixXd endS;
    getFlatState(localTraj.front()[0], localTraj.front()[1], localTraj.front()[2], initv,
        localTraj.front()[3], init_ctrl_input, sig, startS);
    getFlatState(localTraj.back()[0], localTraj.back()[1], localTraj.back()[2], finv,
        localTraj.back()[3], final_ctrl_input, sig, endS);

    FlatTrajData flat_traj;
    flat_traj.traj_pts = traj_pts;
    flat_traj.thetas = thetas;
    flat_traj.start_state = startS;
    flat_traj.final_state = endS;
    flat_traj.singul = sig;
    flat_trajs_.push_back(flat_traj);
  }
  // compute total time
  totalTrajTime = 0;
  for (uint i = 0; i < shot_timeList.size(); i++) totalTrajTime += shot_timeList[i];
}

double KinoAstar::evaluateDistance(const Eigen::Vector2d& state1, const Eigen::Vector2d& state2)
{
  Eigen::Vector2d diff = state2 - state1;
  double euclidean_dist = diff.norm();
  return euclidean_dist;
}

// compute duration using a trapezoidal velocity profile
double KinoAstar::evaluateDuration(const double& length, const double& startV, const double& endV)
{
  double critical_len;
  if (startV > max_vel_ || endV > max_vel_) {
    ROS_ERROR("kinoAstar:evaluateDuration:start or end vel is larger that the limit!");
  }
  double startv2 = pow(startV, 2);
  double endv2 = pow(endV, 2);
  double maxv2 = pow(max_vel_, 2);
  critical_len = (maxv2 - startv2) / (2 * max_acc_) + (maxv2 - endv2) / (2 * max_acc_);
  if (length >= critical_len) {
    return (max_vel_ - startV) / max_acc_ + (max_vel_ - endV) / max_acc_ +
           (length - critical_len) / max_vel_;
  }
  else {
    double tmpv = sqrt(0.5 * (startv2 + endv2 + 2 * max_acc_ * length));
    return (tmpv - startV) / max_acc_ + (tmpv - endV) / max_acc_;
  }
}

int KinoAstar::yawToIndex(const double& yaw)
{
  double nor_yaw = normalizeAngle(yaw);
  int idx = floor((nor_yaw - yaw_origin_) * inv_yaw_resolution_);
  return idx;
}

// Determine initial direction from initial velocity: >0 forward, <0 backward, ~0 keep previous
inline int KinoAstar::getSingularity(const double& vel)
{
  int singul = 0;
  if (fabs(vel) > non_siguav_) {
    if (vel >= 0.0) {
      singul = 1;
    }
    else {
      singul = -1;
    }
  }

  return singul;
}

bool KinoAstar::isCollisionPosYaw(const Eigen::Vector2d& pos, const double& yaw)
{
  Eigen::Vector2d point;
  uint8_t check_occ;
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  Eigen::Matrix2d egoR;
  egoR << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  // Assume the odometry pose is at the vehicle center
  Eigen::Vector2d center(pos + egoR * Eigen::Vector2d(0.0, 0));

  // vehicle body
  Eigen::Vector2d corner1 = center + egoR * Eigen::Vector2d(length_ / 2, width_ / 2);
  Eigen::Vector2d corner2 = center + egoR * Eigen::Vector2d(length_ / 2, -width_ / 2);
  Eigen::Vector2d corner3 = center + egoR * Eigen::Vector2d(-length_ / 2, -width_ / 2);
  Eigen::Vector2d corner4 = center + egoR * Eigen::Vector2d(-length_ / 2, width_ / 2);

  double norm12 = (corner2 - corner1).norm();
  double norm23 = (corner3 - corner2).norm();
  double norm34 = (corner4 - corner3).norm();
  double norm41 = (corner1 - corner4).norm();

  double height = 0.0;
  check_occ = checkCollision(corner1.x(), corner1.y(), height);
  if (check_occ == 1)
    return true;
  check_occ = checkCollision(corner2.x(), corner2.y(), height);
  if (check_occ == 1)
    return true;
  check_occ = checkCollision(corner3.x(), corner3.y(), height);
  if (check_occ == 1)
    return true;
  check_occ = checkCollision(corner4.x(), corner4.y(), height);
  if (check_occ == 1)
    return true;

  for (double dl = collision_interval_; dl < norm12; dl += collision_interval_) {
    point = dl / norm12 * (corner2 - corner1) + corner1;
    check_occ = checkCollision(point.x(), point.y(), height);
    if (check_occ == 1)
      return true;
  }
  for (double dl = collision_interval_; dl < norm23; dl += collision_interval_) {
    point = dl / norm23 * (corner3 - corner2) + corner2;
    check_occ = checkCollision(point.x(), point.y(), height);
    if (check_occ == 1)
      return true;
  }
  for (double dl = collision_interval_; dl < norm34; dl += collision_interval_) {
    point = dl / norm34 * (corner4 - corner3) + corner3;
    check_occ = checkCollision(point.x(), point.y(), height);
    if (check_occ == 1)
      return true;
  }
  for (double dl = collision_interval_; dl < norm41; dl += collision_interval_) {
    point = dl / norm41 * (corner1 - corner4) + corner4;
    check_occ = checkCollision(point.x(), point.y(), height);
    if (check_occ == 1)
      return true;
  }

  return false;
}

bool KinoAstar::checkCollision(double x, double y, double z)
{
  Eigen::Vector2d pos(x, y);
  if (map_->getOccupancy(pos) == SDFMap2D::OCCUPIED ||
      map_->getOccupancy(pos) == SDFMap2D::UNKNOWN) {
    return true;
  }
  return false;
}

double KinoAstar::getHeu(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = lambda_heu_ + 1e-6 * (dx + dy);

  return tie_breaker * sqrt(dx * dx + dy * dy);
}

// state1 is current state, state2 is goal state
bool KinoAstar::isShotSuccess(const Eigen::Vector4d& state1, const Eigen::Vector4d& state2)
{
  std::vector<Eigen::Vector3d> path_list;
  double len;

  for (int i = 0; i < shotptrsize; i++) {
    computeShotTraj(state1.head(3), state2.head(3), i, path_list, len);
    bool is_occ = false;

    for (unsigned int j = 0; j < path_list.size(); ++j) {
      if (isCollisionPosYaw(path_list[j].head(2), path_list[j][2])) {
        is_occ = true;
        break;
      }
    }
    if (!is_occ) {
      is_shot_succ_ = true;
      shotptrindex = i;
      return true;
    }
  }
  return false;
}

double KinoAstar::computeShotTraj(const Eigen::Vector3d& state1, const Eigen::Vector3d& state2,
    const int shotptrind, std::vector<Eigen::Vector3d>& path_list, double& len)
{
  namespace ob = ompl::base;
  namespace og = ompl::geometric;
  ompl::base::StateSpacePtr shotptr_ = shotptr_s[shotptrind];
  ob::ScopedState<> from(shotptr_), to(shotptr_), s(shotptr_);
  from[0] = state1[0];
  from[1] = state1[1];
  from[2] = state1[2];
  to[0] = state2[0];
  to[1] = state2[1];
  to[2] = state2[2];
  std::vector<double> reals;

  len = shotptr_->distance(from(), to());
  double sum_T = len / max_vel_;

  // sample to obtain intermediate point
  for (double l = 0.0; l <= len; l += checkl_) {
    shotptr_->interpolate(from(), to(), l / len, s());
    reals = s.reals();
    path_list.push_back(Eigen::Vector3d(reals[0], reals[1], reals[2]));
  }

  return sum_T;
}

// to retrieve the path in correct order by tracing parent nodes recorded by A*
void KinoAstar::retrievePath(const PathNodePtr& end_node)
{
  PathNodePtr cur_node = end_node;
  path_nodes_.push_back(cur_node);
  while (cur_node->parent != NULL) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node);
  }

  reverse(path_nodes_.begin(), path_nodes_.end());
}

// get the successor state given the input
void KinoAstar::stateTransit(
    const Eigen::Vector4d& state0, const Eigen::Vector3d& ctrl_input, Eigen::Vector4d& state1)
{
  double psi = ctrl_input[0];
  double s = ctrl_input[1];
  if (fabs(psi) > 1e-4) {
    // curved motion
    double k = wheel_base_ / tan(psi);
    state1[0] = state0[0] + k * (sin(state0[2] + s / k) - sin(state0[2]));
    state1[1] = state0[1] - k * (cos(state0[2] + s / k) - cos(state0[2]));
    state1[2] = state0[2] + s / k;
  }
  else {
    // straight-line motion
    state1[0] = state0[0] + s * cos(state0[2]);
    state1[1] = state0[1] + s * sin(state0[2]);
    state1[2] = state0[2];
  }
}

void KinoAstar::nodeVis(const Eigen::Vector3d& state)
{
  sensor_msgs::PointCloud2 globalMap_pcd;
  pcl::PointCloud<pcl::PointXYZ> cloudMap;
  pcl::PointXYZ pt;
  pt.x = state[0];
  pt.y = state[1];
  pt.z = 0.2;

  cloudMap.points.push_back(pt);
  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;
  pcl::toROSMsg(cloudMap, globalMap_pcd);
  globalMap_pcd.header.stamp = ros::Time::now();
  globalMap_pcd.header.frame_id = "base";
  expandNodes_pub_.publish(globalMap_pcd);

  return;
}

// get distance at current timestamp using trapezoidal velocity profile
double KinoAstar::evaluateLength(const double& curt, const double& locallength,
    const double& localtime, const double& startV, const double& endV)
{
  double critical_len;
  if (startV > max_vel_ * 1.5 || endV > max_vel_ * 1.5)
    ROS_ERROR_THROTTLE(1.0, "kinoAstar:evaluateLength:start or end vel is larger that the limit!");

  double startv2 = pow(startV, 2);
  double endv2 = pow(endV, 2);
  double maxv2 = pow(max_vel_, 2);
  critical_len = (maxv2 - startv2) / (2 * max_acc_) + (maxv2 - endv2) / (2 * max_acc_);
  // get time/distance from trapezoidal velocity profile
  if (locallength >= critical_len) {  // if locallength > critical_len, accelerate to max speed then
                                      // decelerate
    double t1 = (max_vel_ - startV) / max_acc_;
    double t2 = t1 + (locallength - critical_len) / max_vel_;
    if (curt <= t1) {
      return startV * curt + 0.5 * max_acc_ * pow(curt, 2);
    }
    else if (curt <= t2) {
      return startV * t1 + 0.5 * max_acc_ * pow(t1, 2) + (curt - t1) * max_vel_;
    }
    else {
      return startV * t1 + 0.5 * max_acc_ * pow(t1, 2) + (t2 - t1) * max_vel_ +
             max_vel_ * (curt - t2) - 0.5 * max_acc_ * pow(curt - t2, 2);
    }
  }
  else {  // if locallength < critical_len, do not reach max speed before decelerating
    double tmpv = sqrt(0.5 * (startv2 + endv2 + 2 * max_acc_ * locallength));
    double tmpt = (tmpv - startV) / max_acc_;
    if (curt <= tmpt) {
      return startV * curt + 0.5 * max_acc_ * pow(curt, 2);
    }
    else {
      return startV * tmpt + 0.5 * max_acc_ * pow(tmpt, 2) + tmpv * (curt - tmpt) -
             0.5 * max_acc_ * pow(curt - tmpt, 2);
    }
  }
}

// state:x y yaw v   flat_state: p v a
void KinoAstar::getFlatState(const Eigen::Vector4d& state, const Eigen::Vector2d& control_input,
    const int& singul, Eigen::MatrixXd& flat_state)
{

  flat_state.resize(2, 3);

  double angle = state(2);
  double vel = state(3);  // vel > 0

  Eigen::Matrix2d init_R;
  init_R << cos(angle), -sin(angle), sin(angle), cos(angle);

  if (abs(vel) <= non_siguav_) {
    vel = singul * non_siguav_;
  }
  else {
    vel = singul * vel;
  }
  //// here traveled distance is (oddly) used as acceleration
  flat_state << state.head(2), init_R * Eigen::Vector2d(vel, 0.0),
      init_R * Eigen::Vector2d(
                   control_input(1), std::tan(control_input(0)) / wheel_base_ * std::pow(vel, 2)),
      Eigen::Vector2d(0, 0);
}

// state:x y yaw v maniangle  flat_state: p v a
void KinoAstar::getFlatState(const double& x, const double& y, const double& angle, const double& v,
    const double& maniangle, const Eigen::Vector3d& control_input, const int& singul,
    Eigen::MatrixXd& flat_state)
{
  // columns: 1=p, 2=v, 3=a
  flat_state.resize(3, 4);

  // double angle = state(2);
  double vel = v;  // vel > 0

  Eigen::Matrix2d init_R;
  init_R << cos(angle), -sin(angle), sin(angle), cos(angle);

  if (abs(vel) <= non_siguav_) {
    vel = singul * non_siguav_;
  }
  else {
    vel = singul * vel;
  }
  //// here the traveled distance is (oddly) treated as acceleration
  Eigen::Vector3d fp(x, y, maniangle);
  Eigen::Vector3d fv(0, 0, 0);
  fv.head(2) = init_R * Eigen::Vector2d(vel, 0.0);
  Eigen::Vector3d fa(0, 0, 0);
  fa.head(2) = init_R * Eigen::Vector2d(control_input(1),
                            std::tan(control_input(0)) / wheel_base_ * std::pow(vel, 2));
  Eigen::Vector3d fj(0, 0, 0);
  flat_state << fp, fv, fa, fj;
}

// output x y yaw theta
Eigen::Vector4d KinoAstar::evaluatePos(const double& input_t)
{
  // ensure t > 0 and < totalTrajTime
  double t = input_t;
  if (t < 0 || t > totalTrajTime) {
    ROS_ERROR("In evaluatePos, t<0 || t>totalTrajTime");
    t = std::min<double>(std::max<double>(0, t), totalTrajTime);
  }
  double start_vel = fabs(start_state_[4]);
  double end_vel = fabs(end_state_[4]);
  int index = -1;
  double tmpT = 0;
  double CutTime;
  // locate the local traj: find the segment containing current time
  for (uint i = 0; i < shot_timeList.size(); i++) {
    tmpT += shot_timeList[i];
    if (tmpT >= t) {
      index = i;
      CutTime = t - tmpT + shot_timeList[i];
      break;
    }
  }
  // find the start and end velocities for the segment
  double initv = non_siguav_, finv = non_siguav_;
  if (index == 0) {
    initv = start_vel;
  }
  if (index == shot_lengthList.size() - 1)
    finv = end_vel;
  // find the time and length for the segment
  double localtime = shot_timeList[index];
  double locallength = shot_lengthList[index];
  // find two turning points from the turning-point array to select two points
  int front = shotindex[index];
  int back = shotindex[index + 1];
  // select the trajectory segment that contains t; initial and final velocities at endpoints
  std::vector<Eigen::Vector4d> localTraj;
  localTraj.assign(sample_traj_.begin() + front, sample_traj_.begin() + back + 1);
  // find the nearest point — compute path length at this time, find nearest point and interpolate
  // to get current state
  double arclength = evaluateLength(CutTime, locallength, localtime, initv, finv);
  double tmparc = 0;
  for (uint i = 0; i < localTraj.size() - 1; i++) {
    tmparc += evaluateDistance(localTraj[i].head(2), localTraj[i + 1].head(2));
    if (tmparc >= arclength) {
      double l1 = tmparc - arclength;
      double l = evaluateDistance(localTraj[i].head(2), localTraj[i + 1].head(2));
      double l2 = l - l1;  // l2
      // directly interpolate to obtain the state; yaw angle needs special handling
      Eigen::Vector4d state = l1 / l * localTraj[i] + l2 / l * localTraj[i + 1];
      if (fabs(localTraj[i + 1][2] - localTraj[i][2]) >= M_PI) {
        double normalize_yaw;
        if (localTraj[i + 1][2] <= 0) {
          normalize_yaw = l1 / l * localTraj[i][2] + l2 / l * (localTraj[i + 1][2] + 2 * M_PI);
        }
        else if (localTraj[i][2] <= 0) {
          normalize_yaw = l1 / l * (localTraj[i][2] + 2 * M_PI) + l2 / l * localTraj[i + 1][2];
        }
        state[2] = normalize_yaw;
      }
      ////  special handling for joint angles
      // std::cout<<"---------------------------------------------------"<<index<<"-----------------------------"<<std::endl;
      FlatTrajData kino_traj = flat_trajs_.at(index);
      std::vector<Eigen::Vector4d> pts = kino_traj.traj_pts;
      double sumt = 0.0;
      int ptssize = pts.size();
      // std::cout<<"---------------------------------------------------"<<ptssize<<"-----------------------------"<<std::endl;
      // std::cout<<"---------------------------------------------------"<<CutTime<<"-----------------------------"<<std::endl;
      for (int j = 0; j < ptssize; j++) {
        sumt += pts[j][2];
        if (sumt > CutTime) {
          if (j == ptssize - 1) {
            state[3] = pts[j][3];
            // std::cout<<"---------------------------------j==ptssize-1"<<std::endl;
          }
          else {
            state[3] = (sumt - CutTime) / pts[j][2] * pts[j + 1][3] +
                       (1 - (sumt - CutTime) / pts[j][2]) * pts[j][3];
          }
          break;
        }
      }
      return state;
    }
  }
  return localTraj.back();
}

void KinoAstar::kinoastarFlatPathPub(const std::vector<FlatTrajData> flat_trajs)
{
  if (!has_path_)
    return;
  nav_msgs::Path path;
  path.header.frame_id = "world";
  path.header.stamp = ros::Time::now();
  geometry_msgs::PoseStamped pose;
  pose.header.frame_id = "world";

  visualization_msgs::MarkerArray markerarraydelete;
  visualization_msgs::MarkerArray markerarray;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.ns = "kinoastarFlatPath";

  marker.lifetime = ros::Duration();
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::DELETEALL;
  marker.scale.x = 0.04;
  marker.scale.y = 0.04;
  marker.scale.z = 0.02;
  marker.color.a = 0.6;
  marker.color.r = 195.0 / 255;
  marker.color.g = 176.0 / 255;
  marker.color.b = 145.0 / 255;
  marker.pose.position.z = 0.1;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  marker.header.stamp = ros::Time::now();
  marker.id = 0;
  marker.pose.position.x = flat_trajs[0].traj_pts[0].x();
  marker.pose.position.y = flat_trajs[0].traj_pts[0].y();
  markerarraydelete.markers.push_back(marker);
  kinoastarFlatPathPub_.publish(markerarraydelete);

  marker.action = visualization_msgs::Marker::ADD;
  for (uint i = 0; i < flat_trajs.size(); i++) {
    marker.pose.position.x = flat_trajs[i].start_state.col(0).x();
    marker.pose.position.y = flat_trajs[i].start_state.col(0).y();
    pose.pose.position.z = 0.1;

    markerarray.markers.push_back(marker);
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = flat_trajs[i].start_state.col(0).x();
    pose.pose.position.y = flat_trajs[i].start_state.col(0).y();
    pose.pose.position.z = 0.1;
    path.poses.push_back(pose);

    for (uint j = 0; j < flat_trajs[i].traj_pts.size(); j++) {
      marker.header.stamp = ros::Time::now();
      marker.id = j * 1000 + (i + 1);
      marker.pose.position.x = flat_trajs[i].traj_pts[j].x();
      marker.pose.position.y = flat_trajs[i].traj_pts[j].y();
      markerarray.markers.push_back(marker);

      pose.header.stamp = ros::Time::now();
      pose.pose.position.x = flat_trajs[i].traj_pts[j].x();
      pose.pose.position.y = flat_trajs[i].traj_pts[j].y();
      pose.pose.position.z = 0.1;
      pose.pose.orientation = tf::createQuaternionMsgFromYaw(flat_trajs[i].thetas[j]);
      path.poses.push_back(pose);
    }
  }
  kinoastarFlatPathPub_.publish(markerarray);
  kinoastarFlatTrajPub_.publish(path);
}

}  // namespace apexnav_planner