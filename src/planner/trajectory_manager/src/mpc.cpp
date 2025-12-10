#include "controller/mpc.h"

using namespace std;

void MPC::init(ros::NodeHandle& nh)
{
  nh.param("mpc/du_threshold", du_th, -1.0);
  nh.param("mpc/dt", dt, -1.0);
  nh.param("mpc/max_iter", max_iter, -1);
  nh.param("mpc/predict_steps", T, -1);
  nh.param("mpc/delay_num", delay_num, -1);
  nh.param("max_correction_vel", max_speed, -1.0);
  nh.param("max_correction_omega", max_omega, -1.0);
  min_speed = 0.0;
  max_accel = max_speed;
  nh.param("mpc/tolerance", tolerance, 0.1);
  nh.param<std::vector<double>>("mpc/matrix_q", Q, std::vector<double>());
  nh.param<std::vector<double>>("mpc/matrix_r", R, std::vector<double>());
  nh.param<std::vector<double>>("mpc/matrix_rd", Rd, std::vector<double>());

  has_odom = false;
  receive_traj_ = false;
  max_comega = max_omega / 3.0 * 2.0 * dt;
  max_cv = max_accel * dt;
  xref = Eigen::Matrix<double, 4, 500>::Zero(4, 500);
  last_output = output = dref = Eigen::Matrix<double, 2, 500>::Zero(2, 500);
  for (int i = 0; i < delay_num; i++) output_buff.push_back(Eigen::Vector2d::Zero());

  predict_pub = nh.advertise<visualization_msgs::Marker>("mpc_car/predict_path", 10);
  ref_pub = nh.advertise<visualization_msgs::Marker>("mpc_car/reference_path", 10);
  err_pub = nh.advertise<std_msgs::Float64>("mpc_car/track_err", 10);
}

void MPC::setOdom(const Eigen::Vector4d& car_state)
{
  has_odom = true;
  now_state.x = car_state(0);
  now_state.y = car_state(1);
  now_state.yaw = car_state(2);
  now_state.v = car_state(3);
}

Eigen::Vector2d MPC::calCmd(const std::vector<Eigen::Vector3d>& _xref)
{
  std_msgs::Float64 err;
  err.data = Eigen::Vector2d(now_state.x - _xref[0](0), now_state.y - _xref[0](1)).norm();
  err_pub.publish(err);
  for (int i = 0; i < T; i++) {
    xref(0, i) = _xref[i](0);
    xref(1, i) = _xref[i](1);
    xref(3, i) = _xref[i](2);
    dref(0, i) = 0.0;
    dref(1, i) = 0.0;
  }
  smooth_yaw();
  getCmd();
  Eigen::Vector2d cmd;
  cmd << output(0, delay_num), output(1, delay_num);
  return cmd;
}

void MPC::getLinearModel(const MPCState& s)
{
  B = Eigen::Matrix<double, 3, 2>::Zero();
  B(0, 0) = cos(s.yaw) * dt;
  B(1, 0) = sin(s.yaw) * dt;
  B(2, 1) = dt;

  A = Eigen::Matrix3d::Identity();
  A(0, 2) = -B(1, 0) * s.v;
  A(1, 2) = B(0, 0) * s.v;

  C = Eigen::Vector3d::Zero();
  C(0) = -A(0, 2) * s.yaw;
  C(1) = -A(1, 2) * s.yaw;
}

void MPC::stateTrans(MPCState& s, double v, double yaw_dot)
{
  if (yaw_dot >= max_omega) {
    yaw_dot = max_omega;
  }
  else if (yaw_dot <= -max_omega) {
    yaw_dot = -max_omega;
  }
  if (s.v >= max_speed) {
    s.v = max_speed;
  }
  else if (s.v <= min_speed) {
    s.v = min_speed;
  }

  s.x = s.x + v * cos(s.yaw) * dt;
  s.y = s.y + v * sin(s.yaw) * dt;
  s.yaw = s.yaw + yaw_dot * dt;
  s.v = v;
}

void MPC::predictMotion(void)
{
  xbar[0] = now_state;

  MPCState temp = now_state;
  for (int i = 1; i < T + 1; i++) {
    stateTrans(temp, output(0, i - 1), output(1, i - 1));
    xbar[i] = temp;
  }
}

void MPC::predictMotion(MPCState* b)
{
  b[0] = xbar[0];

  Eigen::MatrixXd Ax;
  Eigen::MatrixXd Bx;
  Eigen::MatrixXd Cx;
  Eigen::MatrixXd xnext;
  MPCState temp = xbar[0];

  for (int i = 1; i < T + 1; i++) {
    Bx = Eigen::Matrix<double, 3, 2>::Zero();
    Bx(0, 0) = cos(xbar[i - 1].yaw) * dt;
    Bx(1, 0) = sin(xbar[i - 1].yaw) * dt;
    Bx(2, 1) = dt;

    Ax = Eigen::Matrix3d::Identity();
    Ax(0, 2) = -Bx(1, 0) * xbar[i - 1].v;
    Ax(1, 2) = Bx(0, 0) * xbar[i - 1].v;

    Cx = Eigen::Vector3d::Zero();
    Cx(0) = -Ax(0, 2) * xbar[i - 1].yaw;
    Cx(1) = -Ax(1, 2) * xbar[i - 1].yaw;
    xnext = Ax * Eigen::Vector3d(temp.x, temp.y, temp.yaw) +
            Bx * Eigen::Vector2d(output(0, i - 1), output(1, i - 1)) + Cx;
    temp.x = xnext(0);
    temp.y = xnext(1);
    temp.yaw = xnext(2);
    b[i] = temp;
  }
}

void MPC::solveMPCV()
{
  const int dimx = 3 * (T - delay_num);
  const int dimu = 2 * (T - delay_num);
  const int nx = dimx + dimu;

  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nx);
  Eigen::SparseMatrix<double> linearMatrix;
  Eigen::VectorXd lowerBound;
  Eigen::VectorXd upperBound;

  // first-order
  for (int i = 0, j = delay_num, k = 0; i < dimx; i += 3, j++, k += 2) {
    gradient[i] = -2 * Q[0] * xref(0, j);
    gradient[i + 1] = -2 * Q[1] * xref(1, j);
    gradient[i + 2] = -2 * Q[3] * xref(3, j);
    gradient[dimx + k] = -2 * Q[2] * dref(0, j);
  }

  // second-order
  const int nnzQ = nx + dimu - 2;
  int irowQ[nnzQ];
  int jcolQ[nnzQ];
  double dQ[nnzQ];
  for (int i = 0; i < nx; i++) {
    irowQ[i] = jcolQ[i] = i;
  }
  for (int i = nx; i < nnzQ; i++) {
    irowQ[i] = i - dimu + 2;
    jcolQ[i] = i - dimu;
  }
  for (int i = 0; i < dimx; i += 3) {
    dQ[i] = Q[0] * 2.0;
    dQ[i + 1] = Q[1] * 2.0;
    dQ[i + 2] = Q[3] * 2.0;
  }
  dQ[dimx] = dQ[nx - 2] = (R[0] + Rd[0] + Q[2]) * 2.0;
  dQ[dimx + 1] = dQ[nx - 1] = (R[1] + Rd[1]) * 2.0;
  for (int i = dimx + 2; i < nx - 2; i += 2) {
    dQ[i] = 2 * (R[0] + 2 * Rd[0] + Q[2]);
    dQ[i + 1] = 2 * (R[1] + 2 * Rd[1]);
  }
  for (int i = nx; i < nnzQ; i += 2) {
    dQ[i] = -Rd[0] * 2.0;
    dQ[i + 1] = -Rd[1] * 2.0;
  }
  hessian.resize(nx, nx);
  Eigen::MatrixXd QQ(nx, nx);
  for (int i = 0; i < nx; i++) {
    hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
  }
  for (int i = nx; i < nnzQ; i++) {
    hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
    hessian.insert(jcolQ[i], irowQ[i]) = dQ[i];
  }

  // equality constraints
  MPCState temp = xbar[delay_num];
  getLinearModel(temp);
  int my = dimx;
  double b[my];
  const int nnzA = 11 * (T - delay_num) - 5;
  int irowA[nnzA];
  int jcolA[nnzA];
  double dA[nnzA];
  Eigen::Vector3d temp_vec(temp.x, temp.y, temp.yaw);
  Eigen::Vector3d temp_b = A * temp_vec + C;

  for (int i = 0; i < dimx; i++) {
    irowA[i] = jcolA[i] = i;
    dA[i] = 1;
  }
  b[0] = temp_b[0];
  b[1] = temp_b[1];
  b[2] = temp_b[2];
  irowA[dimx] = 0;
  jcolA[dimx] = dimx;
  dA[dimx] = -B(0, 0);
  irowA[dimx + 1] = 1;
  jcolA[dimx + 1] = dimx;
  dA[dimx + 1] = -B(1, 0);
  irowA[dimx + 2] = 2;
  jcolA[dimx + 2] = dimx + 1;
  dA[dimx + 2] = -B(2, 1);
  int ABidx = 8 * (T - delay_num) - 8;
  int ABbegin = dimx + 3;
  for (int i = 0, j = 1; i < ABidx; i += 8, j++) {
    getLinearModel(xbar[j + delay_num]);
    for (int k = 0; k < 3; k++) {
      b[3 * j + k] = C[k];
      irowA[ABbegin + i + k] = 3 * j + k;
      jcolA[ABbegin + i + k] = irowA[ABbegin + i + k] - 3;
      dA[ABbegin + i + k] = -A(k, k);
    }
    irowA[ABbegin + i + 3] = 3 * j;
    jcolA[ABbegin + i + 3] = 3 * j - 1;
    dA[ABbegin + i + 3] = -A(0, 2);

    irowA[ABbegin + i + 4] = 3 * j + 1;
    jcolA[ABbegin + i + 4] = 3 * j - 1;
    dA[ABbegin + i + 4] = -A(1, 2);

    irowA[ABbegin + i + 5] = 3 * j;
    jcolA[ABbegin + i + 5] = dimx + 2 * j;
    dA[ABbegin + i + 5] = -B(0, 0);

    irowA[ABbegin + i + 6] = 3 * j + 1;
    jcolA[ABbegin + i + 6] = dimx + 2 * j;
    dA[ABbegin + i + 6] = -B(1, 0);

    irowA[ABbegin + i + 7] = 3 * j + 2;
    jcolA[ABbegin + i + 7] = dimx + 2 * j + 1;
    dA[ABbegin + i + 7] = -B(2, 1);
  }

  // iequality constraints
  const int mz = 2 * (T - delay_num) - 2;
  const int nnzC = 2 * dimu - 4;
  int irowC[nnzC];
  int jcolC[nnzC];
  double dC[nnzC];
  for (int i = 0, k = 0; i < mz; i += 2, k += 4) {
    irowC[k] = i;
    jcolC[k] = dimx + i;
    dC[k] = -1.0;

    irowC[k + 1] = i;
    jcolC[k + 1] = jcolC[k] + 2;
    dC[k + 1] = 1.0;

    irowC[k + 2] = i + 1;
    jcolC[k + 2] = dimx + 1 + i;
    dC[k + 2] = -1.0;

    irowC[k + 3] = i + 1;
    jcolC[k + 3] = jcolC[k + 2] + 2;
    dC[k + 3] = 1.0;
  }

  // xlimits and all
  int mx = dimu;
  int nc = mx + my + mz;
  lowerBound.resize(nc);
  upperBound.resize(nc);
  linearMatrix.resize(nc, nx);
  for (int i = 0; i < mx; i += 2) {
    lowerBound[i] = min_speed;
    lowerBound[i + 1] = -max_omega;
    upperBound[i] = max_speed;
    upperBound[i + 1] = max_omega;
    linearMatrix.insert(i, dimx + i) = 1;
    linearMatrix.insert(i + 1, dimx + i + 1) = 1;
  }

  for (int i = 0; i < nnzA; i++) {
    linearMatrix.insert(irowA[i] + mx, jcolA[i]) = dA[i];
  }

  for (int i = 0; i < my; i++) {
    lowerBound[mx + i] = upperBound[mx + i] = b[i];
  }

  for (int i = 0; i < nnzC; i++) {
    linearMatrix.insert(irowC[i] + mx + my, jcolC[i]) = dC[i];
  }

  for (int i = 0; i < mz; i += 2) {
    lowerBound[mx + my + i] = -max_cv;
    upperBound[mx + my + i] = max_cv;
    lowerBound[mx + my + i + 1] = -max_comega;
    upperBound[mx + my + i + 1] = max_comega;
  }

  // instantiate the solver
  OsqpEigen::Solver solver;

  // settings
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);
  solver.settings()->setAbsoluteTolerance(1e-6);
  solver.settings()->setMaxIteration(30000);
  solver.settings()->setRelativeTolerance(1e-6);

  // set the initial data of the QP solver
  solver.data()->setNumberOfVariables(nx);
  solver.data()->setNumberOfConstraints(nc);
  if (!solver.data()->setHessianMatrix(hessian))
    return;
  if (!solver.data()->setGradient(gradient))
    return;
  if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
    return;
  if (!solver.data()->setLowerBound(lowerBound))
    return;
  if (!solver.data()->setUpperBound(upperBound))
    return;

  // instantiate the solver
  if (!solver.initSolver())
    return;

  // controller input and QPSolution vector
  Eigen::VectorXd QPSolution;

  // solve the QP problem
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    return;

  // get the controller input
  QPSolution = solver.getSolution();
  // ROS_INFO("Solution: v0=%f     omega0=%f", QPSolution[dimx], QPSolution[dimx+1]);
  for (int i = 0; i < delay_num; i++) {
    output(0, i) = output_buff[i][0];
    output(1, i) = output_buff[i][1];
  }
  for (int i = 0, j = 0; i < dimu; i += 2, j++) {
    output(0, j + delay_num) = QPSolution[dimx + i];
    output(1, j + delay_num) = QPSolution[dimx + i + 1];
  }
}

void MPC::getCmd()
{
  int iter;
  ros::Time begin = ros::Time::now();
  for (iter = 0; iter < max_iter; iter++) {
    predictMotion();
    last_output = output;
    solveMPCV();
    double du = 0;
    for (int i = 0; i < output.cols(); i++) {
      du = du + fabs(output(0, i) - last_output(0, i)) + fabs(output(1, i) - last_output(1, i));
    }
    // break;
    if (du <= du_th || (ros::Time::now() - begin).toSec() > 0.01) {
      break;
    }
  }
  if (iter == max_iter) {
    ROS_WARN("MPC Iterative is max iter");
  }

  predictMotion(xopt);
  drawRefPath();
  drawPredictPath(xopt);
  // cmd.speed = output(0, delay_num);
  // cmd.omega = output(1, delay_num);
  if (delay_num > 0) {
    output_buff.erase(output_buff.begin());
    output_buff.push_back(Eigen::Vector2d(output(0, delay_num), output(1, delay_num)));
  }
}
