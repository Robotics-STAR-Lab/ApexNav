#include "trajectory_manager/optimizer.h"

namespace apexnav_planner {

void Gcopter::attachPenaltyFunctional(const int& traj_id, double& cost)
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
        cost += omg * step * colli_weight_ * violaPosPena;
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

void Gcopter::visInnerPoints()
{
  visualization_msgs::msg::MarkerArray markerarraydelete;
  visualization_msgs::msg::MarkerArray markerarray;
  visualization_msgs::msg::Marker marker;

  marker.header.frame_id = "world";
  marker.ns = "initinnerPoint";
  marker.lifetime = rclcpp::Duration(0, 0);
  marker.type = visualization_msgs::msg::Marker::CYLINDER;

  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  markerarraydelete.markers.push_back(marker);
  inner_init_point_pub_->publish(markerarraydelete);

  marker.action = visualization_msgs::msg::Marker::ADD;
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
      marker.type = visualization_msgs::msg::Marker::CYLINDER;
      marker.header.stamp = node_->get_clock()->now();
      marker.id = j * 10000 + i * 100;
      marker.pose.position.x = innerPointses[i].col(j).x();
      marker.pose.position.y = innerPointses[i].col(j).y();
      markerarray.markers.push_back(marker);

      marker.scale.z = 0.2;
      marker.color.a = 1.0;
      std::ostringstream str;
      marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      double mani_angle =
          fabs(innerPointses[i].col(j).z()) > 1e-4 ? innerPointses[i].col(j).z() : 0.0;
      str << mani_angle;
      marker.text = str.str();
      marker.id = j * 10000 + i * 100 + 1;
      markerarray.markers.push_back(marker);
    }
  }
  inner_init_point_pub_->publish(markerarray);
}

void Gcopter::visFinalInnerPoints()
{
  visualization_msgs::msg::MarkerArray markerarraydelete;
  visualization_msgs::msg::MarkerArray markerarray;
  visualization_msgs::msg::Marker marker;

  marker.header.frame_id = "world";
  marker.ns = "innerPoint";
  marker.lifetime = rclcpp::Duration(0, 0);
  marker.type = visualization_msgs::msg::Marker::CYLINDER;

  marker.action = visualization_msgs::msg::Marker::DELETEALL;
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
  marker.header.stamp = node_->get_clock()->now();
  marker.id = 0;
  markerarray.markers.push_back(marker);
  inner_point_pub_->publish(markerarray);
  markerarray.markers.clear();
  marker.action = visualization_msgs::msg::Marker::ADD;

  for (uint i = 0; i < finalInnerpointses.size(); i++) {
    for (uint j = 0; j < finalInnerpointses[i].cols(); j++) {
      marker.header.stamp = node_->get_clock()->now();
      marker.id = j * 100 + i * 1;
      marker.pose.position.x = finalInnerpointses[i].col(j).x();
      marker.pose.position.y = finalInnerpointses[i].col(j).y();
      markerarray.markers.push_back(marker);
    }
  }
  inner_point_pub_->publish(markerarray);
}

void Gcopter::mincoInitTrajPub(
    const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls)
{
  if (final_trajes.size() != static_cast<size_t>(final_singuls.size()))
    RCLCPP_ERROR(node_->get_logger(), "[mincoInitTrajPub] Input size ERROR !!!!");

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

  nav_msgs::msg::Path path;
  path.header.frame_id = "world";
  path.header.stamp = node_->get_clock()->now();

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

    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = node_->get_clock()->now();
    pose.pose.position.x = (currPos).x();
    pose.pose.position.y = (currPos).y();
    pose.pose.position.z = 0.15;

    pose.pose.orientation = createQuaternionMsgFromYaw(yaw);
    path.poses.push_back(pose);
  }

  minco_init_path_pub_->publish(path);
}

void Gcopter::mincoInitPathPubwithAlpha(const std::vector<Trajectory<7, 3>>& final_trajes,
    const Eigen::VectorXi& final_singuls, const int& k)
{
  if (final_trajes.size() != static_cast<size_t>(final_singuls.size()))
    RCLCPP_ERROR(node_->get_logger(), "[mincoInitTrajPub] Input size ERROR !!!!");

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

  visualization_msgs::msg::MarkerArray markerarraydelete;
  visualization_msgs::msg::MarkerArray markerarray;
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "world";
  marker.ns = "minco_opt_path_alpha_pub_";
  marker.lifetime = rclcpp::Duration(0, 0);
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
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

  marker.header.stamp = node_->get_clock()->now();
  marker.id = 0;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  markerarraydelete.markers.push_back(marker);
  marker.action = visualization_msgs::msg::Marker::ADD;

  nav_msgs::msg::Path path;
  path.header.frame_id = "world";
  path.header.stamp = node_->get_clock()->now();

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

    Eigen::Matrix2d B_h_local;
    B_h_local << 0, -1.0, 1.0, 0;
    Eigen::VectorXd currAcc = final_trajes[index].getAcc(time - index_time).head(2);
    Eigen::VectorXd currJer = final_trajes[index].getJer(time - index_time).head(2);
    double normVel = currVel.head(2).norm();
    double help1 = 1 / (normVel * normVel);
    double z_h1 = currAcc.transpose() * currVel.head(2);
    double z_h3 = currAcc.transpose() * B_h_local * currVel.head(2);

    double help2 = currJer.transpose() * B_h_local * currVel.head(2);
    double domega = help2 * help1 - 2.0 * help1 * help1 * z_h3 * z_h1;

    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = node_->get_clock()->now();
    pose.pose.position.x = 5.0 + time;
    pose.pose.position.y = domega / 2.0;
    pose.pose.position.z = 0;

    pose.pose.orientation = createQuaternionMsgFromYaw(0.0);
    path.poses.push_back(pose);

    double index_piece_time = 0;
    Eigen::VectorXd trajDurations = final_trajes[index].getDurations();
    for (int pieceindex = 0; pieceindex < final_trajes[index].getPieceNum(); pieceindex++) {
      if (time > index_piece_time && time < index_piece_time + trajDurations[pieceindex]) {
        if (std::fmod(time - index_piece_time, trajDurations[pieceindex] / 8.0) < 1.1e-4) {
          marker.header.stamp = node_->get_clock()->now();
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
    minco_opt_path_alpha_pub_->publish(markerarraydelete);
    minco_opt_path_alpha_pub_->publish(markerarray);
  }
  minco_init_path_alpha_pub_->publish(path);
}

void Gcopter::mincoPathPub(
    const std::vector<Trajectory<7, 3>>& final_trajes, const Eigen::VectorXi& final_singuls)
{
  if (final_trajes.size() != static_cast<size_t>(final_singuls.size()))
    RCLCPP_ERROR(node_->get_logger(), "[mincoCarPathPub] Input size ERROR !!!!");

  int traj_size = final_trajes.size();
  double total_time;
  Eigen::VectorXd traj_time;
  traj_time.resize(traj_size);
  for (int i = 0; i < traj_size; i++) traj_time[i] = final_trajes[i].getTotalDuration();
  total_time = traj_time.sum();

  int index = 0;
  Eigen::VectorXd currPos, currVel;

  nav_msgs::msg::Path path;
  path.header.frame_id = "world";
  path.header.stamp = node_->get_clock()->now();

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

    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = node_->get_clock()->now();
    pose.pose.position.x = (currPos).x();
    pose.pose.position.y = (currPos).y();
    pose.pose.position.z = 0.15;

    pose.pose.orientation = createQuaternionMsgFromYaw(yaw);
    path.poses.push_back(pose);
  }

  minco_path_pub_->publish(path);
}

}  // namespace apexnav_planner
