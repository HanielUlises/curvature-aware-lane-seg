// Lateral MPC over a kinematic bicycle. See mpc.hpp.

#include "curvature_port/mpc.hpp"

#include <algorithm>
#include <cmath>

namespace curvature_port {

KinematicLateralMPC::KinematicLateralMPC(VehicleParams params, MPCWeights weights,
                                         int horizon)
    : params_(params), weights_(weights), horizon_(std::max(horizon, 1)) {}

void KinematicLateralMPC::Condense(double speed) {
  const int n = horizon_;
  const double dt = params_.dt;
  Eigen::Matrix2d a;
  a << 1.0, speed * dt, 0.0, 1.0;
  Eigen::Vector2d b(0.0, speed * dt / params_.wheelbase_m);

  sx_.setZero(2 * n, 2);
  su_.setZero(2 * n, n);
  sd_.setZero(2 * n, 2 * n);

  // A^i for i = 1..n, built incrementally; A^(i-j) multiplies both input j and
  // disturbance j.
  std::vector<Eigen::Matrix2d> a_pow(static_cast<std::size_t>(n) + 1);
  a_pow[0] = Eigen::Matrix2d::Identity();
  for (int i = 1; i <= n; ++i) a_pow[static_cast<std::size_t>(i)] = a_pow[static_cast<std::size_t>(i - 1)] * a;

  for (int i = 0; i < n; ++i) {
    sx_.block<2, 2>(2 * i, 0) = a_pow[static_cast<std::size_t>(i + 1)];
    for (int j = 0; j <= i; ++j) {
      const Eigen::Matrix2d& ak = a_pow[static_cast<std::size_t>(i - j)];
      su_.block<2, 1>(2 * i, j) = ak * b;
      sd_.block<2, 2>(2 * i, 2 * j) = ak;
    }
  }

  // Hessian is speed-dependent but state-independent, so factor it once per speed.
  Eigen::VectorXd q_diag(2 * n);
  for (int i = 0; i < n; ++i) {
    q_diag(2 * i) = weights_.cross_track;
    q_diag(2 * i + 1) = weights_.heading;
  }
  hessian_ = su_.transpose() * q_diag.asDiagonal() * su_ +
             Eigen::MatrixXd::Identity(n, n) * weights_.steer;
  hessian_llt_ = hessian_.llt();
  condensed_speed_ = speed;
}

MPCSolution KinematicLateralMPC::Solve(double cross_track_m, double heading_rad,
                                       double curvature_1pm, double speed_mps) {
  MPCSolution out;
  // The lateral dynamics vanish at a standstill: B is proportional to speed, so there is
  // no steering that changes the state and the problem is degenerate.
  if (!(speed_mps > 0.0)) return out;

  const int n = horizon_;
  if (condensed_speed_ != speed_mps) Condense(speed_mps);

  Eigen::Vector2d x0(cross_track_m, heading_rad);
  Eigen::VectorXd disturbance(2 * n);
  const double d = -speed_mps * params_.dt * curvature_1pm;
  for (int i = 0; i < n; ++i) {
    disturbance(2 * i) = 0.0;
    disturbance(2 * i + 1) = d;
  }

  Eigen::VectorXd q_diag(2 * n);
  for (int i = 0; i < n; ++i) {
    q_diag(2 * i) = weights_.cross_track;
    q_diag(2 * i + 1) = weights_.heading;
  }

  const Eigen::VectorXd free = sx_ * x0 + sd_ * disturbance;
  const Eigen::VectorXd gradient = su_.transpose() * q_diag.asDiagonal() * free;
  const Eigen::VectorXd planned = hessian_llt_.solve(-gradient);
  const Eigen::VectorXd states = free + su_ * planned;

  const double raw = planned(0);
  const double limit = params_.max_steer_rad;
  const double clipped = std::min(std::max(raw, -limit), limit);

  out.steer_rad = clipped;
  out.steer_unsaturated_rad = raw;
  out.planned_steer_rad.assign(planned.data(), planned.data() + planned.size());
  out.predicted_states.assign(states.data(), states.data() + states.size());
  out.saturated = clipped != raw;
  out.valid = true;
  return out;
}

MPCSolution KinematicLateralMPC::SteerForGeometry(double lateral_offset_m,
                                                  double heading_error_rad,
                                                  double curvature_1pm,
                                                  double speed_mps) {
  // Measured quantities are right-positive; the model above is left-positive. Curvature
  // flips going in and the steer flips coming back out.
  MPCSolution s = Solve(lateral_offset_m, heading_error_rad, -curvature_1pm, speed_mps);
  if (!s.valid) return s;
  const double raw = -s.steer_unsaturated_rad;
  const double limit = params_.max_steer_rad;
  s.steer_unsaturated_rad = raw;
  s.steer_rad = std::min(std::max(raw, -limit), limit);
  return s;
}

}  // namespace curvature_port
