// Temporal filtering of the control quantities. See temporal.hpp.

#include "curvature_port/temporal.hpp"

#include <cmath>

namespace curvature_port {

ConstantVelocityFilter::ConstantVelocityFilter(double process_var,
                                               double measurement_var,
                                               double gate_sigma)
    : process_var_(process_var),
      measurement_var_(measurement_var),
      gate_sigma_(gate_sigma) {
  Reset();
}

void ConstantVelocityFilter::Reset() {
  x_[0] = x_[1] = 0.0;
  // A large initial covariance, so the first accepted measurement dominates.
  p_[0][0] = p_[1][1] = 1e3;
  p_[0][1] = p_[1][0] = 0.0;
  initialized_ = false;
}

double ConstantVelocityFilter::Predict(double dt) {
  // x = A x with A = [[1, dt], [0, 1]].
  x_[0] += dt * x_[1];

  // P = A P A^T + Q, with Q the continuous white-noise acceleration discretized over dt.
  const double p00 = p_[0][0] + dt * (p_[1][0] + p_[0][1]) + dt * dt * p_[1][1];
  const double p01 = p_[0][1] + dt * p_[1][1];
  const double p10 = p_[1][0] + dt * p_[1][1];
  const double p11 = p_[1][1];

  const double dt2 = dt * dt, dt3 = dt2 * dt;
  p_[0][0] = p00 + process_var_ * dt3 / 3.0;
  p_[0][1] = p01 + process_var_ * dt2 / 2.0;
  p_[1][0] = p10 + process_var_ * dt2 / 2.0;
  p_[1][1] = p11 + process_var_ * dt;
  return value();
}

bool ConstantVelocityFilter::Update(double measurement, double dt) {
  if (!initialized_) {
    // Seed on the first measurement rather than dragging up from zero.
    x_[0] = measurement;
    x_[1] = 0.0;
    p_[0][0] = measurement_var_;
    p_[1][1] = process_var_;
    p_[0][1] = p_[1][0] = 0.0;
    initialized_ = true;
    return true;
  }

  Predict(dt);
  const double residual = measurement - x_[0];
  const double innovation_var = p_[0][0] + measurement_var_;
  if (gate_sigma_ > 0.0 && innovation_var > 0.0) {
    if (std::abs(residual) > gate_sigma_ * std::sqrt(innovation_var)) return false;
  }

  // Gain is the first column of P over the innovation variance.
  const double k0 = p_[0][0] / innovation_var;
  const double k1 = p_[1][0] / innovation_var;
  x_[0] += k0 * residual;
  x_[1] += k1 * residual;

  // P -= outer(K, P[0, :]).
  const double p_row0 = p_[0][0], p_row1 = p_[0][1];
  p_[0][0] -= k0 * p_row0;
  p_[0][1] -= k0 * p_row1;
  p_[1][0] -= k1 * p_row0;
  p_[1][1] -= k1 * p_row1;
  return true;
}

RoadGeometryFilter::RoadGeometryFilter(FilterConfig cfg)
    : cfg_(cfg),
      offset_(cfg.offset_process_var, cfg.offset_measurement_var, cfg.gate_sigma),
      heading_(cfg.heading_process_var, cfg.heading_measurement_var, cfg.gate_sigma),
      curvature_(cfg.curvature_process_var, cfg.curvature_measurement_var,
                 cfg.gate_sigma) {}

void RoadGeometryFilter::Reset() {
  offset_.Reset();
  heading_.Reset();
  curvature_.Reset();
  coasting_ = 0;
}

FilteredGeometry RoadGeometryFilter::Update(bool has_geometry, double lateral_offset_m,
                                            double heading_error_rad,
                                            double curvature_1pm) {
  FilteredGeometry out;
  if (!has_geometry) {
    offset_.Predict(cfg_.dt);
    heading_.Predict(cfg_.dt);
    curvature_.Predict(cfg_.dt);
    ++coasting_;
    out.measured = false;
    out.accepted = false;
  } else {
    const bool a0 = offset_.Update(lateral_offset_m, cfg_.dt);
    const bool a1 = heading_.Update(heading_error_rad, cfg_.dt);
    const bool a2 = curvature_.Update(curvature_1pm, cfg_.dt);
    // A frame counts as tracked only if the offset was accepted: it is the quantity the
    // controller is most sensitive to.
    coasting_ = a0 ? 0 : coasting_ + 1;
    out.measured = true;
    out.accepted = a0 && a1 && a2;
  }
  out.lateral_offset_m = offset_.value();
  out.heading_error_rad = heading_.value();
  out.curvature_1pm = curvature_.value();
  out.coasting_frames = coasting_;
  return out;
}

}  // namespace curvature_port
