// Temporal filtering of the control quantities.
//
// C++ port of src/geometry/temporal.py. Geometry is estimated independently per frame,
// so the signal handed to a controller carries the full per-frame estimation noise and
// disappears entirely on frames where no ego lane is recovered. Neither is acceptable as
// a control input: the first is steering jitter, the second is a dropout.
//
// Two properties matter on the vehicle. A missed detection advances the state on the
// motion model instead of returning nothing, and the filter reports how long it has been
// doing so, which is what a supervisor watches to hand over before the extrapolation is
// trusted too far. And a measurement far outside the predicted distribution is rejected
// rather than followed, which stops one badly placed centreline stepping the steering.
//
// Fixed size and allocation-free: the state is two doubles and a 2x2 covariance.
//
// See docs/geometry_port_spec.md, section 11.

#ifndef CURVATURE_PORT_TEMPORAL_HPP
#define CURVATURE_PORT_TEMPORAL_HPP

namespace curvature_port {

// Scalar Kalman filter over [value, rate] with a constant-rate model.
class ConstantVelocityFilter {
 public:
  // gate_sigma <= 0 disables gating.
  ConstantVelocityFilter(double process_var = 1.0, double measurement_var = 1.0,
                         double gate_sigma = 4.0);

  void Reset();
  // Advance one step with no measurement. Returns the predicted value.
  double Predict(double dt = 1.0);
  // Advance and fold in a measurement. False means the measurement was gated out; the
  // state still advanced on the motion model.
  bool Update(double measurement, double dt = 1.0);

  double value() const { return x_[0]; }
  double rate() const { return x_[1]; }
  double variance() const { return p_[0][0]; }
  bool initialized() const { return initialized_; }

 private:
  double process_var_, measurement_var_, gate_sigma_;
  double x_[2];
  double p_[2][2];
  bool initialized_ = false;
};

struct FilteredGeometry {
  double lateral_offset_m = 0.0;
  double heading_error_rad = 0.0;
  double curvature_1pm = 0.0;
  bool measured = false;      // a geometry was supplied this frame
  bool accepted = false;      // and every one of the three passed its gate
  int coasting_frames = 0;    // consecutive frames without an accepted offset
};

struct FilterConfig {
  double dt = 0.05;
  double offset_process_var = 4.0;
  double heading_process_var = 1.0;
  double curvature_process_var = 1e-3;
  double offset_measurement_var = 0.25;
  double heading_measurement_var = 0.01;
  double curvature_measurement_var = 1e-4;
  double gate_sigma = 4.0;
};

// Tracks the three control quantities across frames.
class RoadGeometryFilter {
 public:
  explicit RoadGeometryFilter(FilterConfig cfg = {});

  void Reset();
  // Fold in one frame. Pass has_geometry = false when no ego lane was recovered, in
  // which case the estimate is the motion-model prediction.
  FilteredGeometry Update(bool has_geometry, double lateral_offset_m = 0.0,
                          double heading_error_rad = 0.0, double curvature_1pm = 0.0);

  int coasting_frames() const { return coasting_; }

 private:
  FilterConfig cfg_;
  ConstantVelocityFilter offset_, heading_, curvature_;
  int coasting_ = 0;
};

}  // namespace curvature_port

#endif  // CURVATURE_PORT_TEMPORAL_HPP
