// Natural parametric cubic spline over normalized arclength — the shared kernel
// behind the curvature estimator and the road-geometry read-out.
//
// Internal to the library (not installed): deployment code includes
// curvature.hpp or road_geometry.hpp. The mathematics is specified in
// docs/geometry_port_spec.md section 3.3 and mirrored by
// src/geometry/curvature_portable.py.

#ifndef CURVATURE_PORT_SPLINE_INTERNAL_HPP
#define CURVATURE_PORT_SPLINE_INTERNAL_HPP

#include <Eigen/Dense>

#include <vector>

#include "curvature_port/curvature.hpp"

namespace curvature_port {
namespace internal {

constexpr double kSpeedEps = 1e-12;
constexpr int kMinPoints = 3;

// Drop consecutive coincident points (zero-length segments break the fit).
std::vector<Point> DedupConsecutive(const std::vector<Point>& pts);

// Cumulative arclength normalized to [0, 1].
std::vector<double> NormalizedArclength(const std::vector<Point>& pts);

// Second-derivative moments of the cubic spline through (u_i, f_i), under
// not-a-knot end conditions (natural for the three-point case, where not-a-knot is
// undefined). The natural condition M_0 = M_{n-1} = 0 was dropped because it forces
// curvature to zero at the polyline ends, and the controller reads curvature at a
// 5 m look-ahead that sits near the near end of a recovered centreline: on a 50 m
// arc it returned 0.0032 1/m there against a true 0.02.
Eigen::VectorXd SplineMoments(const std::vector<double>& u,
                              const std::vector<double>& f);

// Value and derivatives of a moment-form cubic segment at parameter u.
struct Deriv {
  double d1;
  double d2;
};

Deriv EvalSegment(double u, double u_lo, double u_hi, double f_lo, double f_hi,
                  double m_lo, double m_hi);

double EvalSegmentValue(double u, double u_lo, double u_hi, double f_lo, double f_hi,
                        double m_lo, double m_hi);

// The s-th of num_samples uniform parameters on [0, 1].
double UniformParameter(int s, int num_samples);

// Linear-method percentile of v (copied and sorted in place). 0 when empty.
double Percentile(std::vector<double> v, double p);

// A polyline prepared for evaluation: deduplicated points, their normalized
// arclength parameter, and the per-coordinate moments.
struct Spline {
  std::vector<Point> pts;
  std::vector<double> u;
  Eigen::VectorXd mx;
  Eigen::VectorXd my;
  bool valid = false;
};

Spline BuildSpline(const std::vector<Point>& points);

}  // namespace internal
}  // namespace curvature_port

#endif  // CURVATURE_PORT_SPLINE_INTERNAL_HPP
