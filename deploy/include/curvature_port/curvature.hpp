// Curvature estimation for the real-time deployment path.
//
// C++ port of src/geometry/curvature.py. The Python reference fits a B-spline
// with scipy/FITPACK, which has no drop-in C++ equivalent; this port uses a
// natural parametric cubic spline instead. It is therefore validated *not* by
// matching FITPACK internals but against the shared golden vectors: on curves
// with a closed-form answer (line, circle) both implementations must recover the
// true geometric curvature, and on arbitrary polylines the port must agree with
// the Python reference within the tolerance recorded per case.
//
// See docs/geometry_port_spec.md for the full numerical contract.

#ifndef CURVATURE_PORT_CURVATURE_HPP
#define CURVATURE_PORT_CURVATURE_HPP

#include <vector>

namespace curvature_port {

struct Point {
  double x;
  double y;
};

// Sample |kappa| along a natural parametric cubic spline through points,
// parameterized by normalized arclength, at num_samples uniform locations.
// Returns an empty vector when curvature is undefined (< 3 unique points).
std::vector<double> CurvatureAlong(const std::vector<Point>& points, int num_samples);

// Scalar lane-curvature summary: the percentile of |kappa| along the lane.
// Mirrors src.geometry.curvature.lane_curvature. Returns 0 when undefined.
double LaneCurvature(const std::vector<Point>& points, double percentile, int num_samples);

}  // namespace curvature_port

#endif  // CURVATURE_PORT_CURVATURE_HPP
