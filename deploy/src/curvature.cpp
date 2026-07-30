// Natural parametric cubic-spline curvature. See curvature.hpp and
// docs/geometry_port_spec.md for the contract this must satisfy. The spline
// kernel itself lives in spline_internal.hpp, shared with road_geometry.cpp.

#include "curvature_port/curvature.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

#include "spline_internal.hpp"

namespace curvature_port {

std::vector<double> CurvatureAlong(const std::vector<Point>& points, int num_samples) {
  const internal::Spline s = internal::BuildSpline(points);
  if (!s.valid || num_samples < 1) return {};

  const int n = static_cast<int>(s.pts.size());
  std::vector<double> kappa(num_samples);
  int seg = 0;
  for (int i = 0; i < num_samples; ++i) {
    const double uu = internal::UniformParameter(i, num_samples);
    while (seg < n - 2 && uu > s.u[seg + 1]) ++seg;

    const internal::Deriv dx = internal::EvalSegment(
        uu, s.u[seg], s.u[seg + 1], s.pts[seg].x, s.pts[seg + 1].x, s.mx[seg], s.mx[seg + 1]);
    const internal::Deriv dy = internal::EvalSegment(
        uu, s.u[seg], s.u[seg + 1], s.pts[seg].y, s.pts[seg + 1].y, s.my[seg], s.my[seg + 1]);

    const double numer = std::abs(dx.d1 * dy.d2 - dy.d1 * dx.d2);
    const double speed_sq =
        std::max(dx.d1 * dx.d1 + dy.d1 * dy.d1, internal::kSpeedEps);
    kappa[i] = numer / std::pow(speed_sq, 1.5);
  }
  return kappa;
}

double LaneCurvature(const std::vector<Point>& points, double percentile, int num_samples) {
  const std::vector<double> kappa = CurvatureAlong(points, num_samples);
  if (kappa.empty()) return 0.0;
  return internal::Percentile(kappa, percentile);
}

}  // namespace curvature_port
