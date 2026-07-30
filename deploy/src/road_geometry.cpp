// Offset, heading, and curvature from a ground-plane centreline.
// See road_geometry.hpp and docs/geometry_port_spec.md section 8.

#include "curvature_port/road_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "spline_internal.hpp"

namespace curvature_port {
namespace {

// Linear interpolation of kappa against depth, NaN outside the sampled range.
// Mirrors numpy.interp with left=right=nan; xs must be ascending.
double InterpAt(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (xs.empty() || x < xs.front() || x > xs.back()) return nan;
  const auto it = std::lower_bound(xs.begin(), xs.end(), x);
  if (it == xs.begin()) return ys.front();
  const std::size_t hi = static_cast<std::size_t>(it - xs.begin());
  const double span = xs[hi] - xs[hi - 1];
  if (span <= 0.0) return ys[hi];
  const double t = (x - xs[hi - 1]) / span;
  return ys[hi - 1] + t * (ys[hi] - ys[hi - 1]);
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t mid = v.size() / 2;
  if (v.size() % 2 == 1) return v[mid];
  return 0.5 * (v[mid - 1] + v[mid]);
}

}  // namespace

bool FitNearLine(const std::vector<Point>& ground, double span_m, double* intercept,
                 double* slope) {
  if (ground.size() < 2) return false;

  // The centreline is ordered near-to-far, so the near span is a prefix.
  std::vector<Point> near;
  for (const Point& p : ground) {
    if (p.y <= ground.front().y + span_m) near.push_back(p);
  }
  if (near.size() < 3) {
    near.assign(ground.begin(),
                ground.begin() + static_cast<long>(std::min<std::size_t>(3, ground.size())));
  }
  if (near.size() < 2) return false;

  double z_min = near.front().y, z_max = near.front().y;
  for (const Point& p : near) {
    z_min = std::min(z_min, p.y);
    z_max = std::max(z_max, p.y);
  }
  if (z_max - z_min < 1e-9) return false;  // no depth span to regress against

  // Ordinary least squares of x on z.
  const double n = static_cast<double>(near.size());
  double sum_z = 0.0, sum_x = 0.0, sum_zz = 0.0, sum_zx = 0.0;
  for (const Point& p : near) {
    sum_z += p.y;
    sum_x += p.x;
    sum_zz += p.y * p.y;
    sum_zx += p.y * p.x;
  }
  const double denom = n * sum_zz - sum_z * sum_z;
  if (std::abs(denom) < 1e-12) return false;
  const double a = (n * sum_zx - sum_z * sum_x) / denom;
  if (slope != nullptr) *slope = a;
  if (intercept != nullptr) *intercept = (sum_x - a * sum_z) / n;
  return true;
}

std::vector<double> SignedCurvatureAlong(const std::vector<Point>& points,
                                         int num_samples) {
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

    const double speed_sq =
        std::max(dx.d1 * dx.d1 + dy.d1 * dy.d1, internal::kSpeedEps);
    kappa[i] = (dx.d1 * dy.d2 - dy.d1 * dx.d2) / std::pow(speed_sq, 1.5);
  }
  return kappa;
}

std::vector<Point> SamplePositions(const std::vector<Point>& points, int num_samples) {
  const internal::Spline s = internal::BuildSpline(points);
  if (!s.valid || num_samples < 1) return {};

  const int n = static_cast<int>(s.pts.size());
  std::vector<Point> out(static_cast<std::size_t>(num_samples));
  int seg = 0;
  for (int i = 0; i < num_samples; ++i) {
    const double uu = internal::UniformParameter(i, num_samples);
    while (seg < n - 2 && uu > s.u[seg + 1]) ++seg;
    out[static_cast<std::size_t>(i)] = {
        internal::EvalSegmentValue(uu, s.u[seg], s.u[seg + 1], s.pts[seg].x,
                                   s.pts[seg + 1].x, s.mx[seg], s.mx[seg + 1]),
        internal::EvalSegmentValue(uu, s.u[seg], s.u[seg + 1], s.pts[seg].y,
                                   s.pts[seg + 1].y, s.my[seg], s.my[seg + 1])};
  }
  return out;
}

RoadGeometry ReadRoadGeometry(const std::vector<Point>& ground_centerline,
                              const std::vector<double>& preview_distances_m,
                              int num_samples, double offset_distance_m) {
  RoadGeometry out;
  out.offset_distance_m = offset_distance_m;
  out.preview_distances_m = preview_distances_m;
  out.preview_curvature_1pm.assign(preview_distances_m.size(),
                                   std::numeric_limits<double>::quiet_NaN());
  if (ground_centerline.size() < 3) return out;

  std::vector<Point> ground = ground_centerline;
  std::stable_sort(ground.begin(), ground.end(),
                   [](const Point& a, const Point& b) { return a.y < b.y; });

  double intercept = 0.0, slope = 0.0;
  if (!FitNearLine(ground, kNearFitSpanM, &intercept, &slope)) return out;
  out.lateral_offset_m = intercept + slope * offset_distance_m;
  out.heading_error_rad = std::atan(slope);
  out.valid = true;

  // Negate the counter-clockwise convention so that right turns are positive.
  std::vector<double> kappa = SignedCurvatureAlong(ground, num_samples);
  for (double& k : kappa) k = -k;
  const std::vector<Point> positions = SamplePositions(ground, num_samples);
  if (kappa.empty() || positions.size() != kappa.size()) return out;

  out.curvature_1pm = Median(kappa);

  // Curvature is indexed by spline parameter, so interpolate it against the depth
  // of the same samples to answer "curvature at z metres ahead".
  std::vector<std::size_t> order(kappa.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return positions[a].y < positions[b].y;
  });
  std::vector<double> z_sorted(kappa.size()), k_sorted(kappa.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    z_sorted[i] = positions[order[i]].y;
    k_sorted[i] = kappa[order[i]];
  }
  for (std::size_t i = 0; i < preview_distances_m.size(); ++i) {
    out.preview_curvature_1pm[i] = InterpAt(z_sorted, k_sorted, preview_distances_m[i]);
  }
  return out;
}

}  // namespace curvature_port
