// Natural parametric cubic spline kernel. See spline_internal.hpp.

#include "spline_internal.hpp"

#include <algorithm>
#include <cmath>

namespace curvature_port {
namespace internal {

std::vector<Point> DedupConsecutive(const std::vector<Point>& pts) {
  std::vector<Point> out;
  for (const Point& p : pts) {
    if (out.empty() || p.x != out.back().x || p.y != out.back().y) {
      out.push_back(p);
    }
  }
  return out;
}

std::vector<double> NormalizedArclength(const std::vector<Point>& pts) {
  std::vector<double> u(pts.size(), 0.0);
  for (std::size_t i = 1; i < pts.size(); ++i) {
    const double dx = pts[i].x - pts[i - 1].x;
    const double dy = pts[i].y - pts[i - 1].y;
    u[i] = u[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  const double total = u.back();
  if (total > 0.0) {
    for (double& v : u) v /= total;
  }
  return u;
}

Eigen::VectorXd SplineMoments(const std::vector<double>& u,
                              const std::vector<double>& f) {
  const int n = static_cast<int>(u.size());
  if (n < kMinPoints) return Eigen::VectorXd::Zero(n);

  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
  for (int i = 1; i < n - 1; ++i) {
    const double h_prev = u[i] - u[i - 1];
    const double h_next = u[i + 1] - u[i];
    a(i, i - 1) = h_prev;
    a(i, i) = 2.0 * (h_prev + h_next);
    a(i, i + 1) = h_next;
    d(i) = 6.0 * ((f[i + 1] - f[i]) / h_next - (f[i] - f[i - 1]) / h_prev);
  }

  if (n >= 4) {
    // Not-a-knot: continuous third derivative across the first and last interior
    // knots. See spline_internal.hpp for why the natural condition was dropped.
    const double h0 = u[1] - u[0], h1 = u[2] - u[1];
    a(0, 0) = h1;
    a(0, 1) = -(h0 + h1);
    a(0, 2) = h0;
    const double hl = u[n - 2] - u[n - 3], hr = u[n - 1] - u[n - 2];
    a(n - 1, n - 3) = hr;
    a(n - 1, n - 2) = -(hl + hr);
    a(n - 1, n - 1) = hl;
  } else {
    // One interior knot leaves not-a-knot undefined; fall back to natural.
    a(0, 0) = 1.0;
    a(n - 1, n - 1) = 1.0;
  }

  return a.colPivHouseholderQr().solve(d);
}

Deriv EvalSegment(double u, double u_lo, double u_hi, double f_lo, double f_hi,
                  double m_lo, double m_hi) {
  const double h = u_hi - u_lo;
  const double a = (u_hi - u) / h;  // weight on the low knot
  const double b = (u - u_lo) / h;  // weight on the high knot
  const double d1 = (f_hi - f_lo) / h +
                    h / 6.0 * ((3.0 * b * b - 1.0) * m_hi - (3.0 * a * a - 1.0) * m_lo);
  const double d2 = a * m_lo + b * m_hi;
  return {d1, d2};
}

double EvalSegmentValue(double u, double u_lo, double u_hi, double f_lo, double f_hi,
                        double m_lo, double m_hi) {
  // Unlike EvalSegment, which uses normalized local coordinates, the value form
  // takes them unnormalized; dividing by h here silently rescales the curve.
  const double h = u_hi - u_lo;
  const double a = u_hi - u;
  const double b = u - u_lo;
  return (m_lo * a * a * a + m_hi * b * b * b) / (6.0 * h) +
         (f_lo / h - m_lo * h / 6.0) * a + (f_hi / h - m_hi * h / 6.0) * b;
}

double UniformParameter(int s, int num_samples) {
  if (num_samples == 1) return 0.0;
  return static_cast<double>(s) / static_cast<double>(num_samples - 1);
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double rank = p / 100.0 * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
  if (lo + 1 >= v.size()) return v.back();
  const double frac = rank - static_cast<double>(lo);
  return v[lo] + frac * (v[lo + 1] - v[lo]);
}

Spline BuildSpline(const std::vector<Point>& points) {
  Spline s;
  s.pts = DedupConsecutive(points);
  const int n = static_cast<int>(s.pts.size());
  if (n < kMinPoints) return s;

  s.u = NormalizedArclength(s.pts);
  if (s.u.back() <= 0.0) return s;

  std::vector<double> fx(n), fy(n);
  for (int i = 0; i < n; ++i) {
    fx[i] = s.pts[i].x;
    fy[i] = s.pts[i].y;
  }
  s.mx = SplineMoments(s.u, fx);
  s.my = SplineMoments(s.u, fy);
  s.valid = true;
  return s;
}

}  // namespace internal
}  // namespace curvature_port
