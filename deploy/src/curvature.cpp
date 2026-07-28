// Natural parametric cubic-spline curvature. See curvature.hpp and
// docs/geometry_port_spec.md for the contract this must satisfy.

#include "curvature_port/curvature.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

namespace curvature_port {
namespace {

constexpr double kSpeedEps = 1e-12;
constexpr int kMinPoints = 3;

// Drop consecutive duplicate points (zero-length segments break the fit).
std::vector<Point> DedupConsecutive(const std::vector<Point>& pts) {
  std::vector<Point> out;
  for (const Point& p : pts) {
    if (out.empty() || p.x != out.back().x || p.y != out.back().y) {
      out.push_back(p);
    }
  }
  return out;
}

// Cumulative arclength normalized to [0, 1].
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

// Second-derivative moments of the natural cubic spline through (u_i, f_i),
// with M_0 = M_{n-1} = 0. Solves the standard tridiagonal system with Eigen.
Eigen::VectorXd NaturalSplineMoments(const std::vector<double>& u,
                                     const std::vector<double>& f) {
  const int n = static_cast<int>(u.size());
  Eigen::VectorXd m = Eigen::VectorXd::Zero(n);
  const int interior = n - 2;
  if (interior <= 0) return m;

  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(interior, interior);
  Eigen::VectorXd d(interior);
  for (int k = 0; k < interior; ++k) {
    const int i = k + 1;  // global index of the interior knot
    const double h_prev = u[i] - u[i - 1];
    const double h_next = u[i + 1] - u[i];
    const double denom = h_prev + h_next;
    a(k, k) = 2.0;
    if (k > 0) a(k, k - 1) = h_prev / denom;             // mu_i
    if (k < interior - 1) a(k, k + 1) = h_next / denom;  // lambda_i
    d(k) = 6.0 / denom *
           ((f[i + 1] - f[i]) / h_next - (f[i] - f[i - 1]) / h_prev);
  }

  const Eigen::VectorXd sol = a.colPivHouseholderQr().solve(d);
  for (int k = 0; k < interior; ++k) m[k + 1] = sol(k);
  return m;
}

// First and second derivatives of a moment-form cubic segment at parameter `u`.
struct Deriv {
  double d1;
  double d2;
};

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

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double rank = p / 100.0 * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
  if (lo + 1 >= v.size()) return v.back();
  const double frac = rank - static_cast<double>(lo);
  return v[lo] + frac * (v[lo + 1] - v[lo]);
}

}  // namespace

std::vector<double> CurvatureAlong(const std::vector<Point>& points, int num_samples) {
  const std::vector<Point> pts = DedupConsecutive(points);
  const int n = static_cast<int>(pts.size());
  if (n < kMinPoints || num_samples < 1) return {};

  const std::vector<double> u = NormalizedArclength(pts);
  if (u.back() <= 0.0) return {};

  std::vector<double> fx(n), fy(n);
  for (int i = 0; i < n; ++i) {
    fx[i] = pts[i].x;
    fy[i] = pts[i].y;
  }
  const Eigen::VectorXd mx = NaturalSplineMoments(u, fx);
  const Eigen::VectorXd my = NaturalSplineMoments(u, fy);

  std::vector<double> kappa(num_samples);
  int seg = 0;
  for (int s = 0; s < num_samples; ++s) {
    const double uu = (num_samples == 1) ? 0.0
                                         : static_cast<double>(s) /
                                               static_cast<double>(num_samples - 1);
    while (seg < n - 2 && uu > u[seg + 1]) ++seg;

    const Deriv dx = EvalSegment(uu, u[seg], u[seg + 1], fx[seg], fx[seg + 1], mx[seg], mx[seg + 1]);
    const Deriv dy = EvalSegment(uu, u[seg], u[seg + 1], fy[seg], fy[seg + 1], my[seg], my[seg + 1]);

    const double numer = std::abs(dx.d1 * dy.d2 - dy.d1 * dx.d2);
    const double speed_sq = std::max(dx.d1 * dx.d1 + dy.d1 * dy.d1, kSpeedEps);
    kappa[s] = numer / std::pow(speed_sq, 1.5);
  }
  return kappa;
}

double LaneCurvature(const std::vector<Point>& points, double percentile, int num_samples) {
  const std::vector<double> kappa = CurvatureAlong(points, num_samples);
  if (kappa.empty()) return 0.0;
  return Percentile(kappa, percentile);
}

}  // namespace curvature_port
