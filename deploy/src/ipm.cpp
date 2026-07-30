// Direct-linear-transform homography and the image-to-ground mapping.
// See ipm.hpp and docs/geometry_port_spec.md.

#include "curvature_port/ipm.hpp"

#include <cmath>

namespace curvature_port {
namespace {

constexpr double kHomogeneousEps = 1e-12;
constexpr int kMinCorrespondences = 4;

}  // namespace

Matrix3 HomographyFromPoints(const std::vector<Point>& src,
                             const std::vector<Point>& dst, bool* ok) {
  if (ok != nullptr) *ok = false;
  if (src.size() < kMinCorrespondences || dst.size() != src.size()) {
    return Matrix3::Identity();
  }

  const int rows = 2 * static_cast<int>(src.size());
  Eigen::MatrixXd a(rows, 9);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const double x = src[i].x, y = src[i].y;
    const double u = dst[i].x, v = dst[i].y;
    a.row(2 * static_cast<int>(i)) << -x, -y, -1, 0, 0, 0, u * x, u * y, u;
    a.row(2 * static_cast<int>(i) + 1) << 0, 0, 0, -x, -y, -1, v * x, v * y, v;
  }

  // The homography is the right singular vector of the smallest singular value.
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  const Eigen::VectorXd null_vec = svd.matrixV().col(8);

  Matrix3 h;
  h << null_vec(0), null_vec(1), null_vec(2),
       null_vec(3), null_vec(4), null_vec(5),
       null_vec(6), null_vec(7), null_vec(8);
  // The null vector is defined up to scale (and sign); pinning H(2,2) removes both.
  if (std::abs(h(2, 2)) > kHomogeneousEps) h /= h(2, 2);
  if (ok != nullptr) *ok = true;
  return h;
}

std::vector<Point> ApplyHomography(const Matrix3& h, const std::vector<Point>& points) {
  std::vector<Point> out;
  out.reserve(points.size());
  for (const Point& p : points) {
    const Eigen::Vector3d mapped = h * Eigen::Vector3d(p.x, p.y, 1.0);
    double w = mapped(2);
    if (std::abs(w) < kHomogeneousEps) w = kHomogeneousEps;
    out.push_back({mapped(0) / w, mapped(1) / w});
  }
  return out;
}

GroundPlane::GroundPlane(const Matrix3& h) : h_(h), h_inv_(h.inverse()) {}

std::vector<Point> GroundPlane::ToGround(const std::vector<Point>& image_points) const {
  return ApplyHomography(h_, image_points);
}

std::vector<Point> GroundPlane::ToImage(const std::vector<Point>& ground_points) const {
  return ApplyHomography(h_inv_, ground_points);
}

}  // namespace curvature_port
