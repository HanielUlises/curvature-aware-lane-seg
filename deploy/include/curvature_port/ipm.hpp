// Inverse-perspective mapping for the real-time deployment path.
//
// C++ port of src/geometry/ipm.py. Unlike the curvature estimator, this stage has
// no library gap to bridge: the Direct Linear Transform is a small SVD, identical
// mathematics on both sides, so the port is expected to agree with the Python
// reference to floating-point precision on the golden vectors.
//
// See docs/geometry_port_spec.md, section 7.

#ifndef CURVATURE_PORT_IPM_HPP
#define CURVATURE_PORT_IPM_HPP

#include <Eigen/Dense>

#include <vector>

#include "curvature_port/curvature.hpp"

namespace curvature_port {

using Matrix3 = Eigen::Matrix3d;

// Solve the 3x3 homography mapping src to dst by DLT, normalized so H(2,2) == 1.
// Needs at least four paired correspondences; returns an identity matrix and sets
// ok to false otherwise.
Matrix3 HomographyFromPoints(const std::vector<Point>& src,
                             const std::vector<Point>& dst, bool* ok = nullptr);

// Apply a homography to points, dividing out the homogeneous coordinate with a
// floor of 1e-12 so points at the horizon do not produce infinities.
std::vector<Point> ApplyHomography(const Matrix3& h, const std::vector<Point>& points);

// A configured image-to-ground mapping: image pixels (u, v) to ground metres
// (x lateral, z ahead).
class GroundPlane {
 public:
  GroundPlane() = default;
  explicit GroundPlane(const Matrix3& h);

  const Matrix3& h() const { return h_; }
  const Matrix3& h_inv() const { return h_inv_; }

  std::vector<Point> ToGround(const std::vector<Point>& image_points) const;
  std::vector<Point> ToImage(const std::vector<Point>& ground_points) const;

 private:
  Matrix3 h_ = Matrix3::Identity();
  Matrix3 h_inv_ = Matrix3::Identity();
};

}  // namespace curvature_port

#endif  // CURVATURE_PORT_IPM_HPP
