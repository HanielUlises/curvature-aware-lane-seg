// Control quantities read off a ground-plane centreline.
//
// C++ port of src/geometry/road_geometry_portable.py, which is itself the portable
// specification of src/geometry/road_geometry.py (the training path reaches
// curvature through FITPACK; this one uses the portable cubic spline of
// curvature.hpp). This is the last geometry stage before the controller: it turns a
// centreline in ground metres into the lateral offset, heading error, and curvature
// a kinematic lateral MPC consumes.
//
// Ground convention: x lateral (right positive), z ahead. Curvature is signed
// right-positive, the negation of the counter-clockwise convention the curvature
// formula returns under this axis layout.
//
// See docs/geometry_port_spec.md, section 8.

#ifndef CURVATURE_PORT_ROAD_GEOMETRY_HPP
#define CURVATURE_PORT_ROAD_GEOMETRY_HPP

#include <vector>

#include "curvature_port/curvature.hpp"

namespace curvature_port {

// Depth span (metres) over which the near-field line is fitted.
constexpr double kNearFitSpanM = 12.0;
// Distance ahead (metres) at which lateral offset is reported. Nothing is observed
// at the vehicle plane itself, so a value quoted at z = 0 would be extrapolated.
constexpr double kDefaultOffsetDistanceM = 5.0;
constexpr int kDefaultNumSamples = 100;

struct RoadGeometry {
  double lateral_offset_m = 0.0;
  double offset_distance_m = kDefaultOffsetDistanceM;
  double heading_error_rad = 0.0;
  double curvature_1pm = 0.0;
  std::vector<double> preview_distances_m;
  // NaN where the preview distance falls outside the reconstructed centreline.
  std::vector<double> preview_curvature_1pm;
  // False when the centreline is too short or degenerate to read geometry from;
  // all other fields are then meaningless.
  bool valid = false;
};

// Least-squares line x = slope * z + intercept over the near span_m of the
// centreline. Offset and heading both come from this fit rather than from the two
// nearest points, whose angular noise would otherwise be extrapolated over the
// long lever arm back to the vehicle. Returns false if fewer than two usable
// points remain.
bool FitNearLine(const std::vector<Point>& ground, double span_m, double* intercept,
                 double* slope);

// Signed curvature along the portable cubic spline through points (counter-clockwise
// positive, i.e. before the road-geometry sign flip). Empty when undefined.
std::vector<double> SignedCurvatureAlong(const std::vector<Point>& points,
                                         int num_samples);

// The spline itself, evaluated on the same uniform parameter grid as
// SignedCurvatureAlong, so curvature can be interpolated against depth.
std::vector<Point> SamplePositions(const std::vector<Point>& points, int num_samples);

// Read offset, heading, and curvature off a ground centreline (x, z) in metres.
// The input need not be sorted; it is ordered near-to-far internally.
RoadGeometry ReadRoadGeometry(const std::vector<Point>& ground_centerline,
                              const std::vector<double>& preview_distances_m,
                              int num_samples = kDefaultNumSamples,
                              double offset_distance_m = kDefaultOffsetDistanceM);

}  // namespace curvature_port

#endif  // CURVATURE_PORT_ROAD_GEOMETRY_HPP
