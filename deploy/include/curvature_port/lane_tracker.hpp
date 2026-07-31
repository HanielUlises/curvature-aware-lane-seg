// Temporal tracking of the ego lane's two boundaries.
//
// C++ port of src/geometry/lane_tracker.py. Without it the ego pair is re-chosen every
// frame as whichever detections bracket the camera axis, and when the segmenter loses
// the ego lane's own marking the next marking out takes its place, moving the centreline
// sideways by half a lane. The tracker associates each boundary with the one it was
// already following, which is what makes the estimate stable enough to steer on.
//
// The whole tracker is fixed-size: the row grid never changes, so every buffer is
// allocated once at construction and the per-frame path does no allocation.
//
// See docs/geometry_port_spec.md, section 10.

#ifndef CURVATURE_PORT_LANE_TRACKER_HPP
#define CURVATURE_PORT_LANE_TRACKER_HPP

#include <vector>

#include "curvature_port/centerline.hpp"
#include "curvature_port/curvature.hpp"

namespace curvature_port {

struct TrackerConfig {
  int track_rows = 96;
  double alpha = 0.60;            // weight on the new observation
  double gate_px = 80.0;          // association gate
  int max_coast_frames = 1;       // frames a row may be carried unobserved
  int max_extend_rows = kMaxExtendRows;
  double min_width_px = 8.0;
  double max_lateral_slope = 4.0;
  int extent_median_frames = 3;
  int smooth_halfwidth = 2;
  double min_centerline_rows = 25.0;
  double width_tol_lo = 0.65;
  double width_tol_hi = 1.5;
  int num_centerline_points = 50;
};

class EgoBoundaryTracker {
 public:
  EgoBoundaryTracker(int image_width, int image_height, TrackerConfig cfg = {});

  // Fold one frame's polylines in. Returns false when no centreline could be formed.
  bool Update(const std::vector<Polyline>& polylines);

  // Valid after Update returned true. Points are ordered top to bottom.
  const std::vector<Point>& centerline() const { return centerline_; }
  const std::vector<double>& rows() const { return rows_; }
  // Tracked boundary columns on the row grid; NaN where the boundary is not held.
  const std::vector<double>& left_columns() const { return left_.columns; }
  const std::vector<double>& right_columns() const { return right_.columns; }
  int coasting_frames() const { return coasting_; }
  int resets() const { return left_.resets + right_.resets; }

 private:
  struct Track {
    std::vector<double> columns;
    std::vector<int> staleness;
    int resets = 0;
    bool alive = false;
  };

  // Median absolute distance between an observation and a track, or -1 if disjoint.
  double Displacement(const Track& track, const std::vector<double>& obs) const;
  void AdoptOrBlend(Track* track, const std::vector<double>* obs);
  bool Associate(const std::vector<Polyline>& polylines);

  int image_width_, image_height_;
  TrackerConfig cfg_;
  std::vector<double> rows_;
  Track left_, right_;
  std::vector<double> width_profile_;
  bool have_width_profile_ = false;
  std::vector<double> recent_bottoms_;
  int coasting_ = 0;
  std::vector<Point> centerline_;

  // Per-frame scratch, allocated once.
  std::vector<std::vector<double>> candidates_;
  std::vector<double> obs_left_, obs_right_;
  bool has_left_ = false, has_right_ = false;
  std::vector<double> centre_rows_, centre_cols_, xl_, xr_, scratch_;
  // Reused by the median helpers. Association compares every candidate against both
  // tracks, so a buffer allocated inside that comparison is allocated tens of times a
  // frame; these exist so the per-frame path allocates nothing.
  mutable std::vector<double> diffs_;
  std::vector<double> width_, ratios_, slopes_, bottoms_, medbuf_;
  std::vector<char> usable_;
  Polyline lp_, rp_;
};

}  // namespace curvature_port

#endif  // CURVATURE_PORT_LANE_TRACKER_HPP
