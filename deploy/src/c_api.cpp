// C ABI for the deployed chain. See c_api.h.

#include "curvature_port/c_api.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "curvature_port/centerline.hpp"
#include "curvature_port/ipm.hpp"
#include "curvature_port/lane_tracker.hpp"
#include "curvature_port/mpc.hpp"
#include "curvature_port/road_geometry.hpp"
#include "curvature_port/temporal.hpp"

namespace {
constexpr int32_t kAbiVersion = 2;
}

// Holds every stage plus its scratch, so the per-frame call allocates nothing.
struct cp_chain {
  int width = 0;
  int height = 0;
  bool metric = false;
  curvature_port::GroundPlane ground;
  curvature_port::DecompositionScratch scratch;
  curvature_port::EgoBoundaryTracker tracker;
  curvature_port::RoadGeometryFilter filter;
  curvature_port::KinematicLateralMPC mpc;
  std::vector<curvature_port::Point> ground_pts;

  cp_chain(int w, int h) : width(w), height(h), tracker(w, h) {}
};

extern "C" {

int32_t cp_abi_version(void) { return kAbiVersion; }

cp_chain* cp_chain_create(int32_t width, int32_t height, const double* calibration) {
  if (width <= 0 || height <= 0) return nullptr;
  cp_chain* chain = new (std::nothrow) cp_chain(width, height);
  if (chain == nullptr) return nullptr;
  if (calibration != nullptr) {
    curvature_port::CameraCalibration calib;
    calib.fx = calibration[0];
    calib.fy = calibration[1];
    calib.cx = calibration[2];
    calib.cy = calibration[3];
    calib.height_m = calibration[4];
    calib.pitch_rad = calibration[5];
    calib.yaw_rad = calibration[6];
    chain->metric =
        curvature_port::GroundPlaneFromCalibration(calib, &chain->ground);
  }
  return chain;
}

void cp_chain_destroy(cp_chain* chain) { delete chain; }

void cp_chain_reset(cp_chain* chain) {
  if (chain == nullptr) return;
  chain->tracker = curvature_port::EgoBoundaryTracker(chain->width, chain->height);
  chain->filter.Reset();
}

void cp_chain_process(cp_chain* chain, const uint8_t* mask, double speed_mps,
                      cp_frame_result* out) {
  if (chain == nullptr || mask == nullptr || out == nullptr) return;
  std::memset(out, 0, sizeof(*out));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < 3; ++i) out->preview_curvature_1pm[i] = nan;

  const curvature_port::MaskView view{mask, chain->width, chain->height};
  const auto polylines =
      curvature_port::ExtractLanePolylines(view, &chain->scratch);
  const bool have_line = chain->tracker.Update(polylines);
  out->has_centerline = have_line ? 1 : 0;
  out->n_centerline = static_cast<int32_t>(chain->tracker.centerline().size());
  out->resets = chain->tracker.resets();

  bool have_geom = false;
  double off = 0.0, head = 0.0, kap = 0.0;
  if (have_line && chain->metric) {
    chain->ground_pts = chain->ground.ToGround(chain->tracker.centerline());
    const auto rg =
        curvature_port::ReadRoadGeometry(chain->ground_pts, {5.0, 10.0, 20.0});
    if (rg.valid) {
      have_geom = true;
      off = rg.lateral_offset_m;
      head = rg.heading_error_rad;
      kap = rg.curvature_1pm;
      out->raw_lateral_offset_m = off;
      out->raw_heading_error_rad = head;
      out->raw_curvature_1pm = kap;
      for (std::size_t i = 0; i < 3 && i < rg.preview_curvature_1pm.size(); ++i) {
        out->preview_curvature_1pm[i] = rg.preview_curvature_1pm[i];
      }
    }
  }
  out->has_geometry = have_geom ? 1 : 0;

  const auto filtered = chain->filter.Update(have_geom, off, head, kap);
  out->lateral_offset_m = filtered.lateral_offset_m;
  out->heading_error_rad = filtered.heading_error_rad;
  out->curvature_1pm = filtered.curvature_1pm;
  out->coasting_frames = filtered.coasting_frames;

  const auto sol = chain->mpc.SteerForGeometry(filtered.lateral_offset_m,
                                               filtered.heading_error_rad,
                                               filtered.curvature_1pm, speed_mps);
  if (sol.valid) {
    out->steer_rad = sol.steer_rad;
    out->steer_unsaturated_rad = sol.steer_unsaturated_rad;
    out->saturated = sol.saturated ? 1 : 0;
  }
}

int32_t cp_chain_centerline(const cp_chain* chain, double* out_xy, int32_t max_points) {
  if (chain == nullptr || out_xy == nullptr) return 0;
  const auto& c = chain->tracker.centerline();
  const int32_t n = static_cast<int32_t>(
      std::min<std::size_t>(c.size(), static_cast<std::size_t>(std::max(max_points, 0))));
  for (int32_t i = 0; i < n; ++i) {
    out_xy[2 * i] = c[static_cast<std::size_t>(i)].x;
    out_xy[2 * i + 1] = c[static_cast<std::size_t>(i)].y;
  }
  return n;
}

int32_t cp_chain_boundary(const cp_chain* chain, int32_t side, double* out_xy,
                          int32_t max_points) {
  if (chain == nullptr || out_xy == nullptr) return 0;
  const std::vector<double>& cols =
      (side == 0) ? chain->tracker.left_columns() : chain->tracker.right_columns();
  const std::vector<double>& rows = chain->tracker.rows();
  int32_t n = 0;
  for (std::size_t i = 0; i < cols.size() && n < max_points; ++i) {
    if (!std::isfinite(cols[i])) continue;
    out_xy[2 * n] = cols[i];
    out_xy[2 * n + 1] = rows[i];
    ++n;
  }
  return n;
}

int32_t cp_extract_polylines(const uint8_t* mask, int32_t width, int32_t height,
                             double* out_xy, int32_t* out_counts, int32_t max_polylines,
                             int32_t max_points) {
  if (mask == nullptr || out_xy == nullptr || out_counts == nullptr) return 0;
  // Stateless, so the scratch is local; callers wanting the allocation-free path use
  // cp_chain_process instead.
  curvature_port::DecompositionScratch scratch;
  const curvature_port::MaskView view{mask, width, height};
  const auto polys = curvature_port::ExtractLanePolylines(view, &scratch);

  int32_t written_points = 0, written_polys = 0;
  for (const auto& poly : polys) {
    if (written_polys >= max_polylines) break;
    if (written_points + static_cast<int32_t>(poly.size()) > max_points) break;
    for (const auto& p : poly) {
      out_xy[2 * written_points] = p.x;
      out_xy[2 * written_points + 1] = p.y;
      ++written_points;
    }
    out_counts[written_polys++] = static_cast<int32_t>(poly.size());
  }
  return written_polys;
}

}  // extern "C"
