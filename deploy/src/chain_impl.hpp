// The chain's internals, shared between the two C ABI translation units.
//
// c_api.cpp exposes the chain on its own, taking a mask the caller produced.
// c_api_infer.cpp exposes the same chain with the segmenter in front of it, so a frame
// goes image to steering in one call. Both need the state and the per-frame logic, and
// only one of them is always compiled, so it lives here rather than in either.

#ifndef CURVATURE_PORT_CHAIN_IMPL_HPP
#define CURVATURE_PORT_CHAIN_IMPL_HPP

#include <cstdint>
#include <vector>

#include "curvature_port/c_api.h"
#include "curvature_port/centerline.hpp"
#include "curvature_port/ipm.hpp"
#include "curvature_port/lane_tracker.hpp"
#include "curvature_port/mpc.hpp"
#include "curvature_port/road_geometry.hpp"
#include "curvature_port/temporal.hpp"

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

namespace curvature_port_detail {

// Fill a chain's calibration from the seven-double array the ABI passes. Returns
// whether a usable ground plane came out of it.
bool ApplyCalibration(cp_chain* chain, const double* calibration);

// One frame, from mask to steering command. The whole body of cp_chain_process, split
// out so the fused image-to-steering entry point runs the identical code rather than a
// copy of it.
void ProcessMask(cp_chain* chain, const std::uint8_t* mask, double speed_mps,
                 cp_frame_result* out);

int32_t CopyCenterline(const cp_chain* chain, double* out_xy, int32_t max_points);
int32_t CopyBoundary(const cp_chain* chain, int32_t side, double* out_xy,
                     int32_t max_points);

}  // namespace curvature_port_detail

#endif  // CURVATURE_PORT_CHAIN_IMPL_HPP
