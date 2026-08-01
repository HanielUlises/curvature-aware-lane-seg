/* C ABI for the deployed chain, so Python (or anything else) can call it without a
 * binding framework.
 *
 * The Python package used to carry a second implementation of every stage. That was
 * never redundancy for its own sake: the two implementations are what the golden-vector
 * contract compares. But having both *run* meant the fast path and the reference path
 * could silently diverge, and the one that shipped was the slow one.
 *
 * With this shim the split is explicit. The C++ is the only implementation that runs;
 * the Python implementations stay as the reference the fixtures are generated from and
 * the port is checked against, which is a test role rather than a runtime one.
 *
 * A C ABI rather than pybind11 on purpose: it needs no build-time dependency beyond the
 * compiler already required, and ctypes ships with Python. The cost is that every type
 * crossing the boundary has to be plain data, which is why the geometry comes back as a
 * flat struct and the polylines are copied into caller-owned buffers.
 */

#ifndef CURVATURE_PORT_C_API_H
#define CURVATURE_PORT_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One frame's output. Fields are only meaningful where the corresponding flag is set. */
typedef struct {
  int32_t has_centerline;  /* the tracker produced an ego centreline */
  int32_t has_geometry;    /* and it projected to a usable road geometry */
  int32_t n_centerline;    /* points available via cp_chain_centerline */
  double lateral_offset_m; /* filtered */
  double heading_error_rad;
  double curvature_1pm;
  /* The same three before filtering. A supervisor wants both: the filtered signal is
   * what steers, the raw one is what tells you whether the filter is coasting through
   * noise or through nothing. Only meaningful when has_geometry is set. */
  double raw_lateral_offset_m;
  double raw_heading_error_rad;
  double raw_curvature_1pm;
  double preview_curvature_1pm[3]; /* at 5, 10, 20 m; NaN outside the centreline */
  double steer_rad;                /* saturated command, right-positive */
  double steer_unsaturated_rad;
  int32_t saturated;
  int32_t coasting_frames;
  int32_t resets;   /* cumulative tracker association resets */
} cp_frame_result;

typedef struct cp_chain cp_chain;

/* Create a chain for a given frame size. Pass calibration = NULL to run without a ground
 * plane, in which case only the centreline is produced and the metric fields stay zero.
 * calibration is seven doubles: fx, fy, cx, cy, height_m, pitch_rad, yaw_rad. */
cp_chain* cp_chain_create(int32_t width, int32_t height, const double* calibration);
void cp_chain_destroy(cp_chain* chain);

/* Reset all temporal state: tracker, filter, and the extent history. */
void cp_chain_reset(cp_chain* chain);

/* Process one binary mask, row-major, width*height bytes, non-zero meaning lane. */
void cp_chain_process(cp_chain* chain, const uint8_t* mask, double speed_mps,
                      cp_frame_result* out);

/* Copy the current centreline into out_xy as interleaved x, y image pixels. Returns the
 * number of points written, at most max_points. */
int32_t cp_chain_centerline(const cp_chain* chain, double* out_xy, int32_t max_points);

/* Copy the tracked ego boundaries, same interleaving. side 0 is left, 1 is right. */
int32_t cp_chain_boundary(const cp_chain* chain, int32_t side, double* out_xy,
                          int32_t max_points);

/* Decompose a mask into lane polylines without touching any temporal state. Writes
 * interleaved x, y into out_xy and the per-polyline point counts into out_counts.
 * Returns the number of polylines. */
int32_t cp_extract_polylines(const uint8_t* mask, int32_t width, int32_t height,
                             double* out_xy, int32_t* out_counts, int32_t max_polylines,
                             int32_t max_points);

/* Version of the shim, so a Python wrapper can refuse a stale shared library. */
int32_t cp_abi_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CURVATURE_PORT_C_API_H */
