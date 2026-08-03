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

/* ---------------------------------------------------------------------------
 * The segmenter, and the whole pipeline with it.
 *
 * Everything above takes a mask the caller produced, which for a long time meant
 * PyTorch: the chain ran in C++ but the stage that costs 98% of the frame ran in
 * Python, so the deployed pipeline was not actually deployed. These entry points run
 * the network here, through ONNX Runtime, and cp_pipeline_process takes a frame all
 * the way to a steering command without the caller touching a pixel.
 *
 * They are only present when the library was built with a segmenter; the rest of the
 * ABI does not depend on it. Call cp_segmenter_available first — a build without it
 * still exports these symbols, and they fail cleanly.
 */

/* Whether this build can run a network at all. */
int32_t cp_segmenter_available(void);

typedef struct {
  const char* model_path;       /* the .onnx written by scripts/export_onnx.py */
  int32_t width;                /* network input size, e.g. 512 x 288 */
  int32_t height;
  const char* backend;          /* "auto", "tensorrt", "cuda", or "cpu"; NULL = auto */
  int32_t fp16;                 /* build the TensorRT engine in half precision */
  const char* engine_cache_dir; /* NULL or "" to rebuild the engine every start */
  double threshold;             /* lane probability, 0.5 unless there is a reason */
  double sky_frac;              /* must match what the model was trained under */
} cp_segmenter_config;

typedef struct cp_segmenter cp_segmenter;

/* Load a model. Returns NULL on failure and writes the reason into err, which may be
 * NULL. The reason matters here: a missing TensorRT library and a graph TensorRT will
 * not take are different problems with different fixes. */
cp_segmenter* cp_segmenter_create(const cp_segmenter_config* config, char* err,
                                  int32_t err_len);
void cp_segmenter_destroy(cp_segmenter* segmenter);

/* Preprocess src, run the network, and write width*height bytes into out_mask.
 * src is 8-bit, 3-channel, row-major with src_stride bytes per row (0 = packed);
 * set bgr for the channel order OpenCV hands back, clear it for RGB.
 * Returns 1 on success. */
int32_t cp_segmenter_run(cp_segmenter* segmenter, const uint8_t* src, int32_t src_width,
                         int32_t src_height, int32_t src_stride, int32_t bgr,
                         uint8_t* out_mask);

/* The backend that actually built the session, which for "auto" is only known after
 * the fact. Never NULL. */
const char* cp_segmenter_backend(const cp_segmenter* segmenter);

/* Microseconds in the last run, split at the session boundary. */
double cp_segmenter_last_preprocess_us(const cp_segmenter* segmenter);
double cp_segmenter_last_network_us(const cp_segmenter* segmenter);

typedef struct cp_pipeline cp_pipeline;

/* Segmenter and chain together. calibration is the same seven doubles cp_chain_create
 * takes, or NULL for centreline-only operation. */
cp_pipeline* cp_pipeline_create(const cp_segmenter_config* config,
                                const double* calibration, char* err, int32_t err_len);
void cp_pipeline_destroy(cp_pipeline* pipeline);
void cp_pipeline_reset(cp_pipeline* pipeline);

/* One frame, image to steering command. Arguments as cp_segmenter_run.
 * Returns 1 if the network ran; check the result's flags for what came out of it. */
int32_t cp_pipeline_process(cp_pipeline* pipeline, const uint8_t* src, int32_t src_width,
                            int32_t src_height, int32_t src_stride, int32_t bgr,
                            double speed_mps, cp_frame_result* out);

int32_t cp_pipeline_centerline(const cp_pipeline* pipeline, double* out_xy,
                               int32_t max_points);
int32_t cp_pipeline_boundary(const cp_pipeline* pipeline, int32_t side, double* out_xy,
                             int32_t max_points);

/* The mask from the last frame, width*height bytes owned by the pipeline and valid
 * until the next call. For drawing an overlay without recomputing anything. */
const uint8_t* cp_pipeline_mask(const cp_pipeline* pipeline);

/* The region of the source frame the network input was taken from, as x, y, width,
 * height, so an overlay can be put back where it came from. */
void cp_pipeline_source_roi(const cp_pipeline* pipeline, int32_t* out_xywh);

/* Microseconds in the last frame, split three ways. Preprocess and network are the
 * segmenter's; chain is everything after the mask. */
double cp_pipeline_last_preprocess_us(const cp_pipeline* pipeline);
double cp_pipeline_last_network_us(const cp_pipeline* pipeline);
double cp_pipeline_last_chain_us(const cp_pipeline* pipeline);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CURVATURE_PORT_C_API_H */
