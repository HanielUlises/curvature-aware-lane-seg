// Lane-mask decomposition for the real-time deployment path.
//
// C++ port of src/geometry/centerline.py: the segmenter's binary mask becomes one
// ordered polyline per lane, and the two polylines bracketing the camera axis become
// the ego centreline. This is the first stage of the chain and the only one that
// touches the mask, so it dominates the per-frame cost; it is written to run over the
// mask once, without allocating per component.
//
// Connected components are labelled with a two-pass union-find rather than by calling
// OpenCV, which the deployment target does not carry. Labels need not match OpenCV's
// numbering: only the set of components matters, and the polylines are ordered
// afterwards by their bottom-most column exactly as the reference does.
//
// See docs/geometry_port_spec.md, section 9.

#ifndef CURVATURE_PORT_CENTERLINE_HPP
#define CURVATURE_PORT_CENTERLINE_HPP

#include <cstdint>
#include <vector>

#include "curvature_port/curvature.hpp"

namespace curvature_port {

// A component must span at least this many rows to be a lane rather than a speck...
constexpr int kMinLaneRows = 8;
// ...and carry at least this many foreground pixels.
constexpr int kMinLanePixels = 40;
// How far a boundary may be extended past its observed extent, in image rows.
constexpr int kMaxExtendRows = 45;
// Rows at the end of a boundary used to fit the direction it is extended along.
constexpr int kFitTailRows = 25;

// A binary lane mask. Non-zero is lane. Row-major, `width` values per row.
struct MaskView {
  const std::uint8_t* data;
  int width;
  int height;
};

// One lane, as image points ordered by increasing row.
using Polyline = std::vector<Point>;

// Reduce a mask to one polyline per lane: each connected component collapsed to its
// row-wise centroid. Sorted left to right by the bottom-most point's column.
//
// Components are found over horizontal runs rather than over pixels. A lane mask is a
// few per cent foreground, so there are orders of magnitude fewer runs than pixels, and
// the row-wise centroid a lane polyline needs is a closed form over a run rather than a
// sum over its pixels. The only pass over the full image is the byte scan that finds the
// runs. Labelling pixel by pixel measured 335 us per 512x288 frame against 25 us for the
// whole rest of the chain, which made it the entire cost of the port.
//
// `scratch` is reused between calls so the per-frame path allocates nothing; pass the
// same instance every frame.
struct Run {
  int y;
  int x0;   // inclusive
  int x1;   // exclusive
  int label;
};

struct DecompositionScratch {
  std::vector<Run> runs;
  std::vector<int> row_begin;   // index of the first run of each row
  std::vector<int> parent;      // union-find over runs
  std::vector<int> root_index;  // compact index per root, -1 when dropped
  std::vector<double> col_sum;
  std::vector<double> col_count;
  std::vector<int> row_lo, row_hi, pixel_count;
};

std::vector<Polyline> ExtractLanePolylines(const MaskView& mask,
                                           DecompositionScratch* scratch,
                                           int min_rows = kMinLaneRows,
                                           int min_pixels = kMinLanePixels);

// Indices into `polylines` of the two lanes bracketing the camera axis, nearest first
// on each side. Returns false when the axis is not bracketed.
bool EgoLanePair(const std::vector<Polyline>& polylines, int image_width, int* left,
                 int* right);

// Sample a boundary at the given image rows, extending a bounded amount past its
// observed span along a line fitted to that end. Rows past the bound are NaN.
void ResampleBoundary(const Polyline& poly, const std::vector<double>& rows,
                      std::vector<double>* out, int max_extend_rows = kMaxExtendRows,
                      int fit_tail_rows = kFitTailRows);

// Centreline midway between the two bracketing lanes, or empty when there is none.
std::vector<Point> EgoCenterline(const std::vector<Polyline>& polylines,
                                 int image_width, int image_height,
                                 int num_points = 50,
                                 int max_extend_rows = kMaxExtendRows);

}  // namespace curvature_port

#endif  // CURVATURE_PORT_CENTERLINE_HPP
