// Lane-mask decomposition. See centerline.hpp.

#include "curvature_port/centerline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace curvature_port {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int Find(std::vector<int>& parent, int x) {
  // Path halving: keeps the union-find flat without recursion.
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void Union(std::vector<int>& parent, int a, int b) {
  a = Find(parent, a);
  b = Find(parent, b);
  if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

// Linear interpolation of x against y, NaN outside the sampled range. y ascending.
double InterpAt(const Polyline& poly, double y) {
  const std::size_t n = poly.size();
  if (n == 0 || y < poly.front().y || y > poly.back().y) return kNaN;
  const auto it = std::lower_bound(
      poly.begin(), poly.end(), y,
      [](const Point& p, double v) { return p.y < v; });
  if (it == poly.begin()) return poly.front().x;
  const std::size_t hi = static_cast<std::size_t>(it - poly.begin());
  const double span = poly[hi].y - poly[hi - 1].y;
  if (span <= 0.0) return poly[hi].x;
  const double t = (y - poly[hi - 1].y) / span;
  return poly[hi - 1].x + t * (poly[hi].x - poly[hi - 1].x);
}

// Least-squares line x = slope * y + intercept over a span of a polyline.
bool FitTail(const Polyline& poly, std::size_t lo, std::size_t hi, double* slope,
             double* intercept) {
  const std::size_t n = hi - lo;
  if (n < 2) return false;
  double sy = 0.0, sx = 0.0, syy = 0.0, syx = 0.0;
  double y_min = poly[lo].y, y_max = poly[lo].y;
  for (std::size_t i = lo; i < hi; ++i) {
    const double y = poly[i].y, x = poly[i].x;
    sy += y;
    sx += x;
    syy += y * y;
    syx += y * x;
    y_min = std::min(y_min, y);
    y_max = std::max(y_max, y);
  }
  if (y_max - y_min < 1e-9) return false;
  const double dn = static_cast<double>(n);
  const double denom = dn * syy - sy * sy;
  if (std::abs(denom) < 1e-12) return false;
  *slope = (dn * syx - sy * sx) / denom;
  *intercept = (sx - *slope * sy) / dn;
  return true;
}

}  // namespace

std::vector<Polyline> ExtractLanePolylines(const MaskView& mask,
                                           DecompositionScratch* scratch, int min_rows,
                                           int min_pixels) {
  const int w = mask.width, h = mask.height;
  std::vector<Polyline> out;
  if (w <= 0 || h <= 0 || mask.data == nullptr) return out;

  // 1. Find the horizontal runs of foreground, row by row. This is the only pass over
  //    the image, and it skips background a byte at a time.
  scratch->runs.clear();
  scratch->row_begin.assign(static_cast<std::size_t>(h) + 1, 0);
  for (int y = 0; y < h; ++y) {
    scratch->row_begin[static_cast<std::size_t>(y)] =
        static_cast<int>(scratch->runs.size());
    const std::uint8_t* row = mask.data + static_cast<std::size_t>(y) * w;
    int x = 0;
    while (x < w) {
      // Most of a lane mask is background, so step over it a word at a time. The load
      // goes through memcpy so it stays defined on unaligned addresses and compiles to
      // the same single instruction.
      while (x + 8 <= w) {
        std::uint64_t chunk;
        std::memcpy(&chunk, row + x, sizeof(chunk));
        if (chunk != 0) break;
        x += 8;
      }
      while (x < w && row[x] == 0) ++x;
      if (x >= w) break;
      const int x0 = x;
      while (x < w && row[x] != 0) ++x;
      scratch->runs.push_back({y, x0, x, static_cast<int>(scratch->runs.size())});
    }
  }
  scratch->row_begin[static_cast<std::size_t>(h)] =
      static_cast<int>(scratch->runs.size());
  const int n_runs = static_cast<int>(scratch->runs.size());
  if (n_runs == 0) return out;

  // 2. Union runs on adjacent rows that touch. Two runs are 8-connected when their
  //    column spans overlap after widening one of them by a pixel, which for half-open
  //    spans is a.x0 <= b.x1 && b.x0 <= a.x1. Both rows are sorted by column, so this
  //    is a two-pointer merge rather than a search.
  scratch->parent.resize(static_cast<std::size_t>(n_runs));
  for (int i = 0; i < n_runs; ++i) scratch->parent[static_cast<std::size_t>(i)] = i;
  for (int y = 0; y + 1 < h; ++y) {
    int a = scratch->row_begin[static_cast<std::size_t>(y)];
    const int a_end = scratch->row_begin[static_cast<std::size_t>(y) + 1];
    int b = a_end;
    const int b_end = scratch->row_begin[static_cast<std::size_t>(y) + 2];
    while (a < a_end && b < b_end) {
      const Run& ra = scratch->runs[static_cast<std::size_t>(a)];
      const Run& rb = scratch->runs[static_cast<std::size_t>(b)];
      if (ra.x0 <= rb.x1 && rb.x0 <= ra.x1) Union(scratch->parent, a, b);
      if (ra.x1 < rb.x1) ++a; else ++b;
    }
  }

  // 3. Per-component pixel count and row extent, over runs.
  scratch->pixel_count.assign(static_cast<std::size_t>(n_runs), 0);
  scratch->row_lo.assign(static_cast<std::size_t>(n_runs), h);
  scratch->row_hi.assign(static_cast<std::size_t>(n_runs), -1);
  for (int i = 0; i < n_runs; ++i) {
    const Run& r = scratch->runs[static_cast<std::size_t>(i)];
    const int root = Find(scratch->parent, i);
    scratch->pixel_count[static_cast<std::size_t>(root)] += r.x1 - r.x0;
    scratch->row_lo[static_cast<std::size_t>(root)] =
        std::min(scratch->row_lo[static_cast<std::size_t>(root)], r.y);
    scratch->row_hi[static_cast<std::size_t>(root)] =
        std::max(scratch->row_hi[static_cast<std::size_t>(root)], r.y);
  }

  scratch->root_index.assign(static_cast<std::size_t>(n_runs), -1);
  int kept = 0;
  for (int i = 0; i < n_runs; ++i) {
    if (Find(scratch->parent, i) != i) continue;  // not a root
    if (scratch->pixel_count[static_cast<std::size_t>(i)] < min_pixels) continue;
    if (scratch->row_hi[static_cast<std::size_t>(i)] -
            scratch->row_lo[static_cast<std::size_t>(i)] + 1 < min_rows) continue;
    scratch->root_index[static_cast<std::size_t>(i)] = kept++;
  }
  if (kept == 0) return out;

  // 4. Row-wise centroid, accumulated from runs. The column sum of a run [x0, x1) is
  //    the closed form below rather than a loop over its pixels.
  const std::size_t stride = static_cast<std::size_t>(h);
  scratch->col_sum.assign(static_cast<std::size_t>(kept) * stride, 0.0);
  scratch->col_count.assign(static_cast<std::size_t>(kept) * stride, 0.0);
  for (int i = 0; i < n_runs; ++i) {
    const Run& r = scratch->runs[static_cast<std::size_t>(i)];
    const int k = scratch->root_index[
        static_cast<std::size_t>(Find(scratch->parent, i))];
    if (k < 0) continue;
    const double len = r.x1 - r.x0;
    const std::size_t idx = static_cast<std::size_t>(k) * stride +
                            static_cast<std::size_t>(r.y);
    scratch->col_sum[idx] += 0.5 * (r.x0 + r.x1 - 1) * len;
    scratch->col_count[idx] += len;
  }

  out.resize(static_cast<std::size_t>(kept));
  for (int k = 0; k < kept; ++k) {
    Polyline& poly = out[static_cast<std::size_t>(k)];
    poly.reserve(static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      const std::size_t idx = static_cast<std::size_t>(k) * stride +
                              static_cast<std::size_t>(y);
      const double c = scratch->col_count[idx];
      if (c > 0.0) poly.push_back({scratch->col_sum[idx] / c, static_cast<double>(y)});
    }
  }

  // Sorted left to right by the bottom-most point's column, as the reference does.
  std::stable_sort(out.begin(), out.end(), [](const Polyline& a, const Polyline& b) {
    return a.back().x < b.back().x;
  });
  return out;
}

bool EgoLanePair(const std::vector<Polyline>& polylines, int image_width, int* left,
                 int* right) {
  if (polylines.size() < 2) return false;
  const double centre = image_width / 2.0;
  int li = -1, ri = -1;
  double best_l = -std::numeric_limits<double>::infinity();
  double best_r = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < polylines.size(); ++i) {
    const double x = polylines[i].back().x;  // bottom-most point
    if (x < centre) {
      if (x > best_l) { best_l = x; li = static_cast<int>(i); }
    } else {
      if (x < best_r) { best_r = x; ri = static_cast<int>(i); }
    }
  }
  if (li < 0 || ri < 0) return false;
  *left = li;
  *right = ri;
  return true;
}

void ResampleBoundary(const Polyline& poly, const std::vector<double>& rows,
                      std::vector<double>* out, int max_extend_rows,
                      int fit_tail_rows) {
  out->assign(rows.size(), kNaN);
  if (poly.size() < 2) return;

  const double y_lo = poly.front().y, y_hi = poly.back().y;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    (*out)[i] = InterpAt(poly, rows[i]);
  }
  if (max_extend_rows <= 0) return;

  // Continue past each end along a line fitted to that end's tail.
  double slope = 0.0, intercept = 0.0;
  std::size_t tail = 0;
  while (tail < poly.size() && poly[tail].y <= y_lo + fit_tail_rows) ++tail;
  if (FitTail(poly, 0, tail, &slope, &intercept)) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i] < y_lo && std::abs(rows[i] - y_lo) <= max_extend_rows) {
        (*out)[i] = slope * rows[i] + intercept;
      }
    }
  }
  std::size_t head = poly.size();
  while (head > 0 && poly[head - 1].y >= y_hi - fit_tail_rows) --head;
  if (FitTail(poly, head, poly.size(), &slope, &intercept)) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i] > y_hi && std::abs(rows[i] - y_hi) <= max_extend_rows) {
        (*out)[i] = slope * rows[i] + intercept;
      }
    }
  }
}

std::vector<Point> EgoCenterline(const std::vector<Polyline>& polylines,
                                 int image_width, int image_height, int num_points,
                                 int max_extend_rows) {
  std::vector<Point> centre;
  int li = 0, ri = 0;
  if (!EgoLanePair(polylines, image_width, &li, &ri)) return centre;
  const Polyline& left = polylines[static_cast<std::size_t>(li)];
  const Polyline& right = polylines[static_cast<std::size_t>(ri)];

  const double top = std::max(left.front().y, right.front().y);
  const double observed_bottom = std::min(left.back().y, right.back().y);
  double target = observed_bottom + max_extend_rows;
  if (image_height > 0) target = std::min(target, static_cast<double>(image_height - 1));
  if (target <= top) return centre;

  std::vector<double> rows(static_cast<std::size_t>(num_points));
  const double step = (target - top) / std::max(num_points - 1, 1);
  for (int i = 0; i < num_points; ++i) {
    rows[static_cast<std::size_t>(i)] = top + step * i;
  }
  rows[static_cast<std::size_t>(num_points - 1)] = target;
  std::vector<double> xl, xr;
  ResampleBoundary(left, rows, &xl, max_extend_rows);
  ResampleBoundary(right, rows, &xr, max_extend_rows);

  centre.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const double mid = 0.5 * (xl[i] + xr[i]);
    if (std::isfinite(mid)) centre.push_back({mid, rows[i]});
  }
  if (centre.size() < 3) centre.clear();
  return centre;
}

}  // namespace curvature_port
