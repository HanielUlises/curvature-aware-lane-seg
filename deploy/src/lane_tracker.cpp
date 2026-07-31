// Temporal tracking of the ego lane's two boundaries. See lane_tracker.hpp.

#include "curvature_port/lane_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace curvature_port {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Median of the finite entries of `values`, which is reordered in place.
double MedianInPlace(std::vector<double>* values) {
  const auto end = std::partition(values->begin(), values->end(),
                                  [](double v) { return std::isfinite(v); });
  const std::size_t n = static_cast<std::size_t>(end - values->begin());
  if (n == 0) return kNaN;
  const std::size_t mid = n / 2;
  std::nth_element(values->begin(), values->begin() + mid, end);
  const double hi = (*values)[mid];
  if (n % 2 == 1) return hi;
  std::nth_element(values->begin(), values->begin() + mid - 1, end);
  return 0.5 * (hi + (*values)[mid - 1]);
}

// Moving average along a run of finite values, edges replicated. Mirrors the reference.
void SmoothColumns(std::vector<double>* cols, int half, std::vector<double>* scratch) {
  if (half < 1) return;
  const int width = 2 * half + 1;
  const std::size_t n = cols->size();
  *scratch = *cols;
  std::size_t start = 0;
  bool in_run = false;
  for (std::size_t i = 0; i <= n; ++i) {
    const bool inside = i < n && std::isfinite((*cols)[i]);
    if (inside && !in_run) {
      start = i;
      in_run = true;
    } else if (!inside && in_run) {
      const std::size_t len = i - start;
      if (len >= static_cast<std::size_t>(width)) {
        for (std::size_t k = start; k < i; ++k) {
          double sum = 0.0;
          for (int d = -half; d <= half; ++d) {
            const long idx = static_cast<long>(k) + d;
            const std::size_t c = static_cast<std::size_t>(
                std::min<long>(std::max<long>(idx, static_cast<long>(start)),
                               static_cast<long>(i) - 1));
            sum += (*cols)[c];
          }
          (*scratch)[k] = sum / width;
        }
      }
      in_run = false;
    }
  }
  cols->swap(*scratch);
}

}  // namespace

EgoBoundaryTracker::EgoBoundaryTracker(int image_width, int image_height,
                                       TrackerConfig cfg)
    : image_width_(image_width), image_height_(image_height), cfg_(cfg) {
  const int n = std::max(cfg_.track_rows, 2);
  rows_.resize(static_cast<std::size_t>(n));
  // As numpy.linspace builds it: step first, endpoint pinned. Computing
  // (height - 1) * i / (n - 1) instead differs in the last bits, and those bits reach
  // the extent cutoff, where they decide whether a row is inside or outside it.
  const double row_step = static_cast<double>(image_height - 1) / (n - 1);
  for (int i = 0; i < n; ++i) rows_[static_cast<std::size_t>(i)] = row_step * i;
  rows_[static_cast<std::size_t>(n - 1)] = static_cast<double>(image_height - 1);
  left_.columns.assign(rows_.size(), kNaN);
  right_.columns.assign(rows_.size(), kNaN);
  left_.staleness.assign(rows_.size(), 0);
  right_.staleness.assign(rows_.size(), 0);
  width_profile_.assign(rows_.size(), kNaN);
  obs_left_.assign(rows_.size(), kNaN);
  obs_right_.assign(rows_.size(), kNaN);
  scratch_.assign(rows_.size(), kNaN);
  recent_bottoms_.reserve(8);
  centre_rows_.reserve(static_cast<std::size_t>(cfg_.num_centerline_points));
  centre_cols_.reserve(static_cast<std::size_t>(cfg_.num_centerline_points));
  centerline_.reserve(static_cast<std::size_t>(cfg_.num_centerline_points));
}

double EgoBoundaryTracker::Displacement(const Track& track,
                                        const std::vector<double>& obs) const {
  diffs_.clear();
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (std::isfinite(obs[i]) && std::isfinite(track.columns[i])) {
      diffs_.push_back(std::abs(obs[i] - track.columns[i]));
    }
  }
  if (diffs_.empty()) return -1.0;
  return MedianInPlace(&diffs_);
}

void EgoBoundaryTracker::AdoptOrBlend(Track* track, const std::vector<double>* obs) {
  const std::size_t n = rows_.size();
  if (!track->alive) {
    if (obs == nullptr) return;
    track->columns = *obs;
    std::fill(track->staleness.begin(), track->staleness.end(), 0);
    track->alive = true;
    return;
  }
  if (obs != nullptr) {
    const double gap = Displacement(*track, *obs);
    if (gap >= 0.0 && gap > cfg_.gate_px) {
      track->columns = *obs;
      std::fill(track->staleness.begin(), track->staleness.end(), 0);
      ++track->resets;
      return;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const bool seen = obs != nullptr && std::isfinite((*obs)[i]);
    const bool held = std::isfinite(track->columns[i]);
    if (seen && held) {
      track->columns[i] =
          (1.0 - cfg_.alpha) * track->columns[i] + cfg_.alpha * (*obs)[i];
    } else if (seen) {
      track->columns[i] = (*obs)[i];
    }
    track->staleness[i] = seen ? 0 : track->staleness[i] + 1;
    if (track->staleness[i] > cfg_.max_coast_frames) track->columns[i] = kNaN;
  }
  SmoothColumns(&track->columns, cfg_.smooth_halfwidth, &scratch_);
  track->alive = std::any_of(track->columns.begin(), track->columns.end(),
                             [](double v) { return std::isfinite(v); });
}

bool EgoBoundaryTracker::Associate(const std::vector<Polyline>& polylines) {
  candidates_.resize(polylines.size());
  for (std::size_t i = 0; i < polylines.size(); ++i) {
    ResampleBoundary(polylines[i], rows_, &candidates_[i], cfg_.max_extend_rows);
  }

  int fb_l = -1, fb_r = -1;
  const bool have_fallback = EgoLanePair(polylines, image_width_, &fb_l, &fb_r);

  auto best = [&](const Track& track, int* chosen, double* dist) {
    *chosen = -1;
    *dist = std::numeric_limits<double>::infinity();
    if (!track.alive) return;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const double d = Displacement(track, candidates_[i]);
      if (d >= 0.0 && d <= cfg_.gate_px && d < *dist) {
        *dist = d;
        *chosen = static_cast<int>(i);
      }
    }
  };

  int li = -1, ri = -1;
  double ld = 0.0, rd = 0.0;
  best(left_, &li, &ld);
  best(right_, &ri, &rd);
  if (li >= 0 && li == ri) {
    if (ld <= rd) ri = -1; else li = -1;
  }
  if (li < 0 && have_fallback) li = fb_l;
  if (ri < 0 && have_fallback) ri = fb_r;

  has_left_ = li >= 0;
  has_right_ = ri >= 0;
  if (has_left_) obs_left_ = candidates_[static_cast<std::size_t>(li)];
  if (has_right_) obs_right_ = candidates_[static_cast<std::size_t>(ri)];
  if (!has_left_ || !has_right_) return has_left_ || has_right_;

  // The pair has to stay a pair: a gate alone lets the right track match the left
  // boundary when the two are close.
  medbuf_ = obs_left_;
  const double ml = MedianInPlace(&medbuf_);
  medbuf_ = obs_right_;
  const double mr = MedianInPlace(&medbuf_);
  if (!(std::isfinite(ml) && std::isfinite(mr)) || ml >= mr) {
    if (!have_fallback) {
      has_left_ = has_right_ = false;
      return false;
    }
    obs_left_ = candidates_[static_cast<std::size_t>(fb_l)];
    obs_right_ = candidates_[static_cast<std::size_t>(fb_r)];
    ld = rd = std::numeric_limits<double>::infinity();
  }

  // Width plausibility, compared row by row: lane width in pixels grows steeply
  // towards the vehicle, so a scalar width says nothing.
  width_.assign(rows_.size(), kNaN);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (std::isfinite(obs_left_[i]) && std::isfinite(obs_right_[i])) {
      width_[i] = obs_right_[i] - obs_left_[i];
    }
  }
  medbuf_ = width_;
  const double med_w = MedianInPlace(&medbuf_);
  if (!std::isfinite(med_w) || med_w <= cfg_.min_width_px) {
    has_left_ = has_right_ = false;
    return false;
  }
  if (have_width_profile_) {
    ratios_.clear();
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      if (std::isfinite(width_[i]) && std::isfinite(width_profile_[i]) &&
          std::abs(width_profile_[i]) > 1e-9) {
        ratios_.push_back(width_[i] / width_profile_[i]);
      }
    }
    if (ratios_.size() >= 3) {
      const double ratio = MedianInPlace(&ratios_);
      if (!(ratio >= cfg_.width_tol_lo && ratio <= cfg_.width_tol_hi)) {
        if (ld <= rd) has_right_ = false; else has_left_ = false;
        return true;
      }
    }
  }
  width_profile_ = width_;
  have_width_profile_ = true;
  return true;
}

bool EgoBoundaryTracker::Update(const std::vector<Polyline>& polylines) {
  centerline_.clear();
  const bool observed = Associate(polylines);
  coasting_ = observed ? 0 : coasting_ + 1;

  AdoptOrBlend(&left_, has_left_ ? &obs_left_ : nullptr);
  AdoptOrBlend(&right_, has_right_ ? &obs_right_ : nullptr);
  if (!left_.alive || !right_.alive) return false;

  // The rows both tracks hold.
  double top = std::numeric_limits<double>::infinity();
  double bottom = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (std::isfinite(left_.columns[i])) {
      top = std::min(top, rows_[i]);
      bottom = std::max(bottom, rows_[i]);
    }
  }
  double rtop = std::numeric_limits<double>::infinity();
  double rbot = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (std::isfinite(right_.columns[i])) {
      rtop = std::min(rtop, rows_[i]);
      rbot = std::max(rbot, rows_[i]);
    }
  }
  top = std::max(top, rtop);
  bottom = std::min(bottom, rbot);
  if (!(bottom > top)) return false;

  const int n = cfg_.num_centerline_points;
  centre_rows_.resize(static_cast<std::size_t>(n));
  // Built the way numpy.linspace builds it, step first and the last point pinned to the
  // endpoint. Computing top + (bottom - top) * i / (n - 1) instead rounds differently in
  // the last bit, which is enough to move a row across the extent cutoff below and drop
  // a point the reference keeps.
  const double step = (bottom - top) / std::max(n - 1, 1);
  for (int i = 0; i < n; ++i) {
    centre_rows_[static_cast<std::size_t>(i)] = top + step * i;
  }
  centre_rows_[static_cast<std::size_t>(n - 1)] = bottom;

  // Sample the tracks themselves; they were already extended once when observed.
  lp_.clear();
  rp_.clear();
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (std::isfinite(left_.columns[i])) lp_.push_back({left_.columns[i], rows_[i]});
    if (std::isfinite(right_.columns[i])) rp_.push_back({right_.columns[i], rows_[i]});
  }
  ResampleBoundary(lp_, centre_rows_, &xl_, 0);
  ResampleBoundary(rp_, centre_rows_, &xr_, 0);

  centre_cols_.assign(static_cast<std::size_t>(n), kNaN);
  usable_.assign(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    const double mid = 0.5 * (xl_[k] + xr_[k]);
    centre_cols_[k] = mid;
    usable_[k] = (std::isfinite(mid) && (xr_[k] - xl_[k]) > cfg_.min_width_px) ? 1 : 0;
  }

  // Sideways step per row, computed over every row rather than over the usable ones:
  // where the centreline has a hole, the row after it has no defined step, and the
  // reference treats that as infinite and drops the row. Skipping holes instead and
  // calling the next step zero keeps a row the reference discards, which diverged on
  // one frame in 1200 before this was matched.
  const double row_step = std::max(centre_rows_[1] - centre_rows_[0], 1e-6);
  slopes_.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = 1; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    const double step = std::abs(centre_cols_[k] - centre_cols_[k - 1]);
    slopes_[k] = std::isfinite(step) ? step / row_step
                                     : std::numeric_limits<double>::infinity();
  }
  if (!std::isfinite(centre_cols_[0])) {
    slopes_[0] = std::numeric_limits<double>::infinity();
  }

  medbuf_.clear();
  for (int i = 0; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (std::isfinite(slopes_[k]) && std::isfinite(centre_cols_[k])) {
      medbuf_.push_back(slopes_[k]);
    }
  }
  const double typical = medbuf_.empty() ? 0.0 : MedianInPlace(&medbuf_);
  const double limit = std::max(cfg_.max_lateral_slope, 3.0 * typical);
  for (int i = 0; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (!(slopes_[k] <= limit)) usable_[k] = 0;
  }

  // Longest contiguous run: rows retire independently, and drawing the islands as one
  // line renders it as disconnected fragments.
  int best_start = 0, best_len = 0, run_start = -1;
  for (int i = 0; i <= n; ++i) {
    const bool ok = i < n && usable_[static_cast<std::size_t>(i)];
    if (ok && run_start < 0) run_start = i;
    else if (!ok && run_start >= 0) {
      if (i - run_start > best_len) { best_len = i - run_start; best_start = run_start; }
      run_start = -1;
    }
  }
  if (best_len < 3) return false;

  // Extent as a median over recent frames, clipped to what this frame supports.
  const double available = centre_rows_[static_cast<std::size_t>(best_start + best_len - 1)];
  recent_bottoms_.push_back(available);
  if (static_cast<int>(recent_bottoms_.size()) > cfg_.extent_median_frames) {
    recent_bottoms_.erase(recent_bottoms_.begin());
  }
  medbuf_ = recent_bottoms_;
  const double target = std::min(MedianInPlace(&medbuf_), available);

  // Apply the extent cap only if it leaves a usable line. When the median lands before
  // the run even starts, which happens after a frame whose geometry sat much closer to
  // the vehicle, capping would erase the line entirely; the reference keeps the
  // uncapped run in that case, and dropping this condition cost four frames in 1200.
  // The comparison is exact. A tolerance was tried here to absorb the row grid being
  // built slightly differently from numpy's, but once the grid was built the same way
  // the tolerance only served to admit rows the reference excludes, which is how two
  // frames in 1200 gained a point they should not have.
  int capped = 0;
  for (int i = best_start; i < best_start + best_len; ++i) {
    if (centre_rows_[static_cast<std::size_t>(i)] <= target) ++capped;
  }
  const bool apply_cap = capped >= 3;

  centerline_.clear();
  for (int i = best_start; i < best_start + best_len; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (apply_cap && centre_rows_[k] > target) continue;
    centerline_.push_back({centre_cols_[k], centre_rows_[k]});
  }
  if (centerline_.size() < 3 ||
      centerline_.back().y - centerline_.front().y < cfg_.min_centerline_rows) {
    centerline_.clear();
    return false;
  }
  return true;
}

}  // namespace curvature_port
