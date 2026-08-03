#include "curvature_port/preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace curvature_port {
namespace {

// Python's round() breaks ties to even, and the crop geometry is derived with it. A
// naive round-half-up would put the region one pixel off on any frame whose dimensions
// land exactly on a half, which is not rare at the resolutions dashcams use. nearbyint
// under the default rounding mode is the same rule.
int RoundHalfEven(double value) {
  return static_cast<int>(std::nearbyint(value));
}

}  // namespace

SourceRegion NetworkInputRegion(int src_width, int src_height, int target_width,
                                int target_height, double sky_frac) {
  if (src_width <= 0 || src_height <= 0 || target_width <= 0 || target_height <= 0 ||
      !(sky_frac >= 0.0) || sky_frac >= 1.0) {
    return SourceRegion{};
  }
  const double target_ratio =
      static_cast<double>(target_width) / static_cast<double>(target_height);

  // Stage one: centre-crop the native frame to the target aspect, no scaling. The
  // tolerance matches the reference: a frame already at the target aspect is left
  // alone rather than cropped by a pixel of rounding.
  int x = 0, y = 0, w = src_width, h = src_height;
  const double native_ratio = static_cast<double>(w) / static_cast<double>(h);
  if (std::abs(native_ratio - target_ratio) >= 1e-3) {
    if (native_ratio > target_ratio) {
      const int new_w = RoundHalfEven(h * target_ratio);
      x = (w - new_w) / 2;
      w = new_w;
    } else {
      const int new_h = RoundHalfEven(w / target_ratio);
      y = (h - new_h) / 2;
      h = new_h;
    }
  }

  // Stage two: drop the sky, then restore the target aspect the sky crop just broke.
  const int top = std::min(RoundHalfEven(h * sky_frac), h - 1);
  y += top;
  h -= top;

  const double cropped_ratio = static_cast<double>(w) / static_cast<double>(h);
  if (cropped_ratio > target_ratio) {
    // Too wide now: symmetric width crop.
    const int new_w = RoundHalfEven(h * target_ratio);
    x += (w - new_w) / 2;
    w = new_w;
  } else if (cropped_ratio < target_ratio) {
    // Too tall: take from the top, keeping the road at the bottom.
    const int new_h = RoundHalfEven(w / target_ratio);
    y += h - new_h;
    h = new_h;
  }
  return SourceRegion{x, y, w, h};
}

}  // namespace curvature_port
