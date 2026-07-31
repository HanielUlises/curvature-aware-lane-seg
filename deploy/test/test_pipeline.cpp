// Runs the full mask-to-centreline chain over a recorded sequence of real predicted
// masks and checks it against what the Python reference produced from the same masks
// (scripts/export_pipeline_vectors.py). Also times it: this is the stage that runs
// per frame on the vehicle, so its cost is part of the contract.

#include "curvature_port/centerline.hpp"
#include "curvature_port/lane_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace {

struct Frame {
  int index = 0;
  int num_polylines = 0;
  std::vector<std::uint8_t> mask;
  std::vector<curvature_port::Point> centerline;
};

struct Sequence {
  int width = 0, height = 0;
  double tolerance = 0.5;
  std::vector<Frame> frames;
};

bool LoadSequence(const std::string& path, Sequence* seq) {
  std::ifstream in(path);
  if (!in) return false;
  std::string key;
  int count = 0;
  Frame* current = nullptr;
  while (in >> key) {
    if (key == "width") in >> seq->width;
    else if (key == "height") in >> seq->height;
    else if (key == "tolerance") in >> seq->tolerance;
    else if (key == "frames") in >> count;
    else if (key == "frame") {
      seq->frames.emplace_back();
      current = &seq->frames.back();
      std::string tmp;
      in >> current->index >> tmp >> current->num_polylines;
      current->mask.assign(
          static_cast<std::size_t>(seq->width) * seq->height, 0);
    } else if (key == "rle") {
      int n = 0;
      in >> n;
      for (int i = 0; i < n; i += 2) {
        long start = 0, len = 0;
        in >> start >> len;
        std::fill_n(current->mask.begin() + start, len, static_cast<std::uint8_t>(255));
      }
    } else if (key == "centerline") {
      int n = 0;
      in >> n;
      current->centerline.resize(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) {
        in >> current->centerline[static_cast<std::size_t>(i)].x
           >> current->centerline[static_cast<std::size_t>(i)].y;
      }
    }
  }
  return static_cast<int>(seq->frames.size()) == count && count > 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Defaults to the committed fixture; a path argument runs the same checks over a
  // longer sequence, which is how throughput is measured on a real video without
  // carrying its masks in the repository.
  const std::string path = (argc > 1) ? argv[1]
                                      : std::string(GOLDEN_DIR) + "/pipeline/sequence.txt";
  Sequence seq;
  if (!LoadSequence(path, &seq)) {
    std::cerr << "cannot read " << path
              << " (run: python -m scripts.export_pipeline_vectors "
                 "infer.source=<clip-dir>)\n";
    return 2;
  }

  curvature_port::EgoBoundaryTracker tracker(seq.width, seq.height);
  curvature_port::DecompositionScratch scratch;

  int failures = 0, compared = 0, detected = 0, poly_mismatch = 0;
  double worst = 0.0;
  double total_us = 0.0, worst_us = 0.0, mask_us = 0.0, track_us = 0.0;

  for (const Frame& f : seq.frames) {
    const curvature_port::MaskView view{f.mask.data(), seq.width, seq.height};

    const auto t0 = std::chrono::steady_clock::now();
    const auto polylines =
        curvature_port::ExtractLanePolylines(view, &scratch);
    const auto t1 = std::chrono::steady_clock::now();
    if (getenv("DBG")) fprintf(stderr, "== frame %d (polys %zu)\n", f.index,
                               polylines.size());
    const bool ok = tracker.Update(polylines);
    const auto t2 = std::chrono::steady_clock::now();
    const double us_mask =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double us_track =
        std::chrono::duration<double, std::micro>(t2 - t1).count();
    mask_us += us_mask;
    track_us += us_track;
    const double us = us_mask + us_track;
    total_us += us;
    worst_us = std::max(worst_us, us);

    if (static_cast<int>(polylines.size()) != f.num_polylines) ++poly_mismatch;

    const bool want = !f.centerline.empty();
    if (ok != want) {
      ++failures;
      int nl = 0, nr = 0;
      for (double v : tracker.left_columns()) nl += std::isfinite(v) ? 1 : 0;
      for (double v : tracker.right_columns()) nr += std::isfinite(v) ? 1 : 0;
      std::cout << "[FAIL] frame " << f.index << ": centreline "
                << (ok ? "produced" : "absent") << ", reference "
                << (want ? "produced" : "absent")
                << " | tracked rows L=" << nl << " R=" << nr
                << " coasting=" << tracker.coasting_frames()
                << " resets=" << tracker.resets()
                << " reference points=" << f.centerline.size() << "\n";
      continue;
    }
    if (!want) continue;
    ++detected;

    const auto& got = tracker.centerline();
    if (got.size() != f.centerline.size()) {
      ++failures;
      std::cout << "[FAIL] frame " << f.index << ": " << got.size()
                << " points against " << f.centerline.size() << "\n";
      continue;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double dx = std::abs(got[i].x - f.centerline[i].x);
      const double dy = std::abs(got[i].y - f.centerline[i].y);
      const double d = std::max(dx, dy);
      worst = std::max(worst, d);
      ++compared;
      if (d > seq.tolerance) {
        ++failures;
        std::cout << "[FAIL] frame " << f.index << " point " << i << ": got ("
                  << got[i].x << ", " << got[i].y << ") want ("
                  << f.centerline[i].x << ", " << f.centerline[i].y << ")\n";
        break;
      }
    }
  }

  const double n = static_cast<double>(seq.frames.size());
  std::cout << "\n" << seq.frames.size() << " frames, " << detected
            << " with a centreline, " << compared << " points compared\n"
            << "worst point disagreement " << worst << " px (tolerance "
            << seq.tolerance << ")\n"
            << "polyline-count mismatches: " << poly_mismatch << "\n"
            << "mask decomposition " << (mask_us / n) << " us/frame, tracking "
            << (track_us / n) << " us/frame, total " << (total_us / n)
            << " us/frame mean, " << worst_us << " us worst\n";
  if (failures == 0) std::cout << "all frames match the reference\n";
  return failures == 0 ? 0 : 1;
}
