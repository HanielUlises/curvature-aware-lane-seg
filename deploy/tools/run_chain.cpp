// Runs the deployed chain over a recorded sequence and reports what a vehicle would
// care about: the command, and how long producing it took.
//
//   mask -> lane polylines -> tracked ego boundaries -> centreline
//        -> ground projection -> offset / heading / curvature
//        -> temporal filter -> lateral MPC -> steering angle
//
// No Python, no OpenCV, no allocation on the per-frame path once the buffers are warm.
// Emits CSV on stdout so a run can be diffed against the Python reference, and a summary
// on stderr with latency percentiles, since a mean latency hides exactly the tail that
// makes a control loop miss its deadline.
//
//   run_chain <sequence.txt> [calibration.txt] [speed_mps]
//
// sequence.txt is the run-length-encoded mask fixture written by
// scripts/export_pipeline_vectors.py. calibration.txt is seven whitespace-separated
// numbers: fx fy cx cy height_m pitch_rad yaw_rad. Without it the chain still runs and
// reports the image-space geometry, but the metric outputs are meaningless, so it says so.

#include "curvature_port/centerline.hpp"
#include "curvature_port/ipm.hpp"
#include "curvature_port/lane_tracker.hpp"
#include "curvature_port/mpc.hpp"
#include "curvature_port/road_geometry.hpp"
#include "curvature_port/temporal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Frame {
  int index = 0;
  std::vector<std::uint8_t> mask;
};

bool LoadSequence(const std::string& path, int* width, int* height,
                  std::vector<Frame>* frames) {
  std::ifstream in(path);
  if (!in) return false;
  std::string key;
  Frame* cur = nullptr;
  while (in >> key) {
    if (key == "width") in >> *width;
    else if (key == "height") in >> *height;
    else if (key == "frame") {
      frames->emplace_back();
      cur = &frames->back();
      std::string tmp;
      int n;
      in >> cur->index >> tmp >> n;
      cur->mask.assign(static_cast<std::size_t>(*width) * *height, 0);
    } else if (key == "rle") {
      int n = 0;
      in >> n;
      for (int i = 0; i < n; i += 2) {
        long start = 0, len = 0;
        in >> start >> len;
        std::fill_n(cur->mask.begin() + start, len, static_cast<std::uint8_t>(255));
      }
    } else if (key == "centerline") {
      int n = 0;
      in >> n;
      double x, y;
      for (int i = 0; i < n; ++i) in >> x >> y;   // reference output, not needed here
    }
  }
  return !frames->empty();
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t k = static_cast<std::size_t>(p / 100.0 * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: run_chain <sequence.txt> [calibration.txt] [speed_mps]\n";
    return 2;
  }
  int width = 0, height = 0;
  std::vector<Frame> frames;
  if (!LoadSequence(argv[1], &width, &height, &frames)) {
    std::cerr << "cannot read sequence " << argv[1] << "\n";
    return 2;
  }

  curvature_port::GroundPlane ground;
  bool metric = false;
  if (argc >= 3) {
    std::ifstream cin_(argv[2]);
    curvature_port::CameraCalibration calib;
    if (cin_ >> calib.fx >> calib.fy >> calib.cx >> calib.cy >> calib.height_m >>
        calib.pitch_rad >> calib.yaw_rad) {
      metric = curvature_port::GroundPlaneFromCalibration(calib, &ground);
      double vu, vv;
      calib.VanishingPoint(&vu, &vv);
      std::cerr << "calibration: fx=" << calib.fx << " cy=" << calib.cy
                << " height=" << calib.height_m << " m  pitch="
                << calib.pitch_rad * 180.0 / M_PI << " deg, vanishing point at row "
                << vv << "\n";
    }
  }
  if (!metric) {
    std::cerr << "no usable calibration: metric outputs would be meaningless, "
                 "so only detection and timing are reported\n";
  }
  const double speed = (argc >= 4) ? std::stod(argv[3]) : 15.0;

  curvature_port::DecompositionScratch scratch;
  curvature_port::EgoBoundaryTracker tracker(width, height);
  curvature_port::RoadGeometryFilter filter;
  curvature_port::KinematicLateralMPC mpc;

  std::vector<double> latency;
  latency.reserve(frames.size());
  int detected = 0, commanded = 0, coasting_max = 0;

  std::cout << "frame,detected,offset_m,heading_deg,curvature_1pm,steer_deg,"
               "coasting,latency_us\n";
  std::cout << std::fixed << std::setprecision(4);

  for (const Frame& f : frames) {
    const curvature_port::MaskView view{f.mask.data(), width, height};
    const auto t0 = std::chrono::steady_clock::now();

    const auto polylines = curvature_port::ExtractLanePolylines(view, &scratch);
    const bool have_line = tracker.Update(polylines);

    bool have_geom = false;
    double off = 0.0, head = 0.0, kap = 0.0;
    if (have_line && metric) {
      const auto ground_pts = ground.ToGround(tracker.centerline());
      const auto rg = curvature_port::ReadRoadGeometry(ground_pts, {5.0, 10.0, 20.0});
      if (rg.valid) {
        have_geom = true;
        off = rg.lateral_offset_m;
        head = rg.heading_error_rad;
        kap = rg.curvature_1pm;
      }
    }
    const auto filtered = filter.Update(have_geom, off, head, kap);
    const auto sol = mpc.SteerForGeometry(filtered.lateral_offset_m,
                                          filtered.heading_error_rad,
                                          filtered.curvature_1pm, speed);

    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    latency.push_back(us);
    detected += have_line ? 1 : 0;
    commanded += sol.valid ? 1 : 0;
    coasting_max = std::max(coasting_max, filtered.coasting_frames);

    std::cout << f.index << "," << (have_line ? 1 : 0) << ","
              << filtered.lateral_offset_m << ","
              << filtered.heading_error_rad * 180.0 / M_PI << ","
              << filtered.curvature_1pm << ","
              << (sol.valid ? sol.steer_rad * 180.0 / M_PI : 0.0) << ","
              << filtered.coasting_frames << "," << us << "\n";
  }

  const double n = static_cast<double>(frames.size());
  double total = 0.0;
  for (double v : latency) total += v;
  std::cerr << "\n" << frames.size() << " frames at " << width << "x" << height << "\n"
            << "  ego lane on          " << detected << " frames ("
            << (100.0 * detected / n) << "%)\n"
            << "  steering command on  " << commanded << " frames\n"
            << "  longest coast        " << coasting_max << " frames\n"
            << "  latency mean         " << (total / n) << " us\n"
            << "  latency p50 / p95    " << Percentile(latency, 50) << " / "
            << Percentile(latency, 95) << " us\n"
            << "  latency p99 / max    " << Percentile(latency, 99) << " / "
            << *std::max_element(latency.begin(), latency.end()) << " us\n"
            << "  budget at 20 Hz      "
            << (100.0 * (total / n) / 50000.0) << "% of the 50 ms frame\n";
  return 0;
}
