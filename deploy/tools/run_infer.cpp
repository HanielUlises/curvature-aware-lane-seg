// The whole pipeline, in one process, on real footage.
//
//   frame -> crop/resize/normalize -> network -> mask -> lane polylines
//         -> tracked ego boundaries -> centreline -> ground projection
//         -> offset / heading / curvature -> temporal filter -> lateral MPC -> steering
//
// run_chain, next to this, runs everything after the mask on a recorded mask fixture.
// That was the honest thing to measure while the network still ran in Python, but it
// measured the cheap 2% of the frame. This runs the network here too, so the latency
// below is the whole per-frame cost of the deployed path and nothing is hidden in
// another process.
//
// CSV on stdout so a run diffs against the Python reference; the summary on stderr,
// with the latency split three ways. The split is the point: it says whether the
// remaining budget is in the network, in the pixel handling around it, or in the
// geometry, and those have completely different answers.
//
//   run_infer --model <model.onnx> --source <dir|video> [options]
//
//     --calibration <file>  seven numbers, or the calibration.json the project writes
//     --backend <name>      auto (default), tensorrt, cuda, cpu
//     --cache <dir>         where TensorRT keeps its built engine
//     --speed <mps>         forward speed handed to the controller (default 15)
//     --max-frames <n>      stop after n frames
//     --size <w>x<h>        network input size (default 512x288)
//     --overlay <file.mp4>  also render the result as video

#include "curvature_port/c_api.h"
#include "curvature_port/segmenter.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char* kImageExtensions[] = {".jpg", ".jpeg", ".png", ".bmp"};

bool IsImage(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (const char* known : kImageExtensions) {
    if (ext == known) return true;
  }
  return false;
}

// Natural order, matching scripts/infer_sequence._natural_key: digit runs compare as
// numbers so frame 2 precedes frame 10. Lexicographic order would interleave the clip
// and hand the tracker a sequence that is not one, which looks exactly like tracker
// failure and is not.
bool NaturalLess(const std::string& a, const std::string& b) {
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit(static_cast<unsigned char>(a[i])) &&
        std::isdigit(static_cast<unsigned char>(b[j]))) {
      std::size_t i0 = i, j0 = j;
      while (i < a.size() && std::isdigit(static_cast<unsigned char>(a[i]))) ++i;
      while (j < b.size() && std::isdigit(static_cast<unsigned char>(b[j]))) ++j;
      const std::string da = a.substr(i0, i - i0), db = b.substr(j0, j - j0);
      const std::string ta = da.substr(std::min(da.find_first_not_of('0'), da.size() - 1));
      const std::string tb = db.substr(std::min(db.find_first_not_of('0'), db.size() - 1));
      if (ta.size() != tb.size()) return ta.size() < tb.size();
      if (ta != tb) return ta < tb;
    } else {
      if (a[i] != b[j]) return a[i] < b[j];
      ++i;
      ++j;
    }
  }
  return a.size() - i < b.size() - j;
}

// Images directly inside the directory, or, failing that, the concatenation of its
// child clip directories in order, so a TuSimple date folder reads as continuous
// driving.
std::vector<fs::path> OrderedImages(const fs::path& source) {
  std::vector<fs::path> direct, dirs, frames;
  for (const auto& entry : fs::directory_iterator(source)) {
    if (entry.is_directory()) dirs.push_back(entry.path());
    else if (IsImage(entry.path())) direct.push_back(entry.path());
  }
  auto by_name = [](const fs::path& a, const fs::path& b) {
    return NaturalLess(a.filename().string(), b.filename().string());
  };
  if (!direct.empty()) {
    std::sort(direct.begin(), direct.end(), by_name);
    return direct;
  }
  std::sort(dirs.begin(), dirs.end(), by_name);
  for (const fs::path& dir : dirs) {
    std::vector<fs::path> clip;
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (IsImage(entry.path())) clip.push_back(entry.path());
    }
    std::sort(clip.begin(), clip.end(), by_name);
    frames.insert(frames.end(), clip.begin(), clip.end());
  }
  return frames;
}

// Seven whitespace-separated numbers, or the flat JSON object the calibration script
// writes. Full JSON parsing would be a dependency for one object of seven numbers, so
// this scans for the keys it needs and ignores everything else; a key that is absent
// leaves its field at zero, which GroundPlaneFromCalibration then rejects.
bool ReadCalibration(const std::string& path, double* out) {
  std::ifstream in(path);
  if (!in) return false;
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  if (text.find('{') == std::string::npos) {
    std::istringstream nums(text);
    for (int i = 0; i < 7; ++i) {
      if (!(nums >> out[i])) return false;
    }
    return true;
  }
  const char* keys[7] = {"fx", "fy", "cx", "cy", "height_m", "pitch_rad", "yaw_rad"};
  int found = 0;
  for (int i = 0; i < 7; ++i) {
    out[i] = 0.0;
    const std::string needle = std::string("\"") + keys[i] + "\"";
    const std::size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    const std::size_t colon = text.find(':', at);
    if (colon == std::string::npos) continue;
    try {
      out[i] = std::stod(text.substr(colon + 1));
      ++found;
    } catch (const std::exception&) {
      return false;
    }
  }
  return found == 7;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t k = static_cast<std::size_t>(p / 100.0 * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

std::string Argument(int argc, char** argv, const std::string& flag,
                     const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return argv[i + 1];
  }
  return fallback;
}

// Mask in red, tracked boundaries and the ego centreline over it, and the numbers the
// controller acted on. Deliberately plain: scripts/viz_control.py is what renders the
// presentation videos, and this exists to show that the same picture comes out of a
// process with no Python in it.
void DrawOverlay(cv::Mat* frame, const std::uint8_t* mask, const cp_pipeline* pipeline,
                 const cp_frame_result& r, double frame_ms) {
  const int w = frame->cols, h = frame->rows;
  for (int y = 0; y < h; ++y) {
    cv::Vec3b* row = frame->ptr<cv::Vec3b>(y);
    const std::uint8_t* m = mask + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      if (m[x]) row[x] = cv::Vec3b(static_cast<std::uint8_t>(row[x][0] * 0.5),
                                   static_cast<std::uint8_t>(row[x][1] * 0.5),
                                   static_cast<std::uint8_t>(row[x][2] * 0.5 + 128));
    }
  }

  std::vector<double> xy(2 * 512);
  for (int side = 0; side < 2; ++side) {
    const int n = cp_pipeline_boundary(pipeline, side, xy.data(), 512);
    for (int i = 0; i < n; ++i) {
      cv::circle(*frame, cv::Point(static_cast<int>(xy[2 * i]),
                                   static_cast<int>(xy[2 * i + 1])),
                 1, cv::Scalar(80, 220, 80), -1, cv::LINE_AA);
    }
  }
  const int n = cp_pipeline_centerline(pipeline, xy.data(), 512);
  for (int i = 0; i < n; ++i) {
    cv::circle(*frame,
               cv::Point(static_cast<int>(xy[2 * i]), static_cast<int>(xy[2 * i + 1])),
               2, cv::Scalar(60, 230, 250), -1, cv::LINE_AA);
  }

  std::ostringstream line;
  line << std::fixed << std::setprecision(2);
  if (r.has_geometry) {
    line << "offset " << r.lateral_offset_m << " m   heading "
         << r.heading_error_rad * 180.0 / M_PI << " deg   steer "
         << r.steer_rad * 180.0 / M_PI << " deg";
  } else {
    line << (r.has_centerline ? "no ground geometry" : "no ego lane");
  }
  cv::putText(*frame, line.str(), cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  std::ostringstream timing;
  timing << std::fixed << std::setprecision(2) << frame_ms << " ms/frame";
  cv::putText(*frame, timing.str(), cv::Point(8, h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.42,
              cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = Argument(argc, argv, "--model", "");
  const std::string source = Argument(argc, argv, "--source", "");
  if (model.empty() || source.empty()) {
    std::cerr << "usage: run_infer --model <model.onnx> --source <dir|video> "
                 "[--calibration <file>] [--backend auto|tensorrt|cuda|cpu] "
                 "[--cache <dir>] [--speed <mps>] [--max-frames <n>] [--size <w>x<h>] "
                 "[--overlay <file.mp4>]\n";
    return 2;
  }
  if (!cp_segmenter_available()) {
    std::cerr << "this build has no segmenter; configure with -DONNXRUNTIME_ROOT=<dir>\n";
    return 2;
  }

  int width = 512, height = 288;
  const std::string size = Argument(argc, argv, "--size", "");
  if (!size.empty()) {
    const std::size_t x = size.find('x');
    if (x == std::string::npos) {
      std::cerr << "--size wants <width>x<height>\n";
      return 2;
    }
    width = std::stoi(size.substr(0, x));
    height = std::stoi(size.substr(x + 1));
  }
  const std::string backend = Argument(argc, argv, "--backend", "auto");
  const std::string cache = Argument(argc, argv, "--cache", "");
  const std::string overlay_path = Argument(argc, argv, "--overlay", "");
  const double speed = std::stod(Argument(argc, argv, "--speed", "15.0"));
  const long max_frames = std::stol(Argument(argc, argv, "--max-frames", "0"));

  double calib[7] = {0, 0, 0, 0, 0, 0, 0};
  bool metric = false;
  const std::string calib_path = Argument(argc, argv, "--calibration", "");
  if (!calib_path.empty()) {
    metric = ReadCalibration(calib_path, calib);
    if (!metric) std::cerr << "cannot read calibration " << calib_path << "\n";
  }
  if (!metric) {
    std::cerr << "no usable calibration: metric outputs would be meaningless, "
                 "so only detection and timing are reported\n";
  }

  cp_segmenter_config config{};
  config.model_path = model.c_str();
  config.width = width;
  config.height = height;
  config.backend = backend.c_str();
  config.fp16 = 1;
  config.engine_cache_dir = cache.empty() ? nullptr : cache.c_str();
  config.threshold = 0.5;
  config.sky_frac = 0.30;

  char err[512] = {0};
  // The engine build is minutes on a cold cache, so say what is happening rather than
  // looking hung.
  std::cerr << "loading " << model << " on " << backend
            << (cache.empty() ? "" : " (engine cache " + cache + ")") << "\n";
  cp_pipeline* pipeline =
      cp_pipeline_create(&config, metric ? calib : nullptr, err, sizeof(err));
  if (pipeline == nullptr) {
    std::cerr << "cannot start the pipeline: " << err << "\n";
    return 2;
  }
  if (err[0] != '\0') std::cerr << "warning: " << err << "\n";

  const fs::path src(source);
  std::vector<fs::path> images;
  cv::VideoCapture capture;
  if (fs::is_directory(src)) {
    images = OrderedImages(src);
    if (images.empty()) {
      std::cerr << "no images under " << source << "\n";
      cp_pipeline_destroy(pipeline);
      return 2;
    }
  } else if (!capture.open(source)) {
    std::cerr << "cannot open " << source << "\n";
    cp_pipeline_destroy(pipeline);
    return 2;
  }

  cv::VideoWriter writer;
  if (!overlay_path.empty()) {
    writer.open(overlay_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20.0,
                cv::Size(width, height));
    if (!writer.isOpened()) std::cerr << "cannot write " << overlay_path << "\n";
  }

  std::vector<double> pre_us, net_us, chain_us, total_us;
  int detected = 0, commanded = 0, index = 0;
  cv::Mat frame, overlay;

  std::cout << "frame,detected,offset_m,heading_deg,curvature_1pm,steer_deg,saturated,"
               "coasting,preprocess_us,network_us,chain_us\n";
  std::cout << std::fixed << std::setprecision(4);

  while (true) {
    if (max_frames > 0 && index >= max_frames) break;
    if (!images.empty()) {
      if (static_cast<std::size_t>(index) >= images.size()) break;
      frame = cv::imread(images[static_cast<std::size_t>(index)].string(),
                         cv::IMREAD_COLOR);
      if (frame.empty()) {
        ++index;
        continue;
      }
    } else if (!capture.read(frame) || frame.empty()) {
      break;
    }

    cp_frame_result r{};
    const auto t0 = std::chrono::steady_clock::now();
    // The frame arrives BGR from both OpenCV readers; the flag says so rather than
    // paying for a colour conversion the normalization can absorb.
    const int ok = cp_pipeline_process(pipeline, frame.data, frame.cols, frame.rows,
                                       static_cast<int>(frame.step), 1, speed, &r);
    const double whole_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
            .count();
    if (!ok) {
      std::cerr << "frame " << index << " failed\n";
      ++index;
      continue;
    }

    pre_us.push_back(cp_pipeline_last_preprocess_us(pipeline));
    net_us.push_back(cp_pipeline_last_network_us(pipeline));
    chain_us.push_back(cp_pipeline_last_chain_us(pipeline));
    total_us.push_back(whole_us);
    detected += r.has_centerline ? 1 : 0;
    commanded += r.has_geometry ? 1 : 0;

    std::cout << index << "," << r.has_centerline << "," << r.lateral_offset_m << ","
              << r.heading_error_rad * 180.0 / M_PI << "," << r.curvature_1pm << ","
              << r.steer_rad * 180.0 / M_PI << "," << r.saturated << ","
              << r.coasting_frames << "," << pre_us.back() << "," << net_us.back() << ","
              << chain_us.back() << "\n";

    if (writer.isOpened()) {
      int roi[4] = {0, 0, 0, 0};
      cp_pipeline_source_roi(pipeline, roi);
      cv::resize(frame(cv::Rect(roi[0], roi[1], roi[2], roi[3])), overlay,
                 cv::Size(width, height), 0, 0, cv::INTER_AREA);
      DrawOverlay(&overlay, cp_pipeline_mask(pipeline), pipeline, r, whole_us / 1000.0);
      writer.write(overlay);
    }
    ++index;
  }

  if (writer.isOpened()) writer.release();
  cp_pipeline_destroy(pipeline);

  if (total_us.empty()) {
    std::cerr << "no frames read from " << source << "\n";
    return 1;
  }

  auto mean = [](const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
  };
  const double n = static_cast<double>(total_us.size());
  const double whole = mean(total_us);
  std::cerr << std::fixed << std::setprecision(1) << "\n" << total_us.size()
            << " frames at " << width << "x" << height << " on " << backend << "\n"
            << "  ego lane on          " << detected << " frames ("
            << (100.0 * detected / n) << "%)\n"
            << "  road geometry on     " << commanded << " frames\n"
            << "  preprocess           " << mean(pre_us) << " us  (p95 "
            << Percentile(pre_us, 95) << ")\n"
            << "  network              " << mean(net_us) << " us  (p95 "
            << Percentile(net_us, 95) << ")\n"
            << "  geometry chain       " << mean(chain_us) << " us  (p95 "
            << Percentile(chain_us, 95) << ")\n"
            << "  whole frame          " << whole << " us  (p95 "
            << Percentile(total_us, 95) << ")  = " << std::setprecision(0)
            << (1e6 / whole) << " fps\n"
            << std::setprecision(1) << "  chain share          "
            << (100.0 * mean(chain_us) / whole) << "%\n";
  return 0;
}
