// What each execution provider is actually worth, measured where the code runs.
//
// This was a Python script, which was the wrong place for it twice over. It timed
// onnxruntime through its Python bindings and reported the result as the cost of the
// deployed path, and it compared those numbers against a PyTorch baseline that the
// deployed path does not contain. The question a deployment has is narrower: given this
// ONNX file, which provider should the vehicle's process open a session with.
//
// Parity comes before timing. A provider that produces a different mask is not faster,
// it is broken, and the tolerance is stated in the terms the rest of the chain uses:
// agreement on the thresholded mask, not on the logits, because the mask is what the
// geometry consumes. The CPU provider is the reference, being the one with no fp16
// engine and no fused kernels between it and the graph.
//
//   bench_backends --model <model.onnx> [--image <frame.jpg>] [--cache <dir>]
//                  [--size <w>x<h>] [--iters <n>] [--skip-cpu]
//
// Without --image the input is deterministic pseudo-random noise, which times the same
// but tells you nothing about whether the mask is plausible; prefer a real frame.

#include "curvature_port/segmenter.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Measurement {
  std::string name;
  double mean_us = 0.0;
  double p95_us = 0.0;
  double network_us = 0.0;
  double preprocess_us = 0.0;
  double agreement = 1.0;
  int iterations = 0;
};

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

bool HasFlag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (flag == argv[i]) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = Argument(argc, argv, "--model", "");
  if (model.empty()) {
    std::cerr << "usage: bench_backends --model <model.onnx> [--image <frame.jpg>] "
                 "[--cache <dir>] [--size <w>x<h>] [--iters <n>] [--skip-cpu]\n";
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
  const std::string cache = Argument(argc, argv, "--cache", "");
  const int max_iters = std::stoi(Argument(argc, argv, "--iters", "60"));
  const bool skip_cpu = HasFlag(argc, argv, "--skip-cpu");

  // One input frame, shared by every backend so the comparison is like for like.
  cv::Mat source;
  const std::string image_path = Argument(argc, argv, "--image", "");
  if (!image_path.empty()) {
    source = cv::imread(image_path, cv::IMREAD_COLOR);
    if (source.empty()) {
      std::cerr << "cannot read " << image_path << "\n";
      return 2;
    }
    std::cerr << "input: " << image_path << " (" << source.cols << "x" << source.rows
              << ")\n";
  } else {
    source.create(720, 1280, CV_8UC3);
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < source.rows; ++y) {
      cv::Vec3b* row = source.ptr<cv::Vec3b>(y);
      for (int x = 0; x < source.cols; ++x) {
        row[x] = cv::Vec3b(static_cast<std::uint8_t>(byte(rng)),
                           static_cast<std::uint8_t>(byte(rng)),
                           static_cast<std::uint8_t>(byte(rng)));
      }
    }
    std::cerr << "input: pseudo-random noise; pass --image for a real frame\n";
  }

  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::vector<std::uint8_t> reference(pixels, 0), mask(pixels, 0);
  bool have_reference = false;

  struct Candidate {
    curvature_port::Backend backend;
    bool fp16;
    const char* label;
  };
  std::vector<Candidate> candidates;
  if (!skip_cpu) candidates.push_back({curvature_port::Backend::kCpu, false, "CPU"});
  candidates.push_back({curvature_port::Backend::kCuda, false, "CUDA"});
  candidates.push_back({curvature_port::Backend::kTensorRt, false, "TensorRT fp32"});
  candidates.push_back({curvature_port::Backend::kTensorRt, true, "TensorRT fp16"});

  std::vector<Measurement> results;
  for (const Candidate& candidate : candidates) {
    curvature_port::SegmenterOptions options;
    options.backend = candidate.backend;
    options.fp16 = candidate.fp16;
    // Separate cache directories: TensorRT keys its engines on the graph, not on the
    // precision, so an fp32 and an fp16 build of the same model would otherwise reuse
    // each other's engine and both rows would report whichever ran first.
    if (!cache.empty()) {
      options.engine_cache_dir =
          cache + (candidate.fp16 ? "/fp16" : "/fp32");
    }

    std::unique_ptr<curvature_port::Segmenter> segmenter;
    try {
      segmenter = std::make_unique<curvature_port::Segmenter>(model, width, height,
                                                              options);
    } catch (const std::exception& exc) {
      // Not a failure of the benchmark. A machine without TensorRT is a machine that
      // should run CUDA, and the table saying so is the useful output.
      std::cerr << "skipping " << candidate.label << ": " << exc.what() << "\n";
      continue;
    }
    std::cerr << "running " << candidate.label << "...\n";

    auto once = [&]() {
      segmenter->Run(source.data, source.cols, source.rows,
                     static_cast<int>(source.step), curvature_port::PixelOrder::kBgr,
                     mask.data());
    };

    once();  // warm up: the first call pays for lazy kernel loading and allocation
    Measurement m;
    m.name = candidate.label;

    if (!have_reference) {
      reference = mask;
      have_reference = true;
    } else {
      std::size_t agree = 0;
      for (std::size_t i = 0; i < pixels; ++i) agree += (mask[i] == reference[i]) ? 1 : 0;
      m.agreement = static_cast<double>(agree) / static_cast<double>(pixels);
    }

    // The CPU provider is three orders of magnitude slower than the others here, so a
    // fixed iteration count either takes minutes on CPU or is too few samples on
    // TensorRT. Probe once, then spend about two seconds on each.
    const auto probe_start = std::chrono::steady_clock::now();
    once();
    const double probe_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                  probe_start)
            .count();
    const int iterations = std::max(
        3, std::min(max_iters, static_cast<int>(2e6 / std::max(probe_us, 1.0))));

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iterations));
    double net_sum = 0.0, pre_sum = 0.0;
    for (int i = 0; i < iterations; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      once();
      times.push_back(
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
              .count());
      net_sum += segmenter->last_network_us();
      pre_sum += segmenter->last_preprocess_us();
    }
    double sum = 0.0;
    for (double t : times) sum += t;
    m.mean_us = sum / static_cast<double>(times.size());
    m.p95_us = Percentile(times, 95);
    m.network_us = net_sum / static_cast<double>(iterations);
    m.preprocess_us = pre_sum / static_cast<double>(iterations);
    m.iterations = iterations;
    results.push_back(m);
  }

  if (results.empty()) {
    std::cerr << "no backend could run the model\n";
    return 1;
  }

  // Against the fastest, not the first: the useful comparison for someone choosing a
  // provider is what the alternatives cost relative to the best one available, and with
  // the CPU three orders of magnitude behind, ratios against it are unreadable.
  double fastest = results.front().mean_us;
  for (const Measurement& m : results) fastest = std::min(fastest, m.mean_us);

  std::cout << std::left << std::setw(18) << "backend" << std::right << std::setw(10)
            << "total us" << std::setw(11) << "network" << std::setw(11) << "preprocess"
            << std::setw(10) << "p95 us" << std::setw(8) << "fps" << std::setw(10)
            << "vs best" << std::setw(14) << "mask agree" << std::setw(7) << "iters"
            << "\n";
  std::cout << std::fixed;
  for (const Measurement& m : results) {
    std::cout << std::left << std::setw(18) << m.name << std::right << std::setprecision(0)
              << std::setw(10) << m.mean_us << std::setw(11) << m.network_us
              << std::setw(11) << m.preprocess_us << std::setw(10) << m.p95_us
              << std::setw(8) << (1e6 / m.mean_us) << std::setprecision(2) << std::setw(9)
              << (m.mean_us / fastest) << "x" << std::setprecision(4) << std::setw(13)
              << (100.0 * m.agreement) << "%" << std::setw(7) << m.iterations << "\n";
  }
  std::cout << "\nAgreement is against " << results.front().name
            << ", on the thresholded mask the geometry chain consumes.\n"
            << "Preprocess is the crop, resize and normalize, which is the same work on "
               "every backend;\nsubtract it to compare the network alone.\n";
  return 0;
}
