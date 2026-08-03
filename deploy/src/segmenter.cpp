#include "curvature_port/segmenter.hpp"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <opencv2/imgproc.hpp>

#include "curvature_port/preprocess.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace curvature_port {
namespace {

// ImageNet statistics, matching src/data/transforms.py. The encoder is an
// ImageNet-pretrained ResNet-18, so these are fixed by the pretraining, not tunable.
constexpr float kMeanRgb[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStdRgb[3] = {0.229f, 0.224f, 0.225f};

double MicrosSince(const std::chrono::steady_clock::time_point& start) {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(now - start).count();
}

// FNV-1a over the model file, as a hexadecimal string.
//
// This exists because of a bug that cost an afternoon and would have cost a great deal
// more in a vehicle. TensorRT engines are cached by ONNX Runtime under a name derived
// from the *graph*, and two checkpoints of the same architecture have the same graph.
// Re-export the model from a different checkpoint into the same path, point at the same
// cache, and TensorRT loads the old engine: same topology, previous weights, no error,
// no warning. The masks quietly get worse — in the case that exposed this, lane-class
// IoU against the reference fell from 0.997 to 0.64, which is bad enough to be obvious
// on a video and subtle enough to be blamed on fp16.
//
// So the engine goes in a subdirectory named for the contents of the file it was built
// from. Different weights, different directory, no possibility of reuse. Reading 57 MB
// costs about fifty milliseconds once at startup, against an engine build of minutes.
std::string ModelFingerprint(const std::string& model_path) {
  std::ifstream in(model_path, std::ios::binary);
  if (!in) return "unknown";
  std::uint64_t hash = 1469598103934665603ULL;
  std::vector<char> buffer(1 << 16);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = in.gcount();
    for (std::streamsize i = 0; i < got; ++i) {
      hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

}  // namespace

bool ParseBackend(const std::string& name, Backend* out) {
  if (name == "auto") *out = Backend::kAuto;
  else if (name == "tensorrt" || name == "trt") *out = Backend::kTensorRt;
  else if (name == "cuda") *out = Backend::kCuda;
  else if (name == "cpu") *out = Backend::kCpu;
  else return false;
  return true;
}

const char* BackendName(Backend backend) {
  switch (backend) {
    case Backend::kTensorRt: return "TensorRT";
    case Backend::kCuda: return "CUDA";
    case Backend::kCpu: return "CPU";
    case Backend::kAuto: return "auto";
  }
  return "unknown";
}

struct Segmenter::Impl {
  SegmenterOptions options;
  int width = 0;
  int height = 0;
  Backend backend = Backend::kCpu;
  float logit_threshold = 0.0f;

  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "curvature_port"};
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::string output_name;

  // Preallocated so the per-frame path never asks the allocator for anything.
  std::vector<float> input_tensor;
  std::vector<float> output_tensor;
  cv::Mat resized;
  cv::Mat channels[3];
  SourceRegion roi{0, 0, 0, 0};
  double preprocess_us = 0.0;
  double network_us = 0.0;

  // Build a session on one specific backend, or throw.
  //
  // Throwing is the whole point. The Python benchmark this replaces had to check, after
  // the fact, which providers the session had actually ended up with, because
  // onnxruntime's Python path swallows a provider that fails to load and only warns; an
  // early version of it duly reported a two-second CPU timing under the heading
  // "TensorRT fp16". Appending a provider through the C++ API surfaces that as an
  // exception at the point of failure, so the wrong number cannot be produced in the
  // first place, and kAuto simply moves to the next backend.
  //
  // Note what is *not* checked: that every node landed on the accelerator. ONNX Runtime
  // deliberately keeps shape-related ops on the CPU because they are faster there, so a
  // healthy CUDA session has a handful of CPU nodes in it and demanding otherwise
  // rejects a session that is working correctly.
  std::unique_ptr<Ort::Session> TrySession(const std::string& model_path,
                                           Backend which) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.intra_op_threads > 0) so.SetIntraOpNumThreads(options.intra_op_threads);

    const OrtApi& api = Ort::GetApi();
    // Outlives the UpdateTensorRTProviderOptions call, which only borrows the strings.
    std::string cache_path;
    if (which == Backend::kTensorRt) {
      OrtTensorRTProviderOptionsV2* trt = nullptr;
      Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
      std::vector<const char*> keys{"trt_fp16_enable"};
      std::vector<const char*> values{options.fp16 ? "1" : "0"};
      if (!options.engine_cache_dir.empty()) {
        // Per-model subdirectory, so re-exporting the model cannot silently reuse the
        // engine built from the previous weights. See ModelFingerprint.
        cache_path = (std::filesystem::path(options.engine_cache_dir) /
                      ModelFingerprint(model_path))
                         .string();
        // TensorRT will not create the directory, and reports the omission as a
        // filesystem error during session creation rather than as a caching problem.
        std::error_code ec;
        std::filesystem::create_directories(cache_path, ec);
        keys.push_back("trt_engine_cache_enable");
        values.push_back("1");
        keys.push_back("trt_engine_cache_path");
        values.push_back(cache_path.c_str());
      }
      // The options object owns no C++ destructor of its own, so it is released
      // explicitly on both the success and the throwing path.
      try {
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(),
                                                            values.data(), keys.size()));
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
            static_cast<OrtSessionOptions*>(so), trt));
      } catch (...) {
        api.ReleaseTensorRTProviderOptions(trt);
        throw;
      }
      api.ReleaseTensorRTProviderOptions(trt);
    } else if (which == Backend::kCuda) {
      OrtCUDAProviderOptionsV2* cuda = nullptr;
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
      try {
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
            static_cast<OrtSessionOptions*>(so), cuda));
      } catch (...) {
        api.ReleaseCUDAProviderOptions(cuda);
        throw;
      }
      api.ReleaseCUDAProviderOptions(cuda);
    }
    return std::make_unique<Ort::Session>(env, model_path.c_str(), so);
  }
};

Segmenter::Segmenter(const std::string& model_path, int width, int height,
                     const SegmenterOptions& options)
    : impl_(new Impl) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("segmenter: network input size must be positive");
  }
  impl_->options = options;
  impl_->width = width;
  impl_->height = height;

  // Threshold in logit space: sigmoid(z) >= t is z >= log(t / (1 - t)), which saves
  // evaluating an exponential per pixel over a 512x288 mask every frame.
  const float t = std::min(std::max(options.threshold, 1e-6f), 1.0f - 1e-6f);
  impl_->logit_threshold = std::log(t / (1.0f - t));

  std::vector<Backend> order;
  if (options.backend == Backend::kAuto) {
    order = {Backend::kTensorRt, Backend::kCuda, Backend::kCpu};
  } else {
    order = {options.backend};
  }

  std::string last_error;
  for (Backend candidate : order) {
    try {
      impl_->session = impl_->TrySession(model_path, candidate);
      impl_->backend = candidate;
      break;
    } catch (const std::exception& exc) {
      last_error = std::string(BackendName(candidate)) + ": " + exc.what();
    }
  }
  if (!impl_->session) {
    throw std::runtime_error("segmenter: no usable backend (" + last_error + ")");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  impl_->input_name = impl_->session->GetInputNameAllocated(0, allocator).get();
  impl_->output_name = impl_->session->GetOutputNameAllocated(0, allocator).get();

  impl_->input_tensor.resize(static_cast<size_t>(3) * height * width);
  impl_->output_tensor.resize(static_cast<size_t>(height) * width);
  impl_->resized.create(height, width, CV_8UC3);
}

Segmenter::~Segmenter() = default;

bool Segmenter::Run(const std::uint8_t* src, int src_width, int src_height,
                    int src_stride, PixelOrder order, std::uint8_t* out_mask) {
  if (src == nullptr || out_mask == nullptr) return false;
  if (src_width <= 1 || src_height <= 1) return false;
  if (src_stride <= 0) src_stride = src_width * 3;

  const auto t_pre = std::chrono::steady_clock::now();

  const int w = impl_->width;
  const int h = impl_->height;
  impl_->roi = NetworkInputRegion(src_width, src_height, w, h,
                                  impl_->options.sky_frac);
  const SourceRegion& roi = impl_->roi;
  if (roi.width <= 0 || roi.height <= 0) return false;

  // No copy: a header over the caller's buffer, cropped to the region.
  const cv::Mat source(src_height, src_width, CV_8UC3,
                       const_cast<std::uint8_t*>(src), static_cast<size_t>(src_stride));
  const cv::Mat region = source(cv::Rect(roi.x, roi.y, roi.width, roi.height));
  cv::resize(region, impl_->resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);

  // Normalize into NCHW. Splitting to planes and letting convertTo do the arithmetic is
  // four times quicker than the obvious strided loop, because both halves then run
  // vectorized over contiguous memory instead of reading every third byte.
  //
  // The channel order is resolved by which plane goes where rather than by a cvtColor
  // pass: the resize does not care about it, so the only place it has to be right is
  // where a channel meets its own mean and standard deviation.
  //
  // convertTo computes v * (1/s) - m/s where the reference computes (v - m) * (1/s).
  // Those differ in the last bit or two of the mantissa. That is four orders of
  // magnitude below what the fp16 engine already moves the logits by, and the mask is
  // thresholded afterwards, so it is not a difference the chain can see.
  const int plane = w * h;
  float* dst = impl_->input_tensor.data();
  cv::split(impl_->resized, impl_->channels);
  for (int c = 0; c < 3; ++c) {
    const int src_c = (order == PixelOrder::kBgr) ? 2 - c : c;
    const float mean = kMeanRgb[c] * 255.0f;
    const float inv_std = 1.0f / (kStdRgb[c] * 255.0f);
    cv::Mat out(h, w, CV_32F, dst + static_cast<size_t>(c) * plane);
    impl_->channels[src_c].convertTo(out, CV_32F, inv_std, -mean * inv_std);
  }
  impl_->preprocess_us = MicrosSince(t_pre);

  const auto t_net = std::chrono::steady_clock::now();
  const std::array<std::int64_t, 4> input_shape{1, 3, h, w};
  const std::array<std::int64_t, 4> output_shape{1, 1, h, w};
  Ort::Value input = Ort::Value::CreateTensor<float>(
      impl_->memory_info, impl_->input_tensor.data(), impl_->input_tensor.size(),
      input_shape.data(), input_shape.size());
  Ort::Value output = Ort::Value::CreateTensor<float>(
      impl_->memory_info, impl_->output_tensor.data(), impl_->output_tensor.size(),
      output_shape.data(), output_shape.size());

  const char* input_names[] = {impl_->input_name.c_str()};
  const char* output_names[] = {impl_->output_name.c_str()};
  impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names,
                      &output, 1);
  impl_->network_us = MicrosSince(t_net);

  const float threshold = impl_->logit_threshold;
  const float* logits = impl_->output_tensor.data();
  for (int i = 0; i < plane; ++i) {
    out_mask[i] = logits[i] >= threshold ? 1 : 0;
  }
  return true;
}

void Segmenter::last_source_roi(int* x, int* y, int* width, int* height) const {
  if (x) *x = impl_->roi.x;
  if (y) *y = impl_->roi.y;
  if (width) *width = impl_->roi.width;
  if (height) *height = impl_->roi.height;
}

double Segmenter::last_preprocess_us() const { return impl_->preprocess_us; }
double Segmenter::last_network_us() const { return impl_->network_us; }
Backend Segmenter::backend() const { return impl_->backend; }
int Segmenter::width() const { return impl_->width; }
int Segmenter::height() const { return impl_->height; }

}  // namespace curvature_port
