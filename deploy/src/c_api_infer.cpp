// C ABI for the segmenter and the fused pipeline. See c_api.h.
//
// Compiled into the shared library only when a segmenter is available; c_api.cpp
// carries stubs for the same symbols otherwise, so a caller always links and finds out
// at run time via cp_segmenter_available rather than at load time via a missing symbol.

#include "curvature_port/c_api.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "chain_impl.hpp"
#include "curvature_port/segmenter.hpp"

namespace {

void WriteError(char* err, int32_t err_len, const std::string& message) {
  if (err == nullptr || err_len <= 0) return;
  const std::size_t n =
      std::min(message.size(), static_cast<std::size_t>(err_len - 1));
  std::memcpy(err, message.data(), n);
  err[n] = '\0';
}

// Translate the flat config into the C++ options, defaulting the fields a caller is
// allowed to leave unset.
bool BuildOptions(const cp_segmenter_config* config,
                  curvature_port::SegmenterOptions* out, std::string* error) {
  if (config == nullptr || config->model_path == nullptr) {
    *error = "no model path";
    return false;
  }
  if (config->width <= 0 || config->height <= 0) {
    *error = "network input size must be positive";
    return false;
  }
  const std::string backend = (config->backend == nullptr) ? "auto" : config->backend;
  if (!curvature_port::ParseBackend(backend, &out->backend)) {
    *error = "unknown backend '" + backend + "'";
    return false;
  }
  out->fp16 = config->fp16 != 0;
  out->engine_cache_dir =
      (config->engine_cache_dir == nullptr) ? "" : config->engine_cache_dir;
  out->threshold = (config->threshold > 0.0 && config->threshold < 1.0)
                       ? static_cast<float>(config->threshold)
                       : 0.5f;
  out->sky_frac = (config->sky_frac >= 0.0 && config->sky_frac < 1.0) ? config->sky_frac
                                                                     : 0.30;
  return true;
}

double MicrosSince(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                   start)
      .count();
}

}  // namespace

struct cp_segmenter {
  curvature_port::Segmenter impl;
  std::string backend_name;

  cp_segmenter(const std::string& path, int w, int h,
               const curvature_port::SegmenterOptions& options)
      : impl(path, w, h, options),
        backend_name(curvature_port::BackendName(impl.backend())) {}
};

struct cp_pipeline {
  curvature_port::Segmenter segmenter;
  cp_chain chain;
  std::vector<std::uint8_t> mask;
  std::string backend_name;
  double chain_us = 0.0;

  cp_pipeline(const std::string& path, int w, int h,
              const curvature_port::SegmenterOptions& options)
      : segmenter(path, w, h, options),
        chain(w, h),
        mask(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0),
        backend_name(curvature_port::BackendName(segmenter.backend())) {}
};

extern "C" {

int32_t cp_segmenter_available(void) { return 1; }

cp_segmenter* cp_segmenter_create(const cp_segmenter_config* config, char* err,
                                  int32_t err_len) {
  curvature_port::SegmenterOptions options;
  std::string error;
  if (!BuildOptions(config, &options, &error)) {
    WriteError(err, err_len, error);
    return nullptr;
  }
  // Every failure below crosses a C boundary, where an exception would be undefined
  // behaviour rather than an error message.
  try {
    return new cp_segmenter(config->model_path, config->width, config->height, options);
  } catch (const std::exception& exc) {
    WriteError(err, err_len, exc.what());
  } catch (...) {
    WriteError(err, err_len, "unknown failure loading the model");
  }
  return nullptr;
}

void cp_segmenter_destroy(cp_segmenter* segmenter) { delete segmenter; }

int32_t cp_segmenter_run(cp_segmenter* segmenter, const uint8_t* src, int32_t src_width,
                         int32_t src_height, int32_t src_stride, int32_t bgr,
                         uint8_t* out_mask) {
  if (segmenter == nullptr) return 0;
  try {
    const auto order =
        bgr != 0 ? curvature_port::PixelOrder::kBgr : curvature_port::PixelOrder::kRgb;
    return segmenter->impl.Run(src, src_width, src_height, src_stride, order, out_mask)
               ? 1
               : 0;
  } catch (...) {
    return 0;
  }
}

const char* cp_segmenter_backend(const cp_segmenter* segmenter) {
  return segmenter == nullptr ? "none" : segmenter->backend_name.c_str();
}

double cp_segmenter_last_preprocess_us(const cp_segmenter* segmenter) {
  return segmenter == nullptr ? 0.0 : segmenter->impl.last_preprocess_us();
}

double cp_segmenter_last_network_us(const cp_segmenter* segmenter) {
  return segmenter == nullptr ? 0.0 : segmenter->impl.last_network_us();
}

cp_pipeline* cp_pipeline_create(const cp_segmenter_config* config,
                                const double* calibration, char* err, int32_t err_len) {
  curvature_port::SegmenterOptions options;
  std::string error;
  if (!BuildOptions(config, &options, &error)) {
    WriteError(err, err_len, error);
    return nullptr;
  }
  try {
    auto* pipeline =
        new cp_pipeline(config->model_path, config->width, config->height, options);
    if (calibration != nullptr &&
        !curvature_port_detail::ApplyCalibration(&pipeline->chain, calibration)) {
      // Not fatal: without a ground plane the chain still tracks the ego lane, it just
      // has nothing to project it onto, and the caller can see that in has_geometry.
      WriteError(err, err_len, "calibration did not yield a usable ground plane");
    }
    return pipeline;
  } catch (const std::exception& exc) {
    WriteError(err, err_len, exc.what());
  } catch (...) {
    WriteError(err, err_len, "unknown failure loading the model");
  }
  return nullptr;
}

void cp_pipeline_destroy(cp_pipeline* pipeline) { delete pipeline; }

void cp_pipeline_reset(cp_pipeline* pipeline) {
  if (pipeline == nullptr) return;
  pipeline->chain.tracker =
      curvature_port::EgoBoundaryTracker(pipeline->chain.width, pipeline->chain.height);
  pipeline->chain.filter.Reset();
}

int32_t cp_pipeline_process(cp_pipeline* pipeline, const uint8_t* src, int32_t src_width,
                            int32_t src_height, int32_t src_stride, int32_t bgr,
                            double speed_mps, cp_frame_result* out) {
  if (pipeline == nullptr || out == nullptr) return 0;
  try {
    const auto order =
        bgr != 0 ? curvature_port::PixelOrder::kBgr : curvature_port::PixelOrder::kRgb;
    if (!pipeline->segmenter.Run(src, src_width, src_height, src_stride, order,
                                 pipeline->mask.data())) {
      std::memset(out, 0, sizeof(*out));
      return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    curvature_port_detail::ProcessMask(&pipeline->chain, pipeline->mask.data(),
                                       speed_mps, out);
    pipeline->chain_us = MicrosSince(start);
    return 1;
  } catch (...) {
    return 0;
  }
}

int32_t cp_pipeline_centerline(const cp_pipeline* pipeline, double* out_xy,
                               int32_t max_points) {
  if (pipeline == nullptr) return 0;
  return curvature_port_detail::CopyCenterline(&pipeline->chain, out_xy, max_points);
}

int32_t cp_pipeline_boundary(const cp_pipeline* pipeline, int32_t side, double* out_xy,
                             int32_t max_points) {
  if (pipeline == nullptr) return 0;
  return curvature_port_detail::CopyBoundary(&pipeline->chain, side, out_xy, max_points);
}

const uint8_t* cp_pipeline_mask(const cp_pipeline* pipeline) {
  return pipeline == nullptr ? nullptr : pipeline->mask.data();
}

void cp_pipeline_source_roi(const cp_pipeline* pipeline, int32_t* out_xywh) {
  if (pipeline == nullptr || out_xywh == nullptr) return;
  int x = 0, y = 0, w = 0, h = 0;
  pipeline->segmenter.last_source_roi(&x, &y, &w, &h);
  out_xywh[0] = x;
  out_xywh[1] = y;
  out_xywh[2] = w;
  out_xywh[3] = h;
}

double cp_pipeline_last_preprocess_us(const cp_pipeline* pipeline) {
  return pipeline == nullptr ? 0.0 : pipeline->segmenter.last_preprocess_us();
}

double cp_pipeline_last_network_us(const cp_pipeline* pipeline) {
  return pipeline == nullptr ? 0.0 : pipeline->segmenter.last_network_us();
}

double cp_pipeline_last_chain_us(const cp_pipeline* pipeline) {
  return pipeline == nullptr ? 0.0 : pipeline->chain_us;
}

}  // extern "C"
