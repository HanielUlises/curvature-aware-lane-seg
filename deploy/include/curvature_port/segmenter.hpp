// The segmenter, on the deployment path.
//
// Everything downstream of the mask already ran in C++; the network did not, so the
// deployed pipeline still had to re-enter Python for the one stage that costs 98% of
// the frame budget. This runs it here instead: an ONNX Runtime session over the
// exported graph, with the same preprocessing the Python reference applies, so a frame
// goes image -> mask -> steering without leaving the process.
//
// The export itself stays in Python and always will, because it reads a PyTorch
// checkpoint. That is a build step, not a per-frame one, and the artefact it writes is
// what this consumes.
//
// Preprocessing is a port of src/data/transforms.py preprocess_geometry composed with
// scripts/infer_sequence._center_crop_aspect. Both are crops followed by one resize, so
// the composition is a single region-of-interest and a single INTER_AREA resize, which
// is what this computes. It matters that the arithmetic matches to the pixel: the crop
// is aspect-preserving precisely so the resize is isotropic and the curvature the whole
// project keys on survives it, and an off-by-one in the region would tilt the ground
// plane the calibration was fitted against.
//
// OpenCV is used for the resize rather than a hand-rolled one, for the same reason:
// INTER_AREA has a specific definition and reimplementing it would introduce a
// difference from the reference that no golden vector would catch.

#ifndef CURVATURE_PORT_SEGMENTER_HPP
#define CURVATURE_PORT_SEGMENTER_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace curvature_port {

// Which execution provider to run the graph on.
//
// kAuto walks TensorRT -> CUDA -> CPU and keeps the first that builds a session. The
// order is a measurement, not a preference: on this machine TensorRT's fp16 engine is
// 3.65x PyTorch while ONNX Runtime's CUDA provider is slower than PyTorch, so CUDA is
// only ever a fallback for a machine where TensorRT will not install.
enum class Backend { kAuto, kTensorRt, kCuda, kCpu };

// Parse a backend name ("auto", "tensorrt", "cuda", "cpu"); returns false if unknown.
bool ParseBackend(const std::string& name, Backend* out);

const char* BackendName(Backend backend);

struct SegmenterOptions {
  Backend backend = Backend::kAuto;
  // Build the TensorRT engine in fp16. Costs about 0.013% of mask pixels against the
  // fp32 graph, which is far below the mask's own error against the road.
  bool fp16 = true;
  // Where TensorRT caches the built engine. The first build takes minutes; without a
  // cache that cost is paid on every process start, which is not what a vehicle does.
  // Empty disables caching.
  std::string engine_cache_dir;
  // Probability above which a pixel is lane. Applied to the logit as its logit-space
  // equivalent, so the sigmoid never has to be evaluated over the whole mask.
  float threshold = 0.5f;
  // Fraction of image height dropped as sky before the resize. Must match the value
  // the model was trained under.
  double sky_frac = 0.30;
  // 0 leaves ONNX Runtime's default thread count.
  int intra_op_threads = 0;
};

// Image layout of the caller's buffer. cv::imread and cv::VideoCapture both hand back
// BGR; the Python reference converts to RGB before normalizing, so the channel order
// has to be stated rather than assumed.
enum class PixelOrder { kBgr, kRgb };

// The network stage: image in, binary mask out.
//
// One session, one preallocated input tensor, one output buffer, all sized at
// construction. Nothing on the per-frame path allocates.
class Segmenter {
 public:
  // Loads model_path and builds a session for a width x height network input.
  // Throws std::runtime_error if the model will not load or the requested backend
  // cannot run the graph.
  Segmenter(const std::string& model_path, int width, int height,
            const SegmenterOptions& options);
  ~Segmenter();

  Segmenter(const Segmenter&) = delete;
  Segmenter& operator=(const Segmenter&) = delete;

  // Preprocess, run the network, and threshold into out_mask, which must hold
  // width*height bytes. src is an 8-bit 3-channel image at its native resolution,
  // row-major with src_stride bytes per row (0 meaning tightly packed).
  //
  // Returns false only if the source is empty or degenerate; a frame with no lane in
  // it is a successful call that writes an empty mask.
  bool Run(const std::uint8_t* src, int src_width, int src_height, int src_stride,
           PixelOrder order, std::uint8_t* out_mask);

  // The region of the source image the last Run consumed, as x, y, width, height.
  // Exposed so a caller drawing an overlay can put the mask back where it came from.
  void last_source_roi(int* x, int* y, int* width, int* height) const;

  // Microseconds spent in the last Run, split at the session boundary. The split is
  // the point: it says whether the remaining budget is in the network or in the
  // memcpy-and-normalize around it.
  double last_preprocess_us() const;
  double last_network_us() const;

  // The backend that actually built the session, which for kAuto is only known after
  // construction.
  Backend backend() const;

  int width() const;
  int height() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curvature_port

#endif  // CURVATURE_PORT_SEGMENTER_HPP
