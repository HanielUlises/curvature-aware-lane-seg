// Where the network input comes from in the source frame.
//
// This is the one piece of the segmenter that is not about the network, and the one
// piece that has to be exactly right. The crop is aspect-preserving precisely so the
// resize after it is isotropic and the curvature the whole project keys on survives it;
// an off-by-one in the region tilts the ground plane the calibration was fitted against
// and every metre of lateral offset downstream is wrong by a little.
//
// So it lives here, apart from the ONNX Runtime and OpenCV code that surrounds it,
// where it costs nothing to link and can be checked against the Python reference by the
// same golden-vector contract as the rest of the port.
//
// Port of scripts/infer_sequence._center_crop_aspect composed with
// src/data/transforms.preprocess_geometry. Both are pure crops, so the composition is a
// single region and the resize is applied once, to it.

#ifndef CURVATURE_PORT_PREPROCESS_HPP
#define CURVATURE_PORT_PREPROCESS_HPP

namespace curvature_port {

// A region of the source image, in pixels.
struct SourceRegion {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

// The region a target_width x target_height network input is resized from, after
// centre-cropping to the target aspect and dropping the top sky_frac of what remains.
//
// Returns a zero-sized region if the arguments cannot produce one.
SourceRegion NetworkInputRegion(int src_width, int src_height, int target_width,
                                int target_height, double sky_frac);

}  // namespace curvature_port

#endif  // CURVATURE_PORT_PREPROCESS_HPP
