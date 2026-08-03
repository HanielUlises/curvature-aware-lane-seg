// Checks the crop the network input is taken from against the Python reference
// (scripts/export_preprocess_vectors.py).
//
// Unlike the rest of the port's golden vectors there is no tolerance here. The region
// is integers all the way down, so the port either reproduces the reference's rounding
// or it does not, and "close" would mean the network is looking at a different piece of
// road than the one the model was trained on.

#include "curvature_port/preprocess.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int failures = 0;
int checked = 0;

void Check(int src_w, int src_h, double sky, int x, int y, int w, int h) {
  const auto got = curvature_port::NetworkInputRegion(src_w, src_h, 512, 288, sky);
  ++checked;
  if (got.x != x || got.y != y || got.width != w || got.height != h) {
    std::printf("FAIL %dx%d sky=%.2f: expected (%d, %d, %d, %d), got (%d, %d, %d, %d)\n",
                src_w, src_h, sky, x, y, w, h, got.x, got.y, got.width, got.height);
    ++failures;
  }
}

}  // namespace

int main() {
  const std::string path = std::string(GOLDEN_DIR) + "/preprocess.txt";
  std::ifstream in(path);
  if (!in) {
    std::printf("cannot open %s\n", path.c_str());
    return 1;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "target") {
      int tw = 0, th = 0;
      fields >> tw >> th;
      if (tw != 512 || th != 288) {
        std::printf("FAIL golden file is for a %dx%d input, this test assumes 512x288\n",
                    tw, th);
        return 1;
      }
    } else if (key == "region") {
      int src_w, src_h, x, y, w, h;
      double sky;
      if (!(fields >> src_w >> src_h >> sky >> x >> y >> w >> h)) {
        std::printf("FAIL malformed line: %s\n", line.c_str());
        return 1;
      }
      Check(src_w, src_h, sky, x, y, w, h);
    }
  }

  if (checked == 0) {
    std::printf("FAIL no regions in %s\n", path.c_str());
    return 1;
  }

  // The degenerate arguments a caller can reach by passing a bad config, which must
  // produce an empty region rather than a negative-sized one that then indexes a Mat.
  const int bad_cases[][2] = {{0, 720}, {1280, 0}, {-4, 100}};
  for (const auto& bad : bad_cases) {
    const auto r = curvature_port::NetworkInputRegion(bad[0], bad[1], 512, 288, 0.30);
    if (r.width != 0 || r.height != 0) {
      std::printf("FAIL %dx%d should give an empty region, got %dx%d\n", bad[0], bad[1],
                  r.width, r.height);
      ++failures;
    }
  }
  for (double sky : {1.0, 1.5, -0.1}) {
    const auto r = curvature_port::NetworkInputRegion(1280, 720, 512, 288, sky);
    if (r.width != 0 || r.height != 0) {
      std::printf("FAIL sky_frac %.2f should give an empty region, got %dx%d\n", sky,
                  r.width, r.height);
      ++failures;
    }
  }

  std::printf("%s: %d regions checked, %d failures\n", failures == 0 ? "PASS" : "FAIL",
              checked, failures);
  return failures == 0 ? 0 : 1;
}
