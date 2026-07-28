// Validates the C++ curvature port against the shared golden vectors exported
// from the Python reference (scripts/export_golden_vectors.py). Reads the flat
// fixtures under GOLDEN_DIR (set by CMake), no JSON dependency.

#include "curvature_port/curvature.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
  std::string name;
  double percentile = 90.0;
  int num_samples = 100;
  double expected_analytic = std::nan("");  // nan => compare to python_p90
  double python_p90 = 0.0;
  double tolerance = 0.05;
  std::string compare = "python";
  std::vector<curvature_port::Point> points;
};

double ParseDouble(const std::string& token) {
  if (token == "nan") return std::nan("");
  return std::stod(token);
}

Case LoadCase(const std::string& path) {
  std::ifstream in(path);
  Case c;
  std::string key;
  int num_points = 0;
  while (in >> key) {
    if (key == "name") in >> c.name;
    else if (key == "percentile") { std::string v; in >> v; c.percentile = ParseDouble(v); }
    else if (key == "num_samples") in >> c.num_samples;
    else if (key == "expected_analytic") { std::string v; in >> v; c.expected_analytic = ParseDouble(v); }
    else if (key == "python_p90") { std::string v; in >> v; c.python_p90 = ParseDouble(v); }
    else if (key == "tolerance") { std::string v; in >> v; c.tolerance = ParseDouble(v); }
    else if (key == "compare") in >> c.compare;
    else if (key == "points") {
      in >> num_points;
      for (int i = 0; i < num_points; ++i) {
        curvature_port::Point p;
        in >> p.x >> p.y;
        c.points.push_back(p);
      }
    }
  }
  return c;
}

}  // namespace

int main() {
  const std::string dir = GOLDEN_DIR;
  std::ifstream index(dir + "/index.txt");
  if (!index) {
    std::cerr << "cannot open " << dir << "/index.txt "
              << "(run: python -m scripts.export_golden_vectors)\n";
    return 2;
  }

  int failures = 0;
  int total = 0;
  std::string name;
  while (index >> name) {
    ++total;
    const Case c = LoadCase(dir + "/" + name + ".txt");
    const double got = curvature_port::LaneCurvature(c.points, c.percentile, c.num_samples);

    // Assert against analytic truth where known, else the Python reference.
    const bool use_analytic = (c.compare == "analytic") && !std::isnan(c.expected_analytic);
    const double target = use_analytic ? c.expected_analytic : c.python_p90;

    bool ok;
    double err;
    if (std::abs(target) < 1e-9) {
      err = std::abs(got);
      ok = err < std::max(c.tolerance, 1e-3);
    } else {
      err = std::abs(got - target) / std::abs(target);
      ok = err <= c.tolerance;
    }

    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name
              << "  got=" << got << "  target=" << target
              << " (" << (use_analytic ? "analytic" : "python") << ")"
              << "  rel_err=" << err << "  tol=" << c.tolerance << "\n";
    if (!ok) ++failures;
  }

  std::cout << "\n" << (total - failures) << "/" << total << " golden cases passed\n";
  return failures == 0 ? 0 : 1;
}
