// Validates the projection and road-geometry ports against the shared golden
// vectors exported from the Python reference (scripts/export_geometry_vectors.py).
// Reads the flat fixtures under GOLDEN_DIR/geometry, no JSON dependency.

#include "curvature_port/ipm.hpp"
#include "curvature_port/road_geometry.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using curvature_port::Point;

double ParseDouble(const std::string& token) {
  if (token == "nan") return std::nan("");
  return std::stod(token);
}

double ReadDouble(std::istream& in) {
  std::string token;
  in >> token;
  return ParseDouble(token);
}

std::vector<Point> ReadPoints(std::istream& in, int count) {
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const double x = ReadDouble(in);
    const double y = ReadDouble(in);
    pts.push_back({x, y});
  }
  return pts;
}

// Absolute-or-relative agreement: absolute near zero, relative for large values.
bool Close(double got, double want, double tol) {
  if (std::isnan(want)) return std::isnan(got);
  if (std::isnan(got)) return false;
  const double scale = std::max(1.0, std::abs(want));
  return std::abs(got - want) <= tol * scale;
}

int failures = 0;

void Check(bool ok, const std::string& label, double got, double want) {
  if (ok) return;
  ++failures;
  std::cout << "    [FAIL] " << label << "  got=" << got << "  want=" << want << "\n";
}

struct ProjectionCase {
  std::string name;
  double tolerance = 1e-9;
  std::vector<Point> src, dst;
  curvature_port::Matrix3 expected_h = curvature_port::Matrix3::Identity();
  std::vector<Point> probe, expected_ground;
};

ProjectionCase LoadProjection(const std::string& path) {
  std::ifstream in(path);
  ProjectionCase c;
  std::string key;
  int n = 0;
  while (in >> key) {
    if (key == "name") in >> c.name;
    else if (key == "tolerance") c.tolerance = ReadDouble(in);
    else if (key == "src") { in >> n; c.src = ReadPoints(in, n); }
    else if (key == "dst") { in >> n; c.dst = ReadPoints(in, n); }
    else if (key == "expected_h") {
      in >> n;
      for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col) c.expected_h(r, col) = ReadDouble(in);
    } else if (key == "probe") {
      in >> n;
      for (int i = 0; i < n; ++i) {
        const double u = ReadDouble(in), v = ReadDouble(in);
        const double x = ReadDouble(in), z = ReadDouble(in);
        c.probe.push_back({u, v});
        c.expected_ground.push_back({x, z});
      }
    }
  }
  return c;
}

void RunProjection(const ProjectionCase& c) {
  std::cout << "[projection] " << c.name << "\n";
  bool ok = false;
  const curvature_port::Matrix3 h =
      curvature_port::HomographyFromPoints(c.src, c.dst, &ok);
  if (!ok) {
    ++failures;
    std::cout << "    [FAIL] homography solve reported failure\n";
    return;
  }
  for (int r = 0; r < 3; ++r) {
    for (int col = 0; col < 3; ++col) {
      Check(Close(h(r, col), c.expected_h(r, col), c.tolerance),
            "H(" + std::to_string(r) + "," + std::to_string(col) + ")",
            h(r, col), c.expected_h(r, col));
    }
  }

  const curvature_port::GroundPlane plane(h);
  const std::vector<Point> got = plane.ToGround(c.probe);
  for (std::size_t i = 0; i < got.size(); ++i) {
    Check(Close(got[i].x, c.expected_ground[i].x, c.tolerance),
          "ground[" + std::to_string(i) + "].x", got[i].x, c.expected_ground[i].x);
    Check(Close(got[i].y, c.expected_ground[i].y, c.tolerance),
          "ground[" + std::to_string(i) + "].z", got[i].y, c.expected_ground[i].y);
  }
  // The inverse must undo the forward map: this is what draws BEV results back on
  // the camera frame, so a silent inversion error would be visible only as a
  // mis-drawn overlay.
  const std::vector<Point> back = plane.ToImage(got);
  for (std::size_t i = 0; i < back.size(); ++i) {
    Check(Close(back[i].x, c.probe[i].x, 1e-6), "roundtrip[" + std::to_string(i) + "].u",
          back[i].x, c.probe[i].x);
    Check(Close(back[i].y, c.probe[i].y, 1e-6), "roundtrip[" + std::to_string(i) + "].v",
          back[i].y, c.probe[i].y);
  }
}

struct GeometryCase {
  std::string name;
  double tolerance = 1e-6;
  int num_samples = 100;
  double offset_distance = 5.0;
  double expected_offset = 0.0, expected_heading = 0.0, expected_curvature = 0.0;
  bool has_homography = false;
  curvature_port::Matrix3 homography = curvature_port::Matrix3::Identity();
  std::vector<double> previews, expected_preview_kappa;
  std::vector<Point> points;
};

GeometryCase LoadGeometry(const std::string& path) {
  std::ifstream in(path);
  GeometryCase c;
  std::string key;
  int n = 0;
  while (in >> key) {
    if (key == "name") in >> c.name;
    else if (key == "tolerance") c.tolerance = ReadDouble(in);
    else if (key == "num_samples") in >> c.num_samples;
    else if (key == "offset_distance") c.offset_distance = ReadDouble(in);
    else if (key == "expected_offset") c.expected_offset = ReadDouble(in);
    else if (key == "expected_heading") c.expected_heading = ReadDouble(in);
    else if (key == "expected_curvature") c.expected_curvature = ReadDouble(in);
    else if (key == "homography") {
      in >> n;
      c.has_homography = n == 3;
      for (int r = 0; r < n; ++r)
        for (int col = 0; col < 3; ++col) c.homography(r, col) = ReadDouble(in);
    } else if (key == "previews") {
      in >> n;
      for (int i = 0; i < n; ++i) {
        c.previews.push_back(ReadDouble(in));
        c.expected_preview_kappa.push_back(ReadDouble(in));
      }
    } else if (key == "points") {
      in >> n;
      c.points = ReadPoints(in, n);
    }
  }
  return c;
}

void RunGeometry(const GeometryCase& c) {
  std::cout << "[geometry]   " << c.name << "\n";
  std::vector<Point> ground = c.points;
  if (c.has_homography) ground = curvature_port::ApplyHomography(c.homography, ground);

  const curvature_port::RoadGeometry rg = curvature_port::ReadRoadGeometry(
      ground, c.previews, c.num_samples, c.offset_distance);
  if (!rg.valid) {
    ++failures;
    std::cout << "    [FAIL] geometry reported invalid\n";
    return;
  }
  Check(Close(rg.lateral_offset_m, c.expected_offset, c.tolerance), "offset_m",
        rg.lateral_offset_m, c.expected_offset);
  Check(Close(rg.heading_error_rad, c.expected_heading, c.tolerance), "heading_rad",
        rg.heading_error_rad, c.expected_heading);
  Check(Close(rg.curvature_1pm, c.expected_curvature, c.tolerance), "curvature_1pm",
        rg.curvature_1pm, c.expected_curvature);
  for (std::size_t i = 0; i < c.previews.size(); ++i) {
    Check(Close(rg.preview_curvature_1pm[i], c.expected_preview_kappa[i], c.tolerance),
          "preview_kappa@" + std::to_string(static_cast<int>(c.previews[i])) + "m",
          rg.preview_curvature_1pm[i], c.expected_preview_kappa[i]);
  }
}

std::vector<std::string> ReadIndex(const std::string& path) {
  std::vector<std::string> names;
  std::ifstream in(path);
  std::string name;
  while (in >> name) names.push_back(name);
  return names;
}

}  // namespace

int main() {
  const std::string dir = std::string(GOLDEN_DIR) + "/geometry";
  const std::vector<std::string> projections = ReadIndex(dir + "/index_projection.txt");
  const std::vector<std::string> geometries = ReadIndex(dir + "/index_geometry.txt");
  if (projections.empty() || geometries.empty()) {
    std::cerr << "no fixtures under " << dir
              << " (run: python -m scripts.export_geometry_vectors)\n";
    return 2;
  }

  for (const std::string& name : projections) RunProjection(LoadProjection(dir + "/" + name + ".txt"));
  for (const std::string& name : geometries) RunGeometry(LoadGeometry(dir + "/" + name + ".txt"));

  const std::size_t total = projections.size() + geometries.size();
  std::cout << "\n" << (failures == 0 ? "all " : "") << total
            << " geometry golden cases run, " << failures << " assertion failure(s)\n";
  return failures == 0 ? 0 : 1;
}
