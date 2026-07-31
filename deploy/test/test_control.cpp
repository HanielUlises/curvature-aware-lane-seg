// Validates the temporal filter and the lateral MPC against the golden vectors exported
// from the Python reference (scripts/export_control_vectors.py).
//
// The filter is checked over a whole sequence rather than a step, because its failure
// modes are stateful: a port that mishandles gating or coasting agrees on step one and
// diverges by step twenty. The controller is checked case by case, and additionally
// against the closed-form Ackermann steer, which holds regardless of any fixture.

#include "curvature_port/mpc.hpp"
#include "curvature_port/temporal.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what, double got, double want) {
  if (ok) return;
  ++failures;
  std::cout << "[FAIL] " << what << "  got=" << got << "  want=" << want << "\n";
}

bool Close(double got, double want, double tol) {
  return std::abs(got - want) <= tol * std::max(1.0, std::abs(want));
}

void RunFilter(const std::string& dir) {
  std::ifstream in(dir + "/filter.txt");
  if (!in) {
    std::cerr << "cannot read " << dir << "/filter.txt\n";
    ++failures;
    return;
  }
  std::string key;
  double dt = 0.05, tol = 1e-9;
  int steps = 0;
  in >> key >> dt >> key >> tol >> key >> steps;

  curvature_port::FilterConfig cfg;
  cfg.dt = dt;
  curvature_port::RoadGeometryFilter filt(cfg);

  int coasted = 0, gated = 0;
  for (int k = 0; k < steps; ++k) {
    std::string tag;
    in >> tag;  // "m"
    std::string first;
    in >> first;
    bool has = first != "none";
    double off = 0.0, head = 0.0, kap = 0.0;
    if (has) {
      off = std::stod(first);
      in >> head >> kap;
    }
    const curvature_port::FilteredGeometry got = filt.Update(has, off, head, kap);

    std::string otag;
    double w_off, w_head, w_kap;
    int w_meas, w_acc, w_coast;
    in >> otag >> w_off >> w_head >> w_kap >> w_meas >> w_acc >> w_coast;

    const std::string at = "filter step " + std::to_string(k);
    Check(Close(got.lateral_offset_m, w_off, tol), at + " offset",
          got.lateral_offset_m, w_off);
    Check(Close(got.heading_error_rad, w_head, tol), at + " heading",
          got.heading_error_rad, w_head);
    Check(Close(got.curvature_1pm, w_kap, tol), at + " curvature",
          got.curvature_1pm, w_kap);
    Check(got.measured == (w_meas != 0), at + " measured", got.measured, w_meas);
    Check(got.accepted == (w_acc != 0), at + " accepted", got.accepted, w_acc);
    Check(got.coasting_frames == w_coast, at + " coasting", got.coasting_frames,
          w_coast);
    if (!got.measured) ++coasted;
    if (got.measured && !got.accepted) ++gated;
  }
  std::cout << "[filter] " << steps << " steps, " << coasted << " coasted, " << gated
            << " gated\n";
}

void RunMPC(const std::string& dir) {
  std::ifstream in(dir + "/mpc.txt");
  if (!in) {
    std::cerr << "cannot read " << dir << "/mpc.txt\n";
    ++failures;
    return;
  }
  std::string key;
  double wheelbase, dt, max_steer;
  int horizon, n;
  in >> key >> wheelbase >> key >> dt >> key >> max_steer >> key >> horizon >> key >> n;

  curvature_port::VehicleParams params;
  params.wheelbase_m = wheelbase;
  params.dt = dt;
  params.max_steer_rad = max_steer;
  curvature_port::KinematicLateralMPC mpc(params, {}, horizon);

  for (int i = 0; i < n; ++i) {
    std::string tag, name;
    in >> tag >> name;
    double off, head, kap, speed, w_steer, w_raw, w_tol;
    int w_sat;
    in >> off >> head >> kap >> speed >> w_steer >> w_raw >> w_sat >> w_tol;

    const curvature_port::MPCSolution s =
        mpc.SteerForGeometry(off, head, kap, speed);
    Check(s.valid, "mpc " + name + " valid", s.valid, 1);
    Check(Close(s.steer_rad, w_steer, w_tol), "mpc " + name + " steer", s.steer_rad,
          w_steer);
    Check(Close(s.steer_unsaturated_rad, w_raw, w_tol), "mpc " + name + " unsaturated",
          s.steer_unsaturated_rad, w_raw);
    Check(s.saturated == (w_sat != 0), "mpc " + name + " saturated", s.saturated, w_sat);
  }
  std::cout << "[mpc] " << n << " cases\n";

  // Closed form, independent of any fixture: on a constant-curvature path with no
  // tracking error, the steady-state steer is the Ackermann value L * kappa.
  for (double kappa : {0.005, 0.01, 0.02, -0.015}) {
    const curvature_port::MPCSolution s = mpc.SteerForGeometry(0.0, 0.0, kappa, 15.0);
    const double want = wheelbase * kappa;
    Check(Close(s.steer_unsaturated_rad, want, 1e-6),
          "ackermann at kappa=" + std::to_string(kappa), s.steer_unsaturated_rad, want);
  }
  std::cout << "[mpc] Ackermann relation holds at four curvatures\n";

  // A standstill has no lateral dynamics, so the solve must decline rather than divide
  // through by a zero input matrix.
  const curvature_port::MPCSolution stopped = mpc.SteerForGeometry(1.0, 0.1, 0.0, 0.0);
  Check(!stopped.valid, "zero speed declines", stopped.valid, 0);
}

}  // namespace

int main() {
  const std::string dir = std::string(GOLDEN_DIR) + "/control";
  RunFilter(dir);
  RunMPC(dir);
  if (failures == 0) std::cout << "\nall control golden cases match the reference\n";
  return failures == 0 ? 0 : 1;
}
