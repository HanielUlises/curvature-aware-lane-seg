// Lateral MPC over a kinematic bicycle. The consumer the rest of this port feeds.
//
// C++ port of src/control/mpc.py. Linearized lateral error dynamics at speed v with
// wheelbase L, discretized with a forward Euler step dt:
//
//   x_{k+1} = A x_k + B u_k + d_k,
//   A = [[1, v dt], [0, 1]],  B = [0, v dt / L]^T,  d_k = [0, -v dt kappa_k]^T.
//
// Curvature enters as a known disturbance, so a curving lane is handled by feedforward
// rather than by waiting for tracking error to build. In steady state on a
// constant-curvature path the optimal steer is delta = L * kappa, the kinematic Ackermann
// relation, which the tests assert.
//
// The cost is quadratic and the dynamics linear, so stacking the horizon gives an
// unconstrained least-squares problem in the input sequence, solved in closed form and
// re-solved every step. Steering limits are applied by saturating the command, so they
// are respected at the plant but not accounted for inside the optimization; a genuinely
// constrained solve would need a QP.
//
// Sign conventions: internally the standard path-tracking one, cross-track positive when
// the vehicle is left of the path and steering positive to the left. The perception stack
// reports right-positive, so SteerForGeometry does the mapping and callers never handle
// it.
//
// See docs/geometry_port_spec.md, section 12.

#ifndef CURVATURE_PORT_MPC_HPP
#define CURVATURE_PORT_MPC_HPP

#include <Eigen/Dense>

#include <vector>

namespace curvature_port {

struct VehicleParams {
  double wheelbase_m = 2.7;
  double dt = 0.05;
  double max_steer_rad = 35.0 * 3.14159265358979323846 / 180.0;
};

struct MPCWeights {
  double cross_track = 1.0;
  double heading = 0.5;
  double steer = 0.05;
};

struct MPCSolution {
  double steer_rad = 0.0;              // saturated command for this step
  double steer_unsaturated_rad = 0.0;  // before the limit
  std::vector<double> planned_steer_rad;
  std::vector<double> predicted_states;  // 2N: [cross_track, heading] per step
  bool saturated = false;
  bool valid = false;  // false when speed was not positive
};

class KinematicLateralMPC {
 public:
  KinematicLateralMPC(VehicleParams params = {}, MPCWeights weights = {},
                      int horizon = 20);

  // Internal left-positive convention.
  MPCSolution Solve(double cross_track_m, double heading_rad, double curvature_1pm,
                    double speed_mps);

  // Measured right-positive convention, as the perception stack reports it.
  MPCSolution SteerForGeometry(double lateral_offset_m, double heading_error_rad,
                               double curvature_1pm, double speed_mps);

 private:
  // Stacks the horizon into X = Sx x0 + Su U + Sd D. Speed-dependent, so it is rebuilt
  // only when the speed changes, which on a vehicle is every step but costs little at
  // this horizon.
  void Condense(double speed);

  VehicleParams params_;
  MPCWeights weights_;
  int horizon_;
  double condensed_speed_ = -1.0;
  Eigen::MatrixXd sx_, su_, sd_;
  Eigen::MatrixXd hessian_;
  Eigen::LLT<Eigen::MatrixXd> hessian_llt_;
};

}  // namespace curvature_port

#endif  // CURVATURE_PORT_MPC_HPP
