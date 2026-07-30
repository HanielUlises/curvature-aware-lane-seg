"""Lateral MPC over a kinematic bicycle model.

Roadmap step six, and the consumer the rest of the repository was built for. The
controller tracks the lane centreline using the three quantities the perception stack
produces: lateral offset, heading error and previewed curvature.

## Model

Linearized lateral error dynamics of a kinematic bicycle at speed ``v`` with wheelbase
``L``, in the standard path-tracking convention where the cross-track error ``e`` is
positive when the **vehicle is left of the path**, the heading error ``psi`` is positive
when the vehicle points left of the path tangent, steering ``delta`` is positive to the
**left**, and path curvature ``kappa`` is positive for a **left** turn:

```math
\\dot{e} = v \\sin\\psi \\approx v\\,\\psi, \\qquad
\\dot{\\psi} = \\frac{v}{L}\\tan\\delta - v\\,\\kappa \\approx \\frac{v}{L}\\delta - v\\,\\kappa.
```

Discretized with a forward Euler step ``dt``:

```math
x_{k+1} = A x_k + B u_k + d_k, \\quad
A = \\begin{bmatrix} 1 & v\\,dt \\\\ 0 & 1 \\end{bmatrix}, \\;
B = \\begin{bmatrix} 0 \\\\ v\\,dt/L \\end{bmatrix}, \\;
d_k = \\begin{bmatrix} 0 \\\\ -v\\,dt\\,\\kappa_k \\end{bmatrix}.
```

Curvature enters as a known disturbance, so a curving lane is handled by feedforward
rather than by waiting for tracking error to build up. In steady state on a constant-
curvature path the optimal steer is ``delta = L * kappa``, which is the kinematic
Ackermann relation and is asserted in the tests.

## What this solves, precisely

The cost is quadratic and the dynamics linear, so stacking the horizon gives an
unconstrained least-squares problem in the input sequence with a closed-form solution,
re-solved every step on the current state. That is a receding-horizon linear-quadratic
MPC. Steering limits are applied by saturating the command rather than by solving a
constrained program, so the limits are respected at the plant but are not accounted for
inside the optimization; a genuinely constrained solve would need a QP.

## Measured quantities use the opposite sign convention

:class:`src.geometry.road_geometry.RoadGeometry` reports offset, heading and curvature
**right-positive**, because that is the natural reading of an image. The mapping into the
convention above is ``e = offset``, ``psi = heading``, ``kappa = -curvature``, and the
returned steering angle is negated back to right-positive.
:meth:`KinematicLateralMPC.steer_for_geometry` does this, so callers never handle it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass(frozen=True)
class VehicleParams:
    """Kinematic bicycle parameters.

    Attributes:
        wheelbase_m: Distance between axles.
        dt: Control interval in seconds.
        max_steer_rad: Steering magnitude limit applied to the command.
    """

    wheelbase_m: float = 2.7
    dt: float = 0.05
    max_steer_rad: float = math.radians(35.0)


@dataclass(frozen=True)
class MPCWeights:
    """Quadratic cost weights.

    Attributes:
        cross_track: Penalty on cross-track error, per ``m^2``.
        heading: Penalty on heading error, per ``rad^2``.
        steer: Penalty on steering effort, per ``rad^2``.
    """

    cross_track: float = 1.0
    heading: float = 0.5
    steer: float = 0.05


@dataclass(frozen=True)
class MPCSolution:
    """Result of one solve.

    Attributes:
        steer_rad: Steering command for this step, right-positive, saturated.
        steer_unsaturated_rad: The same command before the limit was applied.
        planned_steer_rad: The whole planned input sequence, left-positive internal sign.
        predicted_states: ``(N, 2)`` predicted ``[cross_track, heading]`` trajectory.
        saturated: Whether the limit was active.
    """

    steer_rad: float
    steer_unsaturated_rad: float
    planned_steer_rad: FloatArray
    predicted_states: FloatArray
    saturated: bool


class KinematicLateralMPC:
    """Receding-horizon linear-quadratic lateral controller.

    Args:
        params: Vehicle and timing parameters.
        weights: Quadratic cost weights.
        horizon: Number of steps in the prediction horizon.
    """

    def __init__(
        self,
        params: VehicleParams | None = None,
        weights: MPCWeights | None = None,
        horizon: int = 20,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.params = params or VehicleParams()
        self.weights = weights or MPCWeights()
        self.horizon = horizon

    def _condense(self, speed: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Stack the horizon into ``X = Sx x0 + Su U + Sd D`` and return the factors."""
        dt, wheelbase = self.params.dt, self.params.wheelbase_m
        a = np.array([[1.0, speed * dt], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[0.0], [speed * dt / wheelbase]], dtype=np.float64)
        n = self.horizon

        sx = np.zeros((2 * n, 2))
        su = np.zeros((2 * n, n))
        sd = np.zeros((2 * n, 2 * n))
        a_pow = np.eye(2)
        for i in range(n):
            a_pow = a_pow @ a if i else a.copy()
            sx[2 * i : 2 * i + 2, :] = a_pow
            for j in range(i + 1):
                # A^(i-j) B multiplies input j; the same power multiplies disturbance j.
                a_k = np.linalg.matrix_power(a, i - j)
                su[2 * i : 2 * i + 2, j : j + 1] = a_k @ b
                sd[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = a_k
        return sx, su, sd

    def solve(
        self,
        cross_track_m: float,
        heading_rad: float,
        curvature_1pm: float,
        speed_mps: float,
    ) -> MPCSolution:
        """Solve one step in the internal left-positive convention.

        Args:
            cross_track_m: Cross-track error, positive when left of the path.
            heading_rad: Heading error, positive when pointing left of the tangent.
            curvature_1pm: Path curvature, positive for a left turn, held over the
                horizon.
            speed_mps: Forward speed. Must be positive; the lateral dynamics vanish at
                a standstill.

        Returns:
            The :class:`MPCSolution`. ``steer_rad`` is left-positive here; use
            :meth:`steer_for_geometry` for the measured right-positive convention.

        Raises:
            ValueError: If ``speed_mps`` is not positive.
        """
        if speed_mps <= 0.0:
            raise ValueError(f"speed_mps must be positive, got {speed_mps}")

        n = self.horizon
        sx, su, sd = self._condense(speed_mps)
        x0 = np.array([cross_track_m, heading_rad], dtype=np.float64)
        disturbance = np.tile(
            np.array([0.0, -speed_mps * self.params.dt * curvature_1pm]), n
        )

        q = np.diag([self.weights.cross_track, self.weights.heading])
        q_bar = np.kron(np.eye(n), q)
        r_bar = np.eye(n) * self.weights.steer

        free = sx @ x0 + sd @ disturbance
        hessian = su.T @ q_bar @ su + r_bar
        gradient = su.T @ q_bar @ free
        planned = np.linalg.solve(hessian, -gradient)

        states = (free + su @ planned).reshape(n, 2)
        raw = float(planned[0])
        limit = self.params.max_steer_rad
        clipped = float(np.clip(raw, -limit, limit))
        return MPCSolution(
            steer_rad=clipped,
            steer_unsaturated_rad=raw,
            planned_steer_rad=planned,
            predicted_states=states,
            saturated=clipped != raw,
        )

    def steer_for_geometry(
        self,
        lateral_offset_m: float,
        heading_error_rad: float,
        curvature_1pm: float,
        speed_mps: float,
    ) -> MPCSolution:
        """Solve from measured, right-positive geometry.

        Args:
            lateral_offset_m: As reported by the perception stack, positive when the
                lane centre lies to the right of the vehicle.
            heading_error_rad: Centreline bearing, positive to the right.
            curvature_1pm: Signed curvature, positive for a right turn.
            speed_mps: Forward speed.

        Returns:
            The solution with ``steer_rad`` in the measured convention, positive to
            steer right.
        """
        solution = self.solve(
            cross_track_m=lateral_offset_m,
            heading_rad=heading_error_rad,
            curvature_1pm=-curvature_1pm,
            speed_mps=speed_mps,
        )
        limit = self.params.max_steer_rad
        raw = -solution.steer_unsaturated_rad
        return MPCSolution(
            steer_rad=float(np.clip(raw, -limit, limit)),
            steer_unsaturated_rad=raw,
            planned_steer_rad=solution.planned_steer_rad,
            predicted_states=solution.predicted_states,
            saturated=solution.saturated,
        )
