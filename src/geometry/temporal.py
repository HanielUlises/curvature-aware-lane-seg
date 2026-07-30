"""Temporal filtering of the control quantities across frames.

Roadmap step five. Geometry is estimated independently per frame, so the signal handed
to a controller carries the full per-frame estimation noise and, worse, disappears
entirely on the frames where no ego lane is recovered. Neither is acceptable as a control
input: the first shows up as steering jitter, the second as a dropout.

Each quantity is tracked by a constant-velocity Kalman filter, which smooths in
proportion to how noisy the measurements have actually been and, on a frame with no
measurement, predicts forward instead of returning nothing. Two properties matter for the
controller:

- **Coasting.** A missed detection advances the state on the motion model alone, so the
  estimate degrades gracefully rather than vanishing. The filter reports how long it has
  been coasting so a supervisor can hand over before the extrapolation is trusted too far.
- **Gating.** A measurement far outside the predicted distribution is rejected rather than
  followed, which is what keeps a single badly misplaced centreline from throwing a step
  into the steering command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.road_geometry import RoadGeometry

FloatArray = np.ndarray


@dataclass
class ConstantVelocityFilter:
    """Scalar Kalman filter over ``[value, rate]`` with a constant-rate model.

    Args:
        process_var: Acceleration noise density driving the rate; larger follows
            manoeuvres faster and smooths less.
        measurement_var: Assumed variance of a single-frame measurement.
        gate_sigma: Reject a measurement further than this many predicted standard
            deviations from the prediction. ``None`` disables gating.
    """

    process_var: float = 1.0
    measurement_var: float = 1.0
    gate_sigma: float | None = 4.0
    _x: FloatArray = field(init=False)
    _p: FloatArray = field(init=False)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._x = np.zeros(2, dtype=np.float64)
        self._p = np.eye(2, dtype=np.float64) * 1e3
        self._initialized = False

    @property
    def value(self) -> float:
        """Current filtered estimate."""
        return float(self._x[0])

    @property
    def rate(self) -> float:
        """Current estimated rate of change per step."""
        return float(self._x[1])

    @property
    def variance(self) -> float:
        """Current variance of the value estimate."""
        return float(self._p[0, 0])

    def _transition(self, dt: float) -> tuple[FloatArray, FloatArray]:
        a = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        # Continuous white-noise acceleration discretized over dt.
        q = self.process_var * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )
        return a, q

    def predict(self, dt: float = 1.0) -> float:
        """Advance the state one step without a measurement.

        Args:
            dt: Time step in the same units the rate is expressed per.

        Returns:
            The predicted value.
        """
        a, q = self._transition(dt)
        self._x = a @ self._x
        self._p = a @ self._p @ a.T + q
        return self.value

    def update(self, measurement: float, dt: float = 1.0) -> bool:
        """Advance the state and fold in a measurement.

        Args:
            measurement: Observed value for this step.
            dt: Time step since the previous call.

        Returns:
            ``True`` if the measurement was accepted, ``False`` if it was gated out
            (the state still advanced on the motion model).
        """
        if not self._initialized:
            # Seed on the first measurement rather than dragging up from zero.
            self._x = np.array([float(measurement), 0.0], dtype=np.float64)
            self._p = np.diag([self.measurement_var, self.process_var])
            self._initialized = True
            return True

        self.predict(dt)
        residual = float(measurement) - self._x[0]
        innovation_var = self._p[0, 0] + self.measurement_var
        if self.gate_sigma is not None and innovation_var > 0.0:
            if abs(residual) > self.gate_sigma * np.sqrt(innovation_var):
                return False

        gain = self._p[:, 0] / innovation_var
        self._x = self._x + gain * residual
        self._p = self._p - np.outer(gain, self._p[0, :])
        return True


@dataclass
class FilteredGeometry:
    """Smoothed control quantities for one frame.

    Attributes:
        lateral_offset_m: Filtered lateral offset.
        heading_error_rad: Filtered heading error.
        curvature_1pm: Filtered signed curvature.
        measured: Whether this frame supplied a measurement.
        accepted: Whether that measurement passed the gate.
        coasting_frames: Consecutive frames without an accepted measurement.
    """

    lateral_offset_m: float
    heading_error_rad: float
    curvature_1pm: float
    measured: bool
    accepted: bool
    coasting_frames: int


class RoadGeometryFilter:
    """Track the three control quantities over a sequence of frames.

    Args:
        dt: Frame interval in seconds.
        offset_process_var: Acceleration noise for lateral offset, ``m^2/s^3``.
        heading_process_var: Acceleration noise for heading, ``rad^2/s^3``.
        curvature_process_var: Acceleration noise for curvature, ``m^-2 s^-3``.
        offset_measurement_var: Per-frame offset variance, ``m^2``.
        heading_measurement_var: Per-frame heading variance, ``rad^2``.
        curvature_measurement_var: Per-frame curvature variance, ``m^-2``.
        gate_sigma: Outlier gate in predicted standard deviations.
    """

    def __init__(
        self,
        dt: float = 0.05,
        offset_process_var: float = 4.0,
        heading_process_var: float = 1.0,
        curvature_process_var: float = 1e-3,
        offset_measurement_var: float = 0.25,
        heading_measurement_var: float = 0.01,
        curvature_measurement_var: float = 1e-4,
        gate_sigma: float | None = 4.0,
    ) -> None:
        self.dt = dt
        self.offset = ConstantVelocityFilter(
            offset_process_var, offset_measurement_var, gate_sigma
        )
        self.heading = ConstantVelocityFilter(
            heading_process_var, heading_measurement_var, gate_sigma
        )
        self.curvature = ConstantVelocityFilter(
            curvature_process_var, curvature_measurement_var, gate_sigma
        )
        self._coasting = 0

    @property
    def coasting_frames(self) -> int:
        """Consecutive frames without an accepted measurement."""
        return self._coasting

    def reset(self) -> None:
        """Clear all tracked state."""
        for f in (self.offset, self.heading, self.curvature):
            f.reset()
        self._coasting = 0

    def update(self, geometry: RoadGeometry | None) -> FilteredGeometry:
        """Fold one frame into the filters.

        Args:
            geometry: Per-frame geometry, or ``None`` when no ego lane was recovered.

        Returns:
            The smoothed :class:`FilteredGeometry` for this frame. When ``geometry`` is
            ``None`` the estimate is the motion-model prediction.
        """
        if geometry is None:
            for f in (self.offset, self.heading, self.curvature):
                f.predict(self.dt)
            self._coasting += 1
            return FilteredGeometry(
                lateral_offset_m=self.offset.value,
                heading_error_rad=self.heading.value,
                curvature_1pm=self.curvature.value,
                measured=False,
                accepted=False,
                coasting_frames=self._coasting,
            )

        accepted = [
            self.offset.update(geometry.lateral_offset_m, self.dt),
            self.heading.update(geometry.heading_error_rad, self.dt),
            self.curvature.update(geometry.curvature_1pm, self.dt),
        ]
        # A frame counts as tracked only if the offset, the quantity the controller is
        # most sensitive to, was actually accepted.
        if accepted[0]:
            self._coasting = 0
        else:
            self._coasting += 1
        return FilteredGeometry(
            lateral_offset_m=self.offset.value,
            heading_error_rad=self.heading.value,
            curvature_1pm=self.curvature.value,
            measured=True,
            accepted=all(accepted),
            coasting_frames=self._coasting,
        )
