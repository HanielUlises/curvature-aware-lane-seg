"""Tests for temporal filtering of the control quantities."""

from __future__ import annotations

import numpy as np

from src.geometry.road_geometry import RoadGeometry
from src.geometry.temporal import ConstantVelocityFilter, RoadGeometryFilter


def _geom(offset=0.0, heading=0.0, curvature=0.0):
    return RoadGeometry(
        ground_centerline=np.zeros((5, 2)),
        lateral_offset_m=offset,
        offset_distance_m=5.0,
        heading_error_rad=heading,
        curvature_1pm=curvature,
        preview_distances_m=np.array([5.0, 10.0, 20.0]),
        preview_curvature_1pm=np.full(3, curvature),
    )


def test_seeds_on_first_measurement():
    f = ConstantVelocityFilter()
    f.update(3.5)
    # No dragging up from zero: the first measurement is taken as the state.
    assert abs(f.value - 3.5) < 1e-12


def test_converges_on_a_constant_signal():
    f = ConstantVelocityFilter(process_var=1.0, measurement_var=0.25)
    for _ in range(60):
        f.update(2.0, dt=0.05)
    assert abs(f.value - 2.0) < 1e-3
    assert abs(f.rate) < 1e-2


def test_smooths_noise():
    rng = np.random.default_rng(0)
    truth = 1.0
    noisy = truth + rng.normal(0.0, 0.5, size=200)
    f = ConstantVelocityFilter(process_var=1.0, measurement_var=0.25, gate_sigma=None)
    out = [f.update(float(m), dt=0.05) or f.value for m in noisy]
    filtered = np.asarray(out[20:])
    raw = noisy[20:]
    # The filtered signal must sit closer to the truth than the raw measurements.
    assert filtered.std() < raw.std() / 2
    assert abs(filtered.mean() - truth) < 0.1


def test_tracks_a_ramp_with_bounded_lag():
    f = ConstantVelocityFilter(process_var=10.0, measurement_var=0.01)
    dt, slope = 0.05, 2.0  # metres per second
    for i in range(120):
        f.update(slope * i * dt, dt=dt)
    expected = slope * 119 * dt
    # A constant-velocity model should track a constant ramp with little lag.
    assert abs(f.value - expected) < 0.05
    assert abs(f.rate - slope) < 0.2


def test_gate_rejects_a_single_outlier():
    f = ConstantVelocityFilter(process_var=0.01, measurement_var=0.01, gate_sigma=3.0)
    for _ in range(40):
        f.update(1.0, dt=0.05)
    before = f.value
    accepted = f.update(50.0, dt=0.05)  # wildly inconsistent measurement
    assert accepted is False
    assert abs(f.value - before) < 0.05  # state barely moved


def test_predict_advances_without_measurement():
    f = ConstantVelocityFilter(process_var=1.0, measurement_var=0.01)
    dt = 0.1
    for i in range(40):
        f.update(1.0 * i * dt, dt=dt)  # ramp at 1 m/s
    value_before = f.value
    predicted = f.predict(dt)
    # Coasting must extrapolate along the estimated rate, not freeze.
    assert predicted > value_before
    assert abs(predicted - (value_before + f.rate * dt)) < 1e-9


def test_geometry_filter_smooths_all_three_quantities():
    rng = np.random.default_rng(1)
    filt = RoadGeometryFilter(dt=0.05)
    outputs = []
    for _ in range(120):
        outputs.append(filt.update(_geom(
            offset=0.5 + rng.normal(0, 0.3),
            heading=0.02 + rng.normal(0, 0.02),
            curvature=0.01 + rng.normal(0, 0.002),
        )))
    tail = outputs[40:]
    assert abs(np.mean([o.lateral_offset_m for o in tail]) - 0.5) < 0.15
    assert abs(np.mean([o.heading_error_rad for o in tail]) - 0.02) < 0.01
    assert abs(np.mean([o.curvature_1pm for o in tail]) - 0.01) < 0.005
    assert np.std([o.lateral_offset_m for o in tail]) < 0.3


def test_coasts_through_a_detection_gap():
    filt = RoadGeometryFilter(dt=0.05)
    for _ in range(40):
        filt.update(_geom(offset=1.0, heading=0.0, curvature=0.0))
    assert filt.coasting_frames == 0

    # Five frames with no ego lane: the estimate must persist, not vanish.
    for i in range(1, 6):
        out = filt.update(None)
        assert out.measured is False
        assert out.coasting_frames == i
        assert np.isfinite(out.lateral_offset_m)
        assert abs(out.lateral_offset_m - 1.0) < 0.2

    # A measurement clears the coast counter.
    out = filt.update(_geom(offset=1.0))
    assert out.measured is True and out.coasting_frames == 0


def test_reset_clears_state():
    filt = RoadGeometryFilter(dt=0.05)
    for _ in range(20):
        filt.update(_geom(offset=2.0))
    filt.update(None)
    filt.reset()
    assert filt.coasting_frames == 0
    out = filt.update(_geom(offset=-1.0))
    assert abs(out.lateral_offset_m + 1.0) < 1e-9  # seeded fresh
