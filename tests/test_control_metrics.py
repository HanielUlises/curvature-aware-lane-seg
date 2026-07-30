"""Tests for control-relevant, curvature-stratified error metrics."""

from __future__ import annotations

import numpy as np

from src.eval.control_metrics import (
    StratifiedControlMetric,
    control_errors,
)
from src.geometry.road_geometry import RoadGeometry

EDGES = [0.0, 1.0, 10.0, float("inf")]


def _geom(offset, heading, previews):
    return RoadGeometry(
        ground_centerline=np.zeros((5, 2)),
        lateral_offset_m=offset,
        offset_distance_m=5.0,
        heading_error_rad=heading,
        curvature_1pm=0.0,
        preview_distances_m=np.array([5.0, 10.0, 20.0]),
        preview_curvature_1pm=np.asarray(previews, dtype=float),
    )


def test_errors_are_absolute_differences():
    pred = _geom(0.5, 0.10, [0.02, 0.03, 0.04])
    truth = _geom(0.2, 0.04, [0.01, 0.05, 0.04])
    e = control_errors(pred, truth)
    assert abs(e.lateral_offset_err_m - 0.3) < 1e-9
    assert abs(e.heading_err_rad - 0.06) < 1e-9
    np.testing.assert_allclose(e.curvature_err_1pm, [0.01, 0.02, 0.0], atol=1e-9)


def test_identical_geometry_gives_zero_error():
    g = _geom(0.4, 0.05, [0.01, 0.02, 0.03])
    e = control_errors(g, g)
    assert e.lateral_offset_err_m == 0.0
    assert e.heading_err_rad == 0.0
    assert np.all(e.curvature_err_1pm == 0.0)


def test_nan_preview_propagates_and_is_excluded_from_mean():
    pred = _geom(0.0, 0.0, [0.02, np.nan, 0.01])
    truth = _geom(0.0, 0.0, [0.01, 0.05, np.nan])
    e = control_errors(pred, truth)
    assert abs(e.curvature_err_1pm[0] - 0.01) < 1e-9
    assert np.isnan(e.curvature_err_1pm[1]) and np.isnan(e.curvature_err_1pm[2])

    m = StratifiedControlMetric(EDGES)
    m.update(e, frame_kappa=0.5)
    overall, _ = m.compute()
    # Only the first preview had a value on both sides.
    assert abs(overall.curvature_mae_1pm[0] - 0.01) < 1e-9
    assert np.isnan(overall.curvature_mae_1pm[1])


def test_stratification_routes_frames_to_bins():
    m = StratifiedControlMetric(EDGES)
    easy = control_errors(_geom(0.1, 0.0, [0.0, 0.0, 0.0]), _geom(0.0, 0.0, [0.0, 0.0, 0.0]))
    hard = control_errors(_geom(1.1, 0.0, [0.0, 0.0, 0.0]), _geom(0.0, 0.0, [0.0, 0.0, 0.0]))
    m.update(easy, frame_kappa=0.2)   # bin 0
    m.update(hard, frame_kappa=50.0)  # bin 2
    overall, per_bin = m.compute()
    assert abs(per_bin[0].offset_mae_m - 0.1) < 1e-9
    assert abs(per_bin[2].offset_mae_m - 1.1) < 1e-9
    assert np.isnan(per_bin[1].offset_mae_m)  # untouched bin
    assert per_bin[1].detected == 0
    assert abs(overall.offset_mae_m - 0.6) < 1e-9  # pooled mean of 0.1 and 1.1


def test_detection_failures_are_counted_not_dropped():
    m = StratifiedControlMetric(EDGES)
    ok = control_errors(_geom(0.1, 0.0, [0.0, 0.0, 0.0]), _geom(0.0, 0.0, [0.0, 0.0, 0.0]))
    m.update(ok, frame_kappa=0.2)
    m.update_failure(frame_kappa=0.2)
    m.update_failure(frame_kappa=0.2)
    overall, per_bin = m.compute()
    assert per_bin[0].detected == 1 and per_bin[0].failed == 2
    assert abs(per_bin[0].detection_rate - 1 / 3) < 1e-9
    assert abs(overall.detection_rate - 1 / 3) < 1e-9


def test_heading_reported_in_degrees():
    m = StratifiedControlMetric(EDGES)
    e = control_errors(_geom(0.0, np.radians(3.0), [0.0, 0.0, 0.0]),
                       _geom(0.0, 0.0, [0.0, 0.0, 0.0]))
    m.update(e, frame_kappa=0.2)
    overall, _ = m.compute()
    assert abs(overall.heading_mae_deg - 3.0) < 1e-6


def test_reset_clears_state():
    m = StratifiedControlMetric(EDGES)
    e = control_errors(_geom(0.5, 0.0, [0.0, 0.0, 0.0]), _geom(0.0, 0.0, [0.0, 0.0, 0.0]))
    m.update(e, frame_kappa=0.2)
    m.update_failure(frame_kappa=0.2)
    m.reset()
    overall, _ = m.compute()
    assert overall.detected == 0 and overall.failed == 0
    assert np.isnan(overall.offset_mae_m)
