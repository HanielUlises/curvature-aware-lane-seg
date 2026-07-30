"""Tests for the portable cubic-spline reference the C++ port mirrors."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.curvature import signed_curvature_along
from src.geometry.curvature_portable import (
    curvature_along_portable,
    lane_curvature_portable,
    sample_positions_portable,
    signed_curvature_along_portable,
)


def _arc(radius: float, right: bool, theta_max: float = 0.8, n: int = 40) -> np.ndarray:
    theta = np.linspace(0.0, theta_max, n)
    x = radius - radius * np.cos(theta)
    return np.column_stack([x if right else -x, radius * np.sin(theta)])


def test_circle_curvature_matches_inverse_radius_everywhere():
    # Not just at the p90 summary: with not-a-knot ends, every sample along a
    # circular arc must sit at 1/R, including the first and last.
    kappa = curvature_along_portable(_arc(60.0, right=True), num_samples=50)
    assert kappa == pytest.approx(1.0 / 60.0, rel=2e-3)


def test_signed_curvature_distinguishes_turn_direction():
    right = signed_curvature_along_portable(_arc(50.0, right=True), num_samples=25)
    left = signed_curvature_along_portable(_arc(50.0, right=False), num_samples=25)
    # In (x right, z ahead) axes a right-hand turn is clockwise, hence negative here;
    # the sign flip to "right positive" happens in the road-geometry stage.
    assert np.all(right < 0) and np.all(left > 0)
    assert np.abs(right) == pytest.approx(np.abs(left), rel=1e-9)


def test_signed_magnitude_matches_unsigned():
    pts = _arc(75.0, right=True)
    assert np.abs(signed_curvature_along_portable(pts, 40)) == pytest.approx(
        curvature_along_portable(pts, 40), rel=1e-12
    )


def test_signed_convention_agrees_with_the_fitpack_reference():
    pts = _arc(90.0, right=False)
    portable = signed_curvature_along_portable(pts, 30)
    fitpack = signed_curvature_along(pts, num_samples=30, smoothing=0.0)
    assert np.sign(portable) == pytest.approx(np.sign(fitpack))
    # Interior samples agree numerically; the ends are where the two formulations
    # differ most, so compare away from them.
    assert portable[3:-3] == pytest.approx(fitpack[3:-3], rel=1e-3)


def test_sample_positions_pass_through_the_input_points():
    pts = _arc(40.0, right=True, n=12)
    # Sampled densely enough that the nearest-sample distance is dominated by the
    # spline's fidelity rather than by the spacing between samples.
    sampled = sample_positions_portable(pts, num_samples=4000)
    # Every input point must lie on the sampled curve: the spline interpolates.
    for p in pts:
        assert np.min(np.hypot(*(sampled - p).T)) < 0.01


def test_degenerate_inputs_decline_rather_than_guess():
    assert curvature_along_portable(np.zeros((2, 2))).size == 0
    assert sample_positions_portable(np.zeros((2, 2))).shape == (0, 2)
    assert lane_curvature_portable(np.zeros((2, 2))) == 0.0
    # All points coincident: zero total arclength, no parameterization.
    assert curvature_along_portable(np.ones((5, 2))).size == 0


def test_three_point_input_still_yields_curvature():
    # Not-a-knot needs two interior knots; with three points the natural condition
    # is used instead, and the estimate must still be finite rather than absent.
    kappa = curvature_along_portable(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]), 10)
    assert kappa.size == 10 and np.all(np.isfinite(kappa))
