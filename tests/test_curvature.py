"""Analytic tests for curvature estimation.

Curvature is a silent-bug hotspot (second-derivative quantity), so it is checked
against curves with known closed-form curvature: straight line, circle, clothoid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.curvelanes import FrameAnnotation, Lane
from src.geometry.curvature import (
    curvature_along,
    frame_curvature,
    lane_curvature,
    signed_curvature_along,
)


def _interior(values: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Trim spline-endpoint samples where a cubic fit is least reliable."""
    n = values.size
    lo = int(n * frac)
    hi = n - lo
    return values[lo:hi]


def test_straight_line_has_zero_curvature() -> None:
    t = np.linspace(0.0, 100.0, 20)
    pts = np.stack([t, 2.0 * t + 5.0], axis=1)  # slope-2 line
    kappa = curvature_along(pts, smoothing=0.0)
    assert np.all(kappa < 1e-6)


def test_vertical_line_has_zero_curvature() -> None:
    # A perfectly vertical line: y=f(x) would blow up here; parametric must not.
    y = np.linspace(0.0, 100.0, 20)
    pts = np.stack([np.full_like(y, 42.0), y], axis=1)
    kappa = curvature_along(pts, smoothing=0.0)
    assert np.all(kappa < 1e-6)


@pytest.mark.parametrize("radius", [50.0, 100.0, 250.0])
def test_circle_curvature_is_inverse_radius(radius: float) -> None:
    # Sample a 120-degree arc (a realistic lane span, not a full loop).
    theta = np.linspace(-np.pi / 3, np.pi / 3, 40)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    kappa = _interior(curvature_along(pts, smoothing=0.0))
    expected = 1.0 / radius
    assert np.allclose(kappa, expected, rtol=0.02)


def test_clothoid_curvature_is_linear_in_arclength() -> None:
    # Euler spiral via Fresnel integrals: curvature grows linearly with arclength.
    from scipy.special import fresnel

    t = np.linspace(0.05, 2.0, 200)
    sin_s, cos_s = fresnel(t)  # x=C(t), y=S(t); unit-speed => arclength s = t
    pts = np.stack([cos_s, sin_s], axis=1) * 50.0  # scale up to pixel-ish magnitudes

    kappa = curvature_along(pts, num_samples=200, smoothing=0.0)
    s = np.linspace(0.0, 1.0, kappa.size)  # normalized arclength proxy (monotone in t)
    k, s = _interior(kappa), _interior(s)

    # Curvature should be a strongly increasing, near-linear function of arclength.
    corr = np.corrcoef(s, k)[0, 1]
    assert corr > 0.99
    assert k[-1] > k[0]


def test_resolution_invariance_of_normalized_curvature() -> None:
    # Same circular arc at two scales, in two image widths scaled the same way:
    # normalized curvature must match; raw pixel curvature must halve.
    theta = np.linspace(-np.pi / 4, np.pi / 4, 40)
    radius = 200.0
    arc = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    small = _frame(arc, width=1000, height=562)
    big = _frame(arc * 2.0, width=2000, height=1124)

    k_small = frame_curvature(small, smoothing=0.0)
    k_big = frame_curvature(big, smoothing=0.0)
    assert np.isclose(k_small, k_big, rtol=0.03)

    # Raw (unnormalized) curvature should scale as 1/length.
    raw_small = lane_curvature(arc, smoothing=0.0)
    raw_big = lane_curvature(arc * 2.0, smoothing=0.0)
    assert np.isclose(raw_big, raw_small / 2.0, rtol=0.03)


def test_normalized_circle_curvature_equals_width_over_radius() -> None:
    radius, width = 300.0, 1500.0
    theta = np.linspace(-np.pi / 4, np.pi / 4, 50)
    arc = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    frame = _frame(arc, width=int(width), height=844)
    # kappa_norm = width / radius
    assert np.isclose(frame_curvature(frame, smoothing=0.0), width / radius, rtol=0.03)


def test_short_and_degenerate_lanes_yield_zero() -> None:
    # Two-point lane: no curvature defined.
    assert lane_curvature(np.array([[0.0, 0.0], [10.0, 10.0]])) == 0.0
    # Duplicate points collapse below the minimum.
    assert lane_curvature(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])) == 0.0


def _frame(points: np.ndarray, width: int, height: int) -> FrameAnnotation:
    return FrameAnnotation(
        image_path=Path("f.jpg"),
        label_path=Path("f.lines.json"),
        width=width,
        height=height,
        lanes=[Lane(points=np.asarray(points, dtype=np.float64))],
    )


def test_frame_curvature_takes_max_over_lanes() -> None:
    straight = np.stack([np.linspace(0, 100, 10), np.full(10, 50.0)], axis=1)
    theta = np.linspace(-np.pi / 4, np.pi / 4, 40)
    curved = np.stack([100.0 * np.cos(theta), 100.0 * np.sin(theta)], axis=1)
    frame = FrameAnnotation(
        image_path=Path("f.jpg"),
        label_path=Path("f.lines.json"),
        width=1000,
        height=562,
        lanes=[Lane(points=straight), Lane(points=curved)],
    )
    # Frame curvature must reflect the curved lane, not be diluted by the straight one.
    assert frame_curvature(frame, smoothing=0.0) == pytest.approx(
        lane_curvature(curved / 1000.0, smoothing=0.0), rel=1e-9
    )


def test_signed_curvature_matches_magnitude_and_carries_sign() -> None:
    R = 50.0
    theta = np.linspace(0.0, 1.0, 60)
    # Counter-clockwise arc: standard convention gives a positive signed curvature.
    ccw = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
    signed = signed_curvature_along(ccw, smoothing=0.0)
    unsigned = curvature_along(ccw, smoothing=0.0)
    assert np.all(signed > 0)
    np.testing.assert_allclose(np.abs(signed), unsigned, rtol=1e-9)
    # Reversing the traversal flips the sign but not the magnitude.
    flipped = signed_curvature_along(ccw[::-1], smoothing=0.0)
    assert np.all(flipped < 0)
    np.testing.assert_allclose(np.abs(flipped), unsigned[::-1], rtol=1e-6)


def test_signed_curvature_undefined_returns_empty() -> None:
    assert signed_curvature_along(np.array([[0.0, 0.0], [1.0, 1.0]])).size == 0
