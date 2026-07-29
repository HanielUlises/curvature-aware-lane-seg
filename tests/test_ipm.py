"""Tests for the inverse-perspective (ground-plane) homography."""

from __future__ import annotations

import numpy as np

from src.geometry.ipm import (
    apply_homography,
    build_ground_homography,
    homography_from_points,
)


def test_identity_correspondence():
    # A square mapped to itself yields the identity transform.
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    h = homography_from_points(sq, sq)
    np.testing.assert_allclose(apply_homography(h, sq), sq, atol=1e-9)


def test_exact_four_point_mapping():
    # The homography must reproduce its own defining correspondences exactly.
    src = np.array([[10, 20], [400, 30], [420, 280], [30, 260]], dtype=float)
    dst = np.array([[-1.85, 30], [1.85, 30], [1.85, 0], [-1.85, 0]], dtype=float)
    h = homography_from_points(src, dst)
    np.testing.assert_allclose(apply_homography(h, src), dst, atol=1e-6)


def test_round_trip_inverse():
    gp = build_ground_homography(
        (512, 288),
        src_trapezoid=((0.42, 0.65), (0.58, 0.65), (0.95, 1.0), (0.05, 1.0)),
    )
    img_pts = np.array([[256, 288], [200, 200], [300, 210]], dtype=float)
    back = gp.to_image(gp.to_ground(img_pts))
    np.testing.assert_allclose(back, img_pts, atol=1e-6)


def test_ground_extent_matches_config():
    w, h = 512, 288
    trap = ((0.42, 0.65), (0.58, 0.65), (0.95, 1.0), (0.05, 1.0))
    gp = build_ground_homography((w, h), trap, lane_width_m=3.7, look_ahead_m=30.0)
    src = np.array([[fx * w, fy * h] for fx, fy in trap])
    ground = gp.to_ground(src)
    # Near base spans the full lane width at z=0; top reaches look-ahead at z=30.
    assert abs(ground[3, 0] - (-1.85)) < 1e-4 and abs(ground[2, 0] - 1.85) < 1e-4
    assert abs(ground[0, 1] - 30.0) < 1e-4 and abs(ground[2, 1] - 0.0) < 1e-4


def test_straight_vertical_image_line_is_straight_in_ground():
    # A column at image centre maps to a straight line in ground coords (x ~ const).
    gp = build_ground_homography(
        (512, 288),
        src_trapezoid=((0.42, 0.65), (0.58, 0.65), (0.95, 1.0), (0.05, 1.0)),
    )
    col = np.column_stack([np.full(20, 256.0), np.linspace(190, 288, 20)])
    ground = gp.to_ground(col)
    # Lateral coordinate stays near zero (centre) across the whole column.
    assert np.ptp(ground[:, 0]) < 0.05


def test_requires_four_points():
    try:
        homography_from_points(np.zeros((3, 2)), np.zeros((3, 2)))
        assert False, "expected ValueError"
    except ValueError:
        pass
