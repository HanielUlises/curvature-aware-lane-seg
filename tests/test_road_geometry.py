"""Tests for metric road geometry (offset, heading, curvature)."""

from __future__ import annotations

import cv2
import numpy as np

from src.geometry.ipm import GroundPlane, build_ground_homography
from src.geometry.road_geometry import road_geometry, road_geometry_from_mask

# Identity mapping: the supplied points are already in (x, z) ground metres.
IDENTITY = GroundPlane(h=np.eye(3), h_inv=np.eye(3))


def _line(x_of_z, z0=0.0, z1=20.0, n=25):
    z = np.linspace(z0, z1, n)
    return np.column_stack([x_of_z(z), z])


def test_straight_centered_lane():
    rg = road_geometry(_line(lambda z: np.zeros_like(z)), IDENTITY)
    assert rg is not None
    assert abs(rg.lateral_offset_m) < 1e-6
    assert abs(rg.heading_error_rad) < 1e-6
    assert rg.curvature_1pm < 1e-3


def test_lateral_offset_recovered():
    rg = road_geometry(_line(lambda z: np.full_like(z, 2.0)), IDENTITY)
    assert rg is not None
    assert abs(rg.lateral_offset_m - 2.0) < 1e-6
    assert abs(rg.heading_error_rad) < 1e-6


def test_heading_error_sign_and_value():
    # Lane drifts right as it goes forward: x = 0.1 z -> heading = atan(0.1) > 0.
    rg = road_geometry(_line(lambda z: 0.1 * z), IDENTITY)
    assert rg is not None
    assert rg.heading_error_rad > 0
    assert abs(rg.heading_error_rad - np.arctan(0.1)) < 1e-6
    assert abs(rg.lateral_offset_m) < 1e-6  # passes through origin at z=0
    assert rg.curvature_1pm < 1e-3


def _arc(R, turn_right=True, theta_max=0.9, n=40):
    theta = np.linspace(0.0, theta_max, n)
    x = R - R * np.cos(theta)
    return np.column_stack([x if turn_right else -x, R * np.sin(theta)])


def test_curvature_of_arc_matches_inverse_radius():
    R = 10.0
    rg = road_geometry(_arc(R, turn_right=True), IDENTITY)
    assert rg is not None
    assert abs(rg.curvature_1pm - 1.0 / R) < 0.02


def test_curvature_sign_distinguishes_turn_direction():
    R = 10.0
    right = road_geometry(_arc(R, turn_right=True), IDENTITY)
    left = road_geometry(_arc(R, turn_right=False), IDENTITY)
    assert right is not None and left is not None
    assert right.curvature_1pm > 0  # right turns positive
    assert left.curvature_1pm < 0
    assert abs(right.curvature_1pm + left.curvature_1pm) < 1e-6  # mirror images


def test_preview_curvature_within_and_beyond_extent():
    R = 10.0
    # The arc spans z up to R*sin(0.9) ~ 7.8 m, so 5 m is covered and 20 m is not.
    rg = road_geometry(_arc(R), IDENTITY, preview_distances_m=(5.0, 20.0))
    assert rg is not None
    assert abs(rg.preview_curvature_1pm[0] - 1.0 / R) < 0.02
    assert np.isnan(rg.preview_curvature_1pm[1])


def test_preview_curvature_zero_on_straight_lane():
    rg = road_geometry(_line(lambda z: np.zeros_like(z)), IDENTITY,
                       preview_distances_m=(5.0, 10.0))
    assert rg is not None
    assert np.all(np.abs(rg.preview_curvature_1pm) < 1e-3)


def test_returns_none_for_too_few_points():
    assert road_geometry(np.array([[0.0, 0.0], [0.0, 1.0]]), IDENTITY) is None


def test_from_mask_end_to_end():
    w, h = 512, 288
    mask = np.zeros((h, w), np.uint8)
    cv2.line(mask, (220, 40), (220, 280), 255, 4)
    cv2.line(mask, (300, 40), (300, 280), 255, 4)
    gp = build_ground_homography(
        (w, h), src_trapezoid=((0.42, 0.55), (0.58, 0.55), (0.95, 1.0), (0.05, 1.0))
    )
    rg = road_geometry_from_mask(mask, gp)
    assert rg is not None
    assert rg.ground_centerline.shape[1] == 2
    assert np.isfinite(rg.lateral_offset_m)
    assert rg.curvature_1pm < 0.1  # near-straight lanes -> low curvature


def test_from_mask_returns_none_without_ego_lane():
    w, h = 512, 288
    mask = np.zeros((h, w), np.uint8)
    cv2.line(mask, (60, 40), (60, 280), 255, 4)  # single lane, no bracketing pair
    gp = build_ground_homography(
        (w, h), src_trapezoid=((0.42, 0.55), (0.58, 0.55), (0.95, 1.0), (0.05, 1.0))
    )
    assert road_geometry_from_mask(mask, gp) is None
