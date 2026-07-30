"""Tests for camera calibration and the calibrated ground projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.geometry.calibration import (
    CameraCalibration,
    calibrate_pitch_from_lane_parallelism,
    calibration_for_preprocessed_frame,
    intrinsics_for_preprocessed_frame,
    lane_parallelism_cost,
    calibration_from_vanishing_point,
    estimate_vanishing_point,
    ground_plane_from_calibration,
    intrinsics_from_fov,
    project_ground_to_image,
    refine_height_from_lane_width,
)
from src.geometry.curvature import curvature_along

IMAGE_SIZE = (1280, 720)


def _calib(pitch_deg=8.0, yaw_deg=0.0, height=1.35, hfov=70.0):
    fx, fy, cx, cy = intrinsics_from_fov(IMAGE_SIZE, hfov)
    return CameraCalibration(
        fx=fx, fy=fy, cx=cx, cy=cy, height_m=height,
        pitch_rad=math.radians(pitch_deg), yaw_rad=math.radians(yaw_deg),
    )


def test_intrinsics_from_fov_geometry():
    fx, fy, cx, cy = intrinsics_from_fov((1280, 720), 90.0)
    # At 90 degrees horizontal FOV, fx equals half the width.
    assert abs(fx - 640.0) < 1e-9
    assert fx == fy and (cx, cy) == (640.0, 360.0)


def test_invalid_fov_rejected():
    with pytest.raises(ValueError):
        intrinsics_from_fov(IMAGE_SIZE, 0.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(IMAGE_SIZE, 180.0)


def test_zero_height_rejected():
    with pytest.raises(ValueError):
        ground_plane_from_calibration(_calib(height=0.0))


def test_ground_image_round_trip():
    calib = _calib(pitch_deg=8.0, yaw_deg=3.0)
    ground = np.array([[0.0, 10.0], [1.85, 20.0], [-1.85, 35.0], [0.5, 5.0]])
    back = ground_plane_from_calibration(calib).to_ground(
        project_ground_to_image(calib, ground)
    )
    np.testing.assert_allclose(back, ground, atol=1e-6)


def test_vanishing_point_matches_analytic_form():
    calib = _calib(pitch_deg=8.0, yaw_deg=4.0)
    # A very distant point on the road centreline must land on the vanishing point.
    far = project_ground_to_image(calib, np.array([[0.0, 1e7]]))[0]
    np.testing.assert_allclose(far, calib.vanishing_point(), atol=1e-3)


def test_pitch_and_yaw_recovered_from_vanishing_point():
    calib = _calib(pitch_deg=7.5, yaw_deg=-2.5)
    recovered = calibration_from_vanishing_point(
        calib.vanishing_point(), IMAGE_SIZE, hfov_deg=70.0, height_m=calib.height_m
    )
    assert abs(recovered.pitch_rad - calib.pitch_rad) < 1e-9
    assert abs(recovered.yaw_rad - calib.yaw_rad) < 1e-9


def test_pitching_down_raises_the_vanishing_point():
    level = _calib(pitch_deg=0.0)
    tilted = _calib(pitch_deg=10.0)
    # Looking further down pushes the horizon up the image (smaller row index).
    assert tilted.vanishing_point()[1] < level.vanishing_point()[1]


def test_parallel_lanes_stay_parallel_on_the_ground():
    # The defining property of a correct inverse-perspective map: lanes that are
    # parallel and 3.7 m apart on the road converge in the image but must come back
    # parallel and 3.7 m apart after projection to the ground.
    calib = _calib(pitch_deg=8.0)
    z = np.linspace(6.0, 40.0, 30)
    left = np.column_stack([np.full_like(z, -1.85), z])
    right = np.column_stack([np.full_like(z, 1.85), z])

    image_left = project_ground_to_image(calib, left)
    image_right = project_ground_to_image(calib, right)
    # Sanity: perspective really does make them converge in the image.
    near_gap = abs(image_right[0, 0] - image_left[0, 0])
    far_gap = abs(image_right[-1, 0] - image_left[-1, 0])
    assert far_gap < near_gap / 2

    plane = ground_plane_from_calibration(calib)
    back_left = plane.to_ground(image_left)
    back_right = plane.to_ground(image_right)
    width = back_right[:, 0] - back_left[:, 0]
    np.testing.assert_allclose(width, 3.7, atol=1e-6)


def test_estimate_vanishing_point_from_converging_lanes():
    calib = _calib(pitch_deg=8.0, yaw_deg=2.0)
    z = np.linspace(6.0, 60.0, 40)
    lanes = [
        project_ground_to_image(calib, np.column_stack([np.full_like(z, x), z]))
        for x in (-1.85, 1.85)
    ]
    estimated = estimate_vanishing_point(lanes)
    assert estimated is not None
    np.testing.assert_allclose(estimated, calib.vanishing_point(), atol=1.0)


def test_estimate_vanishing_point_needs_two_lanes():
    assert estimate_vanishing_point([]) is None
    assert estimate_vanishing_point([np.array([[0.0, 0.0], [1.0, 1.0]])]) is None


def test_estimate_vanishing_point_rejects_parallel_lanes():
    # Two exactly vertical image lines never intersect; the estimate must decline.
    a = np.column_stack([np.full(10, 400.0), np.linspace(300, 700, 10)])
    b = np.column_stack([np.full(10, 800.0), np.linspace(300, 700, 10)])
    assert estimate_vanishing_point([a, b]) is None


def test_height_refined_from_known_lane_width():
    truth = _calib(pitch_deg=8.0, height=1.35)
    z = np.linspace(6.0, 40.0, 30)
    image_left = project_ground_to_image(truth, np.column_stack([np.full_like(z, -1.85), z]))
    image_right = project_ground_to_image(truth, np.column_stack([np.full_like(z, 1.85), z]))

    # Start from a badly wrong height; the lane width should pull it back.
    guessed = CameraCalibration(
        fx=truth.fx, fy=truth.fy, cx=truth.cx, cy=truth.cy,
        height_m=2.5, pitch_rad=truth.pitch_rad, yaw_rad=truth.yaw_rad,
    )
    refined = refine_height_from_lane_width(guessed, image_left, image_right, 3.7)
    assert abs(refined.height_m - truth.height_m) < 1e-6


def test_curvature_of_projected_arc_recovers_inverse_radius():
    # End-to-end check against exact geometry: a circular arc of radius R on the road,
    # projected into the image and mapped back, must have curvature 1/R.
    calib = _calib(pitch_deg=8.0)
    radius = 80.0
    theta = np.linspace(0.0, 0.45, 40)
    arc = np.column_stack([radius * (1 - np.cos(theta)), radius * np.sin(theta)])
    recovered = ground_plane_from_calibration(calib).to_ground(
        project_ground_to_image(calib, arc)
    )
    kappa = curvature_along(recovered, smoothing=0.0)
    assert abs(float(np.median(kappa)) - 1.0 / radius) < 1e-4


def test_preprocessed_intrinsics_account_for_sky_crop():
    # preprocess_geometry removes the top 30%, so the principal point must move up in
    # the cropped frame rather than sitting at its centre.
    fx, fy, cx, cy = intrinsics_for_preprocessed_frame(
        (2560, 1440), (512, 288), sky_frac=0.30, hfov_deg=70.0
    )
    assert abs(cx - 256.0) < 1e-6      # width crop is symmetric
    assert cy < 288 / 2               # the crop pushes the principal point up
    assert abs(cy - 82.3) < 0.5
    assert abs(fx - 522.3) < 0.5 and fx == fy


def test_preprocessed_intrinsics_are_resolution_independent():
    a = intrinsics_for_preprocessed_frame((2560, 1440), (512, 288), 0.30, 70.0)
    b = intrinsics_for_preprocessed_frame((1280, 720), (512, 288), 0.30, 70.0)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_preprocessed_intrinsics_reject_bad_sky_frac():
    with pytest.raises(ValueError):
        intrinsics_for_preprocessed_frame((1280, 720), (512, 288), 1.0, 70.0)


def _synthetic_lane_pairs(truth, n_frames=25, z_lo=8.0, z_hi=25.0, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_frames):
        z = np.linspace(z_lo, z_hi, 20)
        offset = rng.uniform(-0.6, 0.6)
        left = project_ground_to_image(
            truth, np.column_stack([np.full_like(z, -1.85 + offset), z])
        )
        right = project_ground_to_image(
            truth, np.column_stack([np.full_like(z, 1.85 + offset), z])
        )
        pairs.append((left, right))
    return pairs


def test_parallelism_cost_is_minimal_at_the_true_pitch():
    truth = _calib(pitch_deg=9.0)
    pairs = _synthetic_lane_pairs(truth)
    at_truth = lane_parallelism_cost(truth, pairs)
    assert at_truth < 1e-9
    for wrong in (5.0, 13.0):
        assert lane_parallelism_cost(_calib(pitch_deg=wrong), pairs) > at_truth


def test_pitch_recovered_from_lane_parallelism():
    truth = _calib(pitch_deg=9.0)
    fx, fy, cx, cy = truth.fx, truth.fy, truth.cx, truth.cy
    estimated = calibrate_pitch_from_lane_parallelism(
        _synthetic_lane_pairs(truth), (fx, fy, cx, cy), height_m=truth.height_m
    )
    assert estimated is not None
    assert abs(math.degrees(estimated.pitch_rad) - 9.0) < 0.05


def test_parallelism_cost_rejects_degenerate_projection():
    # Points at or behind the camera plane must not be scored as a good fit.
    truth = _calib(pitch_deg=9.0)
    pairs = _synthetic_lane_pairs(truth, n_frames=6)
    assert not np.isfinite(lane_parallelism_cost(_calib(pitch_deg=-1.9), pairs))


def test_pitch_estimation_returns_none_without_usable_frames():
    assert calibrate_pitch_from_lane_parallelism([], (500.0, 500.0, 256.0, 82.0)) is None


def test_calibration_moved_to_preprocessed_frame_matches_direct_intrinsics():
    # Cropping and scaling an image changes K but not where the camera is, so moving a
    # native calibration into preprocessed coordinates must agree with computing the
    # preprocessed intrinsics from scratch, and must leave the extrinsics alone.
    native = CameraCalibration(
        *intrinsics_from_fov((2560, 1440), 70.0),
        height_m=1.5, pitch_rad=math.radians(8.0), yaw_rad=math.radians(2.0),
    )
    moved = calibration_for_preprocessed_frame(native, (2560, 1440), (512, 288), 0.30)
    expected = intrinsics_for_preprocessed_frame((2560, 1440), (512, 288), 0.30, 70.0)
    np.testing.assert_allclose((moved.fx, moved.fy, moved.cx, moved.cy), expected, atol=1e-6)
    assert moved.pitch_rad == native.pitch_rad
    assert moved.yaw_rad == native.yaw_rad
    assert moved.height_m == native.height_m


def test_preprocessed_calibration_projects_ground_points_consistently():
    # A ground point must land at the same place whether projected with the native
    # calibration and then cropped, or projected with the moved calibration directly.
    source, target, sky = (1280, 720), (512, 288), 0.30
    native = CameraCalibration(
        *intrinsics_from_fov(source, 70.0),
        height_m=1.6, pitch_rad=math.radians(7.5), yaw_rad=0.0,
    )
    moved = calibration_for_preprocessed_frame(native, source, target, sky)

    ground = np.array([[0.0, 15.0], [1.5, 25.0], [-1.2, 10.0]])
    native_px = project_ground_to_image(native, ground)
    # Apply the same crop and scale preprocess_geometry would.
    top = int(round(source[1] * sky))
    crop_h = source[1] - top
    crop_w = int(round(crop_h * target[0] / target[1]))
    x0 = (source[0] - crop_w) // 2
    scale = target[0] / crop_w
    cropped = (native_px - np.array([x0, top])) * scale

    np.testing.assert_allclose(project_ground_to_image(moved, ground), cropped, atol=1e-6)
