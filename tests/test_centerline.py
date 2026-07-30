"""Tests for lane-polyline and ego-centerline extraction from masks."""

from __future__ import annotations

import cv2
import numpy as np

from src.geometry.centerline import (
    ego_centerline,
    resample_boundary,
    extract_lane_polylines,
    image_lateral_offset,
)


def _blank(h=288, w=512):
    return np.zeros((h, w), dtype=np.uint8)


def test_two_vertical_lanes_recovered():
    mask = _blank()
    cv2.line(mask, (200, 20), (200, 260), 255, 3)
    cv2.line(mask, (320, 20), (320, 260), 255, 3)
    polys = extract_lane_polylines(mask)
    assert len(polys) == 2
    # Sorted left-to-right; centroid columns match the drawn lines.
    assert abs(polys[0][:, 0].mean() - 200) < 1.5
    assert abs(polys[1][:, 0].mean() - 320) < 1.5
    # Ordered by row (monotonic increasing y).
    assert np.all(np.diff(polys[0][:, 1]) > 0)


def test_diagonal_lane_follows_center():
    mask = _blank()
    cv2.line(mask, (150, 20), (350, 260), 255, 3)
    polys = extract_lane_polylines(mask)
    assert len(polys) == 1
    poly = polys[0]
    # At the top the column is ~150, at the bottom ~350.
    top = poly[np.argmin(poly[:, 1]), 0]
    bot = poly[np.argmax(poly[:, 1]), 0]
    assert abs(top - 150) < 3 and abs(bot - 350) < 3


def test_small_speck_is_filtered():
    mask = _blank()
    cv2.line(mask, (200, 20), (200, 260), 255, 3)  # a real lane
    mask[100:103, 400:403] = 255  # a 3x3 speck
    polys = extract_lane_polylines(mask)
    assert len(polys) == 1  # speck dropped by min_rows / min_pixels


def test_ego_centerline_is_midpoint():
    w = 512
    mask = _blank(w=w)
    cv2.line(mask, (200, 20), (200, 260), 255, 3)
    cv2.line(mask, (320, 20), (320, 260), 255, 3)
    polys = extract_lane_polylines(mask)
    center = ego_centerline(polys, w)
    assert center is not None
    # Midway between 200 and 320 is 260.
    assert abs(center[:, 0].mean() - 260) < 2.0
    # Ordered top-to-bottom.
    assert np.all(np.diff(center[:, 1]) > 0)


def test_ego_centerline_needs_bracketing_pair():
    w = 512
    mask = _blank(w=w)
    # Two lanes both left of center -> no right lane to bracket the ego column.
    cv2.line(mask, (100, 20), (100, 260), 255, 3)
    cv2.line(mask, (200, 20), (200, 260), 255, 3)
    polys = extract_lane_polylines(mask)
    assert ego_centerline(polys, w) is None


def test_lateral_offset_sign():
    w = 512
    mask = _blank(w=w)
    # Lanes bracket the ego column (240 < 256 <= 460) but their midpoint 350 sits
    # right of the camera axis.
    cv2.line(mask, (240, 20), (240, 260), 255, 3)
    cv2.line(mask, (460, 20), (460, 260), 255, 3)
    polys = extract_lane_polylines(mask)
    center = ego_centerline(polys, w)
    assert center is not None
    off = image_lateral_offset(center, w)
    assert off > 0  # centerline is right of the camera axis
    assert abs(off - (350 - 256)) < 3.0


def test_empty_mask_yields_nothing():
    assert extract_lane_polylines(_blank()) == []
    assert ego_centerline([], 512) is None


def test_resample_boundary_interpolates_and_extends_within_bound():
    poly = np.column_stack([np.linspace(100.0, 140.0, 5), np.linspace(100.0, 180.0, 5)])
    rows = np.array([100.0, 140.0, 180.0, 200.0, 400.0])
    out = resample_boundary(poly, rows, max_extend_rows=45)
    # Inside the observed span it interpolates the drawn columns.
    assert abs(out[0] - 100.0) < 1e-6 and abs(out[2] - 140.0) < 1e-6
    # Just past the end it continues along the fitted direction.
    assert abs(out[3] - 150.0) < 1.0
    # Far past the end it declines rather than inventing geometry.
    assert np.isnan(out[4])


def test_resample_boundary_declines_degenerate_input():
    assert np.all(np.isnan(resample_boundary(np.array([[1.0, 2.0]]), np.array([2.0]))))


def test_centerline_extent_survives_one_short_boundary():
    # The defect this fixed: taking only the rows both boundaries share let a single
    # short boundary truncate the centreline, and the truncation moved frame to frame.
    w, h = 512, 288
    long_mask = np.zeros((h, w), np.uint8)
    cv2.line(long_mask, (220, 40), (220, 280), 255, 4)
    cv2.line(long_mask, (300, 40), (300, 280), 255, 4)
    short_mask = np.zeros((h, w), np.uint8)
    cv2.line(short_mask, (220, 40), (220, 280), 255, 4)
    cv2.line(short_mask, (300, 40), (300, 150), 255, 4)  # right boundary stops early

    polys = extract_lane_polylines(short_mask)
    full = ego_centerline(extract_lane_polylines(long_mask), w, image_height=h)
    short = ego_centerline(polys, w, image_height=h)
    assert full is not None and short is not None

    # Where the old shared-range rule would have stopped: the nearer end of the shorter
    # boundary. The centreline must now reach past it by the permitted extension.
    shared_bottom = min(p[:, 1].max() for p in polys)
    extension = short[:, 1].max() - shared_bottom
    assert 35 < extension <= 45 + 1  # extended, but only by the bounded amount
    # It must still sit between the two boundaries.
    assert abs(short[:, 0].mean() - 260) < 12
