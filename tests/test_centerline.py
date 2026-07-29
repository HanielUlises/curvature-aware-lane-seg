"""Tests for lane-polyline and ego-centerline extraction from masks."""

from __future__ import annotations

import cv2
import numpy as np

from src.geometry.centerline import (
    ego_centerline,
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
