"""Tests for polyline-to-mask rasterization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.curvelanes import FrameAnnotation, Lane
from src.data.rasterize import (
    assert_target_aspect,
    is_target_aspect,
    rasterize_frame,
    rasterize_lanes,
)


def _frame(width: int, height: int, lanes: list[Lane]) -> FrameAnnotation:
    return FrameAnnotation(
        image_path=Path(f"{width}x{height}.jpg"),
        label_path=Path(f"{width}x{height}.lines.json"),
        width=width,
        height=height,
        lanes=lanes,
    )


def _lane(points: list[tuple[float, float]]) -> Lane:
    return Lane(points=np.array(points, dtype=np.float64))


def test_horizontal_line_lands_on_expected_row() -> None:
    # A horizontal lane at y=50 across a 100x100 canvas, 1px stroke.
    mask = rasterize_lanes([_lane([(10.0, 50.0), (90.0, 50.0)])], (100, 100), 1)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    ys, xs = np.nonzero(mask)
    assert set(ys) == {50}  # only row 50 lit
    assert xs.min() <= 10 and xs.max() >= 90


def test_stroke_width_thickens_the_line() -> None:
    thin = rasterize_lanes([_lane([(10.0, 50.0), (90.0, 50.0)])], (100, 100), 1)
    thick = rasterize_lanes([_lane([(10.0, 50.0), (90.0, 50.0)])], (100, 100), 7)
    assert thick.sum() > thin.sum()
    # A 7px stroke should light roughly 7 rows around y=50.
    rows = np.unique(np.nonzero(thick)[0])
    assert 5 <= len(rows) <= 9


def test_offframe_points_are_clipped_not_erroring() -> None:
    # Line runs from well outside the frame to inside; must clip cleanly.
    mask = rasterize_lanes([_lane([(-500.0, 50.0), (150.0, 50.0)])], (100, 100), 1)
    ys, xs = np.nonzero(mask)
    assert xs.min() >= 0 and xs.max() <= 99  # nothing drawn outside the canvas
    assert 50 in set(ys)


def test_near_vertical_segment_is_drawn() -> None:
    # Almost-vertical lane: 1px horizontal drift over the full height.
    mask = rasterize_lanes([_lane([(50.0, 0.0), (51.0, 99.0)])], (100, 100), 1)
    ys = np.unique(np.nonzero(mask)[0])
    assert len(ys) >= 95  # spans essentially the whole height, no gaps


def test_native_then_nearest_resize_keeps_thin_diagonal_connected() -> None:
    # Rasterize a thin diagonal at native res, downscale 4x with nearest.
    frame = _frame(400, 224, [_lane([(0.0, 0.0), (399.0, 223.0)])])
    mask = rasterize_frame(frame, target_size=(100, 56), stroke_px=2)
    assert mask.shape == (56, 100)
    # The diagonal must remain a connected foreground, not shatter into dots.
    assert mask.sum() > 0
    rows_with_fg = np.unique(np.nonzero(mask)[0])
    # Nearly every output row should carry some foreground for a full diagonal.
    assert len(rows_with_fg) >= 50


def test_rasterize_frame_no_resize_when_native_equals_target() -> None:
    frame = _frame(100, 56, [_lane([(0.0, 28.0), (99.0, 28.0)])])
    mask = rasterize_frame(frame, target_size=(100, 56), stroke_px=1)
    assert mask.shape == (56, 100)
    assert 28 in set(np.nonzero(mask)[0])


def test_empty_frame_yields_blank_mask() -> None:
    frame = _frame(256, 144, [])
    mask = rasterize_frame(frame, target_size=(128, 72))
    assert mask.shape == (72, 128)
    assert mask.sum() == 0


def test_aspect_guard_accepts_16x9_and_rejects_odd() -> None:
    assert is_target_aspect(2560, 1440, (512, 288))
    assert not is_target_aspect(1570, 660, (512, 288))

    ok = _frame(2560, 1440, [_lane([(0.0, 0.0), (10.0, 10.0)])])
    assert_target_aspect(ok)  # no raise

    bad = _frame(1570, 660, [_lane([(0.0, 0.0), (10.0, 10.0)])])
    with pytest.raises(ValueError, match="does not match target aspect"):
        rasterize_frame(bad)


def test_stroke_scaled_to_native_resolution() -> None:
    # Same target stroke, two native widths: the higher-res native mask should
    # use a proportionally thicker stroke so the post-resize width matches.
    big = _frame(1024, 576, [_lane([(0.0, 288.0), (1023.0, 288.0)])])
    small = _frame(256, 144, [_lane([(0.0, 72.0), (255.0, 72.0)])])
    m_big = rasterize_frame(big, target_size=(128, 72), stroke_px=4)
    m_small = rasterize_frame(small, target_size=(128, 72), stroke_px=4)
    rows_big = len(np.unique(np.nonzero(m_big)[0]))
    rows_small = len(np.unique(np.nonzero(m_small)[0]))
    # Post-resize stroke thickness should be within 1 row across resolutions.
    assert abs(rows_big - rows_small) <= 1
