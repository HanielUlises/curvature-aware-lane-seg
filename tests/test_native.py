"""Equivalence tests between the native chain and the Python reference.

These are the tests that keep the two-implementation contract meaningful now that only
one of them runs in production. If the Python implementations were deleted and Python
became a thin wrapper, the golden vectors would be comparing the port against itself.
Here the reference is exercised directly and the native output must match it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import native
from src.geometry.centerline import extract_lane_polylines as py_extract
from src.geometry.lane_tracker import EgoBoundaryTracker

pytestmark = pytest.mark.skipif(
    not native.available(),
    reason=f"native library unavailable: {native.why_unavailable()}",
)

W, H = 512, 288


def _mask(left_x=220, right_x=300, right_bottom=280, curve=0):
    m = np.zeros((H, W), np.uint8)
    for y in range(40, 280):
        t = (y - 40) / 240.0
        cv2.circle(m, (int(left_x + curve * t * t), y), 2, 255, -1)
        if y <= right_bottom:
            cv2.circle(m, (int(right_x + curve * t * t), y), 2, 255, -1)
    return m


def test_abi_version_is_checked():
    """A stale shared library must be refused, not called with a mismatched struct."""
    assert native.REQUIRED_ABI == native._load().cp_abi_version()


def test_polyline_decomposition_matches_the_reference_exactly():
    for curve in (0, 40, -60):
        m = _mask(curve=curve)
        got, want = native.extract_lane_polylines(m), py_extract(m)
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert a.shape == b.shape
            assert np.abs(a - b).max() == 0.0


def test_centerline_matches_the_python_tracker_over_a_sequence():
    """The tracker is stateful, so equivalence has to hold across frames, not on one."""
    chain = native.NativeChain(W, H)
    tracker = EgoBoundaryTracker(W, H)
    worst = 0.0
    compared = 0
    for k in range(30):
        m = _mask(left_x=220 + (k % 5), right_x=300 - (k % 3), curve=k)
        res = chain.process(m)
        py = tracker.centerline(tracker.update(py_extract(m)))
        assert res.has_centerline == (py is not None), f"frame {k}"
        if py is None:
            continue
        got = chain.centerline()
        assert got.shape == py.shape, f"frame {k}"
        worst = max(worst, float(np.abs(got - py).max()))
        compared += got.size
    assert compared > 0
    assert worst < 1e-9, f"worst disagreement {worst}"


def test_a_dropout_is_handled_the_same_way():
    chain = native.NativeChain(W, H)
    tracker = EgoBoundaryTracker(W, H)
    frames = [_mask()] * 3 + [np.zeros((H, W), np.uint8)] * 3 + [_mask()] * 3
    for k, m in enumerate(frames):
        res = chain.process(m)
        py = tracker.centerline(tracker.update(py_extract(m)))
        assert res.has_centerline == (py is not None), f"frame {k}"


def test_geometry_and_steering_need_a_calibration():
    plain = native.NativeChain(W, H)
    res = plain.process(_mask())
    assert res.has_centerline and not res.has_geometry
    # Without a ground plane the metric fields stay at zero rather than being invented.
    assert res.lateral_offset_m == 0.0 and res.curvature_1pm == 0.0

    calib = (522.29, 522.29, 256.0, 82.29, 1.6168, 0.13036, 0.0)
    metric = native.NativeChain(W, H, calib)
    res = metric.process(_mask())
    assert res.has_centerline and res.has_geometry
    assert np.isfinite(res.lateral_offset_m)
    assert abs(res.steer_rad) <= np.radians(35.0) + 1e-9


def test_reset_clears_temporal_state():
    calib = (522.29, 522.29, 256.0, 82.29, 1.6168, 0.13036, 0.0)
    chain = native.NativeChain(W, H, calib)
    for _ in range(5):
        chain.process(_mask())
    # Two blank frames: the tracker coasts a row for one frame, so a single blank still
    # yields a centreline and the filter never sees a gap.
    chain.process(np.zeros((H, W), np.uint8))
    before = chain.process(np.zeros((H, W), np.uint8))
    assert before.coasting_frames >= 1
    chain.reset()
    after = chain.process(np.zeros((H, W), np.uint8))
    assert after.coasting_frames == 1


def test_mask_shape_is_validated():
    chain = native.NativeChain(W, H)
    with pytest.raises(ValueError):
        chain.process(np.zeros((10, 10), np.uint8))
