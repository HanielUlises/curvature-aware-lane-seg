"""Tests for the temporal ego-boundary tracker."""

from __future__ import annotations

import cv2
import numpy as np

from src.geometry.centerline import ego_centerline, extract_lane_polylines
from src.geometry.lane_tracker import EgoBoundaryTracker, TrackedBoundaries

W, H = 512, 288


def _mask(left_x=220, right_x=300, right_bottom=280, thickness=4):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (left_x, 40), (left_x, 280), 255, thickness)
    cv2.line(m, (right_x, 40), (right_x, right_bottom), 255, thickness)
    return m


def _polys(**kw):
    return extract_lane_polylines(_mask(**kw))


def test_tracker_follows_a_static_lane():
    tracker = EgoBoundaryTracker(W, H)
    for _ in range(10):
        tracked = tracker.update(_polys())
    centre = tracker.centerline(tracked)
    assert centre is not None
    assert abs(centre[:, 0].mean() - 260) < 3.0


def test_tracking_reduces_jitter_on_a_stationary_lane():
    """The point of the tracker: a lane that is not moving must not appear to move."""
    rng = np.random.default_rng(0)
    tracker = EgoBoundaryTracker(W, H)
    raw_bases, tracked_bases = [], []
    for _ in range(40):
        # Same lane every frame, but the detection wobbles by a couple of pixels.
        jitter = int(rng.integers(-3, 4))
        polys = _polys(left_x=220 + jitter, right_x=300 + jitter)
        raw = ego_centerline(polys, W, image_height=H)
        tracked = tracker.centerline(tracker.update(polys))
        assert raw is not None and tracked is not None
        raw_bases.append(raw[np.argmax(raw[:, 1]), 0])
        tracked_bases.append(tracked[np.argmax(tracked[:, 1]), 0])

    # Compare frame-to-frame movement, discarding the tracker's initial convergence.
    raw_step = np.mean(np.abs(np.diff(raw_bases[5:])))
    tracked_step = np.mean(np.abs(np.diff(tracked_bases[5:])))
    assert tracked_step < 0.6 * raw_step


def test_extent_survives_a_frame_that_loses_the_near_end():
    tracker = EgoBoundaryTracker(W, H)
    for _ in range(6):
        full = tracker.update(_polys(right_bottom=280))
    reach_before = tracker.centerline(full)[:, 1].max()

    # The right boundary now stops 130 rows short: without the track the centreline
    # would retreat with it, which is the flicker this exists to remove.
    short = tracker.update(_polys(right_bottom=150))
    reach_after = tracker.centerline(short)[:, 1].max()
    assert reach_after > reach_before - 10

    naive = ego_centerline(_polys(right_bottom=150), W, image_height=H)
    assert naive[:, 1].max() < reach_after - 50


def test_coasting_is_bounded():
    tracker = EgoBoundaryTracker(W, H, max_coast_frames=4)
    for _ in range(6):
        tracker.update(_polys())
    # No lanes at all from here on.
    for i in range(1, 6):
        tracked = tracker.update([])
        assert tracked.coasting_frames == i
        assert not tracked.observed
    # Past the coast budget every row has been retired, so nothing is drawn.
    assert tracker.centerline(tracked) is None


def test_a_lane_change_resets_rather_than_smears():
    tracker = EgoBoundaryTracker(W, H, gate_px=40.0)
    for _ in range(8):
        tracker.update(_polys(left_x=180, right_x=260))
    # Both boundaries step sideways by more than the gate while the lane keeps its
    # width: a lane change, not a mis-association. They must still bracket the camera
    # axis, or there is no ego pair to fall back on.
    jumped = tracker.update(_polys(left_x=240, right_x=320))
    centre = tracker.centerline(jumped)
    assert jumped.observed and jumped.resets == 2
    # Followed immediately, not blended towards over several frames.
    assert abs(centre[:, 0].mean() - 280) < 6.0


def test_a_pair_that_is_suddenly_a_lane_too_wide_is_refused():
    """The defect: losing the ego marking let the next one out take its place.

    That boundary is roughly a lane further away, so the centreline steps sideways by
    half a lane and steps back when the real marking returns. A pair whose width jumps
    against the width being tracked is not the lane being followed, however well one of
    its boundaries matches.
    """
    tracker = EgoBoundaryTracker(W, H, gate_px=200.0)
    for _ in range(8):
        good = tracker.update(_polys(left_x=220, right_x=300))
    before = tracker.centerline(good)[:, 0].mean()

    # The right marking is lost and one a lane further out is found instead.
    wide = tracker.update(_polys(left_x=220, right_x=460))
    centre = tracker.centerline(wide)
    assert centre is not None
    # The centreline must not lurch towards the wrong pair's midpoint of 340.
    assert abs(centre[:, 0].mean() - before) < 15.0


def test_small_motion_is_smoothed_not_reset():
    tracker = EgoBoundaryTracker(W, H, gate_px=40.0)
    for _ in range(8):
        tracker.update(_polys(left_x=220, right_x=300))
    nudged = tracker.update(_polys(left_x=228, right_x=308))
    assert nudged.resets == 0
    centre = tracker.centerline(nudged)
    # Partway towards the new position, since alpha < 1.
    assert 260.0 < centre[:, 0].mean() < 268.0


def test_tracker_declines_before_it_has_seen_anything():
    tracker = EgoBoundaryTracker(W, H)
    tracked = tracker.update([])
    assert tracked.left is None and tracked.right is None
    assert tracker.centerline(tracked) is None


def _boundaries(left_cols, right_cols, rows):
    """A TrackedBoundaries built directly, to exercise the drawing guards."""
    def poly(cols):
        ok = np.isfinite(cols)
        return np.column_stack([cols[ok], rows[ok]])
    return TrackedBoundaries(left=poly(left_cols), right=poly(right_cols),
                             observed=True, coasting_frames=0, resets=0)


def test_crossed_boundaries_are_not_drawn():
    # Where both boundaries are extrapolated they can cross, and the midpoint of a
    # crossed pair is meaningless. Here the left track sits right of the right one.
    tracker = EgoBoundaryTracker(W, H)
    rows = tracker.rows
    tracked = _boundaries(np.full(rows.shape, 400.0), np.full(rows.shape, 200.0), rows)
    assert tracker.centerline(tracked) is None


def test_line_is_truncated_at_an_implausible_sideways_step():
    """The defect this fixed rendered as a detached fragment in the demo video."""
    tracker = EgoBoundaryTracker(W, H, max_lateral_slope=4.0)
    rows = tracker.rows
    left = np.full(rows.shape, 220.0)
    right = np.full(rows.shape, 300.0)
    right[rows > 150] += 200.0          # a sharp sideways step partway down
    centre = tracker.centerline(_boundaries(left, right, rows))
    assert centre is not None
    slope = np.abs(np.diff(centre[:, 0])) / np.diff(centre[:, 1])
    assert slope.max() <= 4.0 + 1e-6    # no segment bends faster than the limit
    assert centre[:, 1].max() < 170      # and the line stops before the bad region


def test_centerline_is_one_connected_run():
    tracker = EgoBoundaryTracker(W, H)
    rows = tracker.rows
    left = np.full(rows.shape, 220.0)
    right = np.full(rows.shape, 300.0)
    right[(rows > 120) & (rows < 170)] = np.nan   # a hole, with valid rows either side
    centre = tracker.centerline(_boundaries(left, right, rows))
    assert centre is not None
    gaps = np.diff(centre[:, 1])
    assert gaps.max() < 3 * np.median(gaps)       # contiguous, no jump across the hole


def test_a_one_frame_extent_spike_is_not_drawn():
    """A single frame that reaches much further must not make the line surge."""
    tracker = EgoBoundaryTracker(W, H, extent_median_frames=3)
    for _ in range(4):
        # The extent median is kept by centerline(), so it has to be called per frame,
        # exactly as the renderer does.
        before = tracker.centerline(tracker.update(_polys(right_bottom=150)))[:, 1].max()

    spike = tracker.update(_polys(right_bottom=280))
    after = tracker.centerline(spike)[:, 1].max()
    assert after - before < 20            # the spike is held back by the median


def test_a_sustained_extent_gain_is_taken_up_without_lag():
    """Unlike a growth rate limit, a median does not trail steady evidence."""
    tracker = EgoBoundaryTracker(W, H, extent_median_frames=3)
    for _ in range(4):
        tracker.centerline(tracker.update(_polys(right_bottom=150)))
    for _ in range(3):
        reach = tracker.centerline(tracker.update(_polys(right_bottom=280)))[:, 1].max()

    # With the same boundaries and no history, this is what the evidence supports.
    fresh = EgoBoundaryTracker(W, H)
    for _ in range(3):
        ref = fresh.centerline(fresh.update(_polys(right_bottom=280)))[:, 1].max()
    assert reach >= ref - 1e-6


def test_extent_never_outruns_the_current_frame():
    tracker = EgoBoundaryTracker(W, H, extent_median_frames=5)
    for _ in range(6):
        tracker.centerline(tracker.update(_polys(right_bottom=280)))
    # Retraction is immediate once the coast is spent: the median can only pull the
    # extent in, never push it past what this frame supports.
    for _ in range(2):
        centre = tracker.centerline(tracker.update(_polys(right_bottom=120)))
    assert centre is not None and centre[:, 1].max() < 200
