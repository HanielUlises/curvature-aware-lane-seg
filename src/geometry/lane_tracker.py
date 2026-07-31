"""Temporal tracking of the two ego-lane boundaries in image space.

Everything upstream of this module is per-frame and memoryless: the segmenter emits a
mask, :mod:`src.geometry.centerline` decomposes it into boundaries, and the ego
centreline is the midpoint of whichever pair happens to bracket the camera axis in
*that* frame. On a recorded sequence the lane count changed on 71 of 99 frame
transitions, so the pair being averaged was frequently not the same pair as the frame
before, and the centreline moved even where the road did not.

The downstream Kalman filter (:mod:`src.geometry.temporal`) cannot repair this. It
smooths the three scalars read off the centreline, so it sees a jump in lateral offset
identically whether the vehicle moved or the boundary set changed underneath it. This
module works one stage earlier, on the boundaries themselves, which is where the
frame-to-frame identity actually lives.

Two things are tracked per boundary, both on a fixed grid of image rows:

- **column**, smoothed by an exponential moving average, so detection jitter on a
  boundary that is genuinely stationary does not reach the centreline;
- **extent**, by carrying rows the current frame failed to detect for a bounded number
  of frames. This is what stops the drawn line from growing and shrinking: the near end
  of a boundary is the first thing a segmenter loses and the last thing it recovers.

A track is *not* smoothed across a real change. If the associated boundary jumps
laterally beyond a gate the track is reset to the new observation, so a lane change or
a switch to a different physical boundary is followed immediately rather than blended
through. The gate is what separates tracking from smearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.centerline import (
    DEFAULT_MAX_EXTEND_ROWS,
    ego_lane_pair,
    resample_boundary,
)

FloatArray = np.ndarray

# Rows sampled across the frame for the tracked column profile. Finer than the number
# of centreline points, so resampling the track back out costs no resolution.
DEFAULT_TRACK_ROWS = 96
# Weight given to the new observation. This is a trade, and measuring only stability gets
# it wrong: a line frozen at a stale position is perfectly stable and also wrong. Over a
# 100-frame TuSimple run at 20 fps, against the midpoint of the boundaries observed in each
# frame, the drawn line deviates 7.2 px at 0.30, 5.5 px at 0.45 and 3.8 px at 0.60, while
# its near end moves 5.8, 8.7 and 9.5 px per frame respectively (25.3 px untracked). 0.60
# keeps most of the steadiness at a third of the drift.
DEFAULT_ALPHA = 0.60
# A boundary further than this from its track (median over shared rows, in pixels) is
# treated as a different boundary rather than as noise, and resets the track. At 40 px the
# same run reset 26 times and at 55 px 10 times, none of which corresponded to a real lane
# change; at 80 px it resets twice. A gate that fires on ordinary noise defeats the
# tracking, so it is set wide enough to fire only on a genuine change of boundary.
DEFAULT_GATE_PX = 80.0
# How long a row with no observation may be carried before it is dropped. One frame, and
# for flicker only. A longer coast was tried at 3 and 8 frames and both drew the centreline
# far past any evidence: on a frame where the segmenter saw the right boundary over 33 rows,
# an 8-frame coast still drew 200 rows of it from columns measured up to 0.4 s earlier, and
# the resulting lateral offset read -2.1 m where the untracked estimate read -0.5 m. Holding
# a boundary the camera can no longer see does not make the estimate steadier, it makes it
# wrong in a way that looks steady.
DEFAULT_MAX_COAST_FRAMES = 1
# Minimum plausible separation between the two boundaries, in pixels. Independently
# tracked boundaries can cross near the vanishing point, where both are extrapolated;
# a crossed or near-coincident pair has no midpoint worth drawing.
DEFAULT_MIN_WIDTH_PX = 8.0
# Largest lateral movement of the centreline per image row, in pixels, before the
# estimate is treated as broken rather than curved. Where one boundary is observed and
# the other extrapolated, the midpoint can step sideways: measured on this footage a
# genuine centreline moves under 2.5 px per row, while the artefact stepped 9 px per row,
# which the renderer draws as a detached fragment because consecutive points stop
# overlapping. The line is truncated at such a step instead of being drawn through it.
DEFAULT_MAX_LATERAL_SLOPE = 4.0
# Frames over which the drawn extent is taken as a median. The near end of the centreline
# follows whatever the mask found, which on this footage jumps by a mean of 21 image rows
# per frame and reads as the line surging and retreating. A median over the last few frames
# rejects the one-frame spikes without introducing lag: where the evidence is steady the
# median equals it exactly, unlike a growth rate limit, which was tried first and left the
# drawn line trailing the evidence by 30 to 50 rows permanently because it never caught up.
# The result is still clipped to what this frame supports, so the line never outruns the
# evidence and retraction stays immediate.
DEFAULT_EXTENT_MEDIAN_FRAMES = 3
# Frames of lane width kept for the plausibility check, and the band a new pair's width
# must fall in relative to their median. A pair a whole lane too wide is not the ego lane,
# however well one of its boundaries happens to match.
# Shortest centreline worth reporting, in image rows. A stub of a few rows carries no
# usable heading and, drawn, is invisible inside the near-end fade while still counting as
# a detection: reporting nothing is both more honest and less confusing to look at.
DEFAULT_MIN_CENTERLINE_ROWS = 25.0
DEFAULT_WIDTH_HISTORY = 15
DEFAULT_WIDTH_TOLERANCE = (0.65, 1.5)


# Half-width, in track rows, of the spatial smoothing applied along each boundary. A lane
# boundary is a smooth curve, so small row-to-row wobble in the tracked columns is
# estimation noise rather than geometry, and it renders as a visible zigzag. Smoothing
# along the boundary is a different operation from the temporal average: one says
# neighbouring rows of the same boundary agree, the other says the same row agrees between
# frames.
DEFAULT_SMOOTH_HALFWIDTH = 2


def _smooth_columns(cols: FloatArray, half: int) -> FloatArray:
    """Moving average along rows, applied within each run of observed rows.

    Averaging across a gap would drag columns towards rows the boundary does not cover,
    so each contiguous run is smoothed on its own and short runs are left alone.
    """
    if half < 1:
        return cols
    out = cols.copy()
    finite = np.isfinite(cols)
    width = 2 * half + 1
    kernel = np.ones(width) / width
    start = None
    for i in range(len(cols) + 1):
        inside = i < len(cols) and finite[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            seg = cols[start:i]
            if seg.size >= width:
                # Pad by edge replication so the ends are not pulled towards zero.
                padded = np.concatenate([np.full(half, seg[0]), seg, np.full(half, seg[-1])])
                out[start:i] = np.convolve(padded, kernel, mode="valid")
            start = None
    return out


def _longest_run(valid: np.ndarray) -> slice | None:
    """Longest contiguous run of ``True`` in a boolean array, as a slice.

    Rows retire from a track independently, so the surviving rows can be split into
    islands with holes between them. Drawing every surviving row renders the centreline
    as disconnected fragments, which reads as far less stable than the untracked line
    even when each fragment is individually steadier. Only one contiguous run is kept.
    """
    best_start = best_len = 0
    start = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if start is not None and len(valid) - start > best_len:
        best_start, best_len = start, len(valid) - start
    if best_len == 0:
        return None
    return slice(best_start, best_start + best_len)


@dataclass
class _BoundaryTrack:
    """Smoothed column profile of one boundary over a fixed row grid."""

    rows: FloatArray
    columns: FloatArray = field(default=None)  # type: ignore[assignment]
    # Frames since each row last had an observation; only meaningful where finite.
    staleness: FloatArray = field(default=None)  # type: ignore[assignment]
    resets: int = 0

    def __post_init__(self) -> None:
        if self.columns is None:
            self.columns = np.full(self.rows.shape, np.nan)
        if self.staleness is None:
            self.staleness = np.zeros(self.rows.shape, dtype=np.int64)

    @property
    def alive(self) -> bool:
        return bool(np.any(np.isfinite(self.columns)))

    def _adopt(self, observed: FloatArray) -> None:
        self.columns = observed.copy()
        self.staleness = np.where(np.isfinite(observed), 0, 0).astype(np.int64)

    def displacement(self, observed: FloatArray) -> float | None:
        """Median lateral distance between an observation and the current track."""
        both = np.isfinite(observed) & np.isfinite(self.columns)
        if not np.any(both):
            return None
        return float(np.median(np.abs(observed[both] - self.columns[both])))

    def update(
        self, observed: FloatArray | None, alpha: float, gate_px: float,
        max_coast_frames: int,
        smooth_halfwidth: int = DEFAULT_SMOOTH_HALFWIDTH,
        min_centerline_rows: float = DEFAULT_MIN_CENTERLINE_ROWS,
        width_history: int = DEFAULT_WIDTH_HISTORY,
        width_tolerance: tuple[float, float] = DEFAULT_WIDTH_TOLERANCE,
    ) -> None:
        """Fold one frame's observation into the track.

        Args:
            observed: Columns on the track's row grid, ``nan`` where unobserved, or
                ``None`` when the boundary was not detected at all this frame.
            alpha: Weight on the observation in the moving average.
            gate_px: Association gate; a larger displacement resets the track.
            max_coast_frames: Rows unobserved for longer than this are dropped.
        """
        if observed is None:
            observed = np.full(self.rows.shape, np.nan)

        if not self.alive:
            self._adopt(observed)
            return

        gap = self.displacement(observed)
        if gap is not None and gap > gate_px:
            # A different boundary, not a noisy version of this one. Follow it.
            self._adopt(observed)
            self.resets += 1
            return

        seen = np.isfinite(observed)
        tracked = np.isfinite(self.columns)

        blended = np.where(
            seen & tracked, (1.0 - alpha) * self.columns + alpha * observed,
            np.where(seen, observed, self.columns),
        )
        # Age every row that went unobserved; retire the ones carried too long.
        self.staleness = np.where(seen, 0, self.staleness + 1)
        blended = np.where(self.staleness > max_coast_frames, np.nan, blended)
        self.columns = _smooth_columns(blended, smooth_halfwidth)

    def polyline(self) -> FloatArray | None:
        """The track as an ``(N, 2)`` ``(x, y)`` polyline, or ``None`` if empty."""
        valid = np.isfinite(self.columns)
        if valid.sum() < 2:
            return None
        return np.column_stack([self.columns[valid], self.rows[valid]])


@dataclass
class TrackedBoundaries:
    """One frame's tracker output.

    Attributes:
        left: Smoothed left ego boundary, or ``None`` if no track is alive.
        right: Smoothed right ego boundary, or ``None``.
        observed: Whether this frame contributed an observation to both tracks.
        coasting_frames: Consecutive frames with no observation, 0 when observed.
        resets: Cumulative association resets across both tracks.
    """

    left: FloatArray | None
    right: FloatArray | None
    observed: bool
    coasting_frames: int
    resets: int


class EgoBoundaryTracker:
    """Associates and smooths the ego lane's two boundaries across frames.

    The tracker holds no motion model. A boundary's apparent motion between frames
    comes from vehicle motion the tracker cannot observe, so predicting it would mean
    inventing ego-motion; the moving average and the bounded coast are deliberately the
    weakest assumptions that still give the centreline frame-to-frame identity.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        track_rows: int = DEFAULT_TRACK_ROWS,
        alpha: float = DEFAULT_ALPHA,
        gate_px: float = DEFAULT_GATE_PX,
        max_coast_frames: int = DEFAULT_MAX_COAST_FRAMES,
        max_extend_rows: int = DEFAULT_MAX_EXTEND_ROWS,
        min_width_px: float = DEFAULT_MIN_WIDTH_PX,
        max_lateral_slope: float = DEFAULT_MAX_LATERAL_SLOPE,
        extent_median_frames: int = DEFAULT_EXTENT_MEDIAN_FRAMES,
        smooth_halfwidth: int = DEFAULT_SMOOTH_HALFWIDTH,
        min_centerline_rows: float = DEFAULT_MIN_CENTERLINE_ROWS,
        width_history: int = DEFAULT_WIDTH_HISTORY,
        width_tolerance: tuple[float, float] = DEFAULT_WIDTH_TOLERANCE,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.alpha = alpha
        self.gate_px = gate_px
        self.max_coast_frames = max_coast_frames
        self.max_extend_rows = max_extend_rows
        self.min_width_px = min_width_px
        self.max_lateral_slope = max_lateral_slope
        self.extent_median_frames = max(int(extent_median_frames), 1)
        self.smooth_halfwidth = smooth_halfwidth
        self.min_centerline_rows = min_centerline_rows
        self.width_history = max(int(width_history), 1)
        self.width_tolerance = width_tolerance
        rows = np.linspace(0.0, float(image_height - 1), track_rows)
        self._left = _BoundaryTrack(rows=rows)
        self._right = _BoundaryTrack(rows=rows)
        self._coasting = 0
        self._recent_bottoms: list[float] = []
        self._width_profile: FloatArray | None = None

    @property
    def rows(self) -> FloatArray:
        return self._left.rows

    def update(self, polylines: list[FloatArray]) -> TrackedBoundaries:
        """Fold one frame's boundaries into the tracks and return the smoothed pair.

        Args:
            polylines: Lane polylines from
                :func:`src.geometry.centerline.extract_lane_polylines`.

        Returns:
            A :class:`TrackedBoundaries` for this frame.
        """
        candidates = [
            resample_boundary(p, self.rows, self.max_extend_rows) for p in polylines
        ]
        pair = ego_lane_pair(polylines, self.image_width)
        fallback = None
        if pair is not None:
            fallback = tuple(
                resample_boundary(p, self.rows, self.max_extend_rows) for p in pair
            )
        observed_left, observed_right = self._associate(candidates, fallback)

        observed = observed_left is not None or observed_right is not None
        self._coasting = 0 if observed else self._coasting + 1

        self._left.update(observed_left, self.alpha, self.gate_px,
                          self.max_coast_frames, self.smooth_halfwidth)
        self._right.update(observed_right, self.alpha, self.gate_px,
                           self.max_coast_frames, self.smooth_halfwidth)
        return TrackedBoundaries(
            left=self._left.polyline(),
            right=self._right.polyline(),
            observed=observed,
            coasting_frames=self._coasting,
            resets=self._left.resets + self._right.resets,
        )

    def _associate(
        self, candidates: list[FloatArray], fallback: tuple[FloatArray, FloatArray] | None
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """Match this frame's boundaries to the ones already being followed.

        The selection rule this replaces took, on each side, whichever boundary sat
        nearest the camera axis in *that* frame. When the segmenter loses the ego lane's
        own marking, the next marking out wins instead, and the centreline steps sideways
        by half a lane and steps back when the marking returns. Measured on the demo clip
        that rule moved a chosen boundary by more than 90 px on four of sixty frames.

        Matching to the track means a boundary has to look like the one already being
        followed. Where no candidate matches a side, the bracketing rule supplies it, so
        a track that is merely late is not starved; what the gate prevents is silently
        adopting a boundary a lane away as though it were the same one.

        Returns:
            ``(left, right)`` columns on the track grid, either entry ``None`` when
            nothing acceptable was found for that side.
        """
        def best(track: _BoundaryTrack) -> tuple[FloatArray | None, float]:
            if not track.alive or not candidates:
                return None, np.inf
            scored = [(track.displacement(c), i) for i, c in enumerate(candidates)]
            scored = [(d, i) for d, i in scored if d is not None and d <= self.gate_px]
            if not scored:
                return None, np.inf
            d, i = min(scored)
            return candidates[i], d

        left, ld = best(self._left)
        right, rd = best(self._right)

        # The same detection cannot be both boundaries; keep the better match.
        if left is not None and right is not None and left is right:
            if ld <= rd:
                right, rd = None, np.inf
            else:
                left, ld = None, np.inf

        # Anything unmatched falls back to the bracketing rule.
        if fallback is not None:
            if left is None:
                left = fallback[0]
            if right is None:
                right = fallback[1]

        if left is None or right is None:
            return left, right

        # A gate alone does not stop the right track matching the left boundary when the
        # two are close; the pair has to stay a pair. If the order is violated, neither
        # match is trustworthy, so take the bracketing pair or nothing.
        med_l, med_r = np.nanmedian(left), np.nanmedian(right)
        if not (np.isfinite(med_l) and np.isfinite(med_r)) or med_l >= med_r:
            if fallback is None:
                return None, None
            left, right = fallback
            ld = rd = np.inf

        width = right - left
        shared = np.isfinite(width)
        if not np.any(shared) or np.nanmedian(width[shared]) <= self.min_width_px:
            return None, None

        # Compare like for like: lane width in pixels grows steeply towards the vehicle,
        # so a scalar width says nothing unless both are measured on the same rows.
        if self._width_profile is not None:
            both = shared & np.isfinite(self._width_profile)
            if np.count_nonzero(both) >= 3:
                ratio = float(np.median(width[both] / self._width_profile[both]))
                lo, hi = self.width_tolerance
                if not (lo <= ratio <= hi):
                    # This pair is not the lane that was being followed. Drop the side
                    # that matched worse rather than accept it.
                    if ld <= rd:
                        return left, None
                    return None, right

        self._width_profile = np.where(shared, width, np.nan)
        return left, right

    def centerline(self, tracked: TrackedBoundaries, num_points: int = 50):
        """Centreline midway between the two tracked boundaries.

        Args:
            tracked: The output of :meth:`update`.
            num_points: Points to sample along the shared row range.

        Returns:
            An ``(M, 2)`` centreline ordered top-to-bottom, or ``None`` when either
            track is empty or too few rows are shared.
        """
        if tracked.left is None or tracked.right is None:
            return None
        top = max(tracked.left[:, 1].min(), tracked.right[:, 1].min())
        bottom = min(tracked.left[:, 1].max(), tracked.right[:, 1].max())
        if bottom <= top:
            return None
        rows = np.linspace(top, bottom, num_points)
        # No further extension here: the tracks were already built from boundaries that
        # were extended once, and extending an extrapolation compounds the error.
        xl = resample_boundary(tracked.left, rows, max_extend_rows=0)
        xr = resample_boundary(tracked.right, rows, max_extend_rows=0)
        centre = (xl + xr) / 2.0
        # Only where the pair is still a plausible lane: the two tracks are maintained
        # independently and can cross where both are extrapolated.
        usable = np.isfinite(centre) & ((xr - xl) > self.min_width_px)

        # Break the line wherever it steps sideways much faster than it has been. The
        # limit adapts to the line's own typical bend rather than being fixed: under
        # perspective a legitimate centreline sweeps sideways several times faster near
        # the vehicle than near the vanishing point, and a fixed limit tuned for the far
        # field truncates a perfectly good near field. What it must still catch is the
        # step change where one boundary is observed and the other extrapolated, which is
        # an order of magnitude above the local trend rather than a factor of two.
        step = np.abs(np.diff(centre, prepend=centre[0]))
        row_step = max(float(rows[1] - rows[0]), 1e-6)
        slope = np.nan_to_num(step / row_step, nan=np.inf)
        finite = slope[np.isfinite(slope) & np.isfinite(centre)]
        typical = float(np.median(finite)) if finite.size else 0.0
        usable &= slope <= max(self.max_lateral_slope, 3.0 * typical)

        run = _longest_run(usable)
        if run is None or run.stop - run.start < 3:
            return None

        out_rows, out_cols = rows[run], centre[run]

        available = float(out_rows.max())
        self._recent_bottoms.append(available)
        del self._recent_bottoms[: -self.extent_median_frames]
        # Never beyond what this frame supports; a median can only pull the extent in.
        target = min(float(np.median(self._recent_bottoms)), available)
        keep = out_rows <= target
        if keep.sum() >= 3:
            out_rows, out_cols = out_rows[keep], out_cols[keep]
        if out_rows.max() - out_rows.min() < self.min_centerline_rows:
            return None
        return np.column_stack([out_cols, out_rows])
