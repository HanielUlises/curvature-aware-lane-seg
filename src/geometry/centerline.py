"""Extract ordered lane polylines and the ego centerline from a lane mask.

This is step one of the perception-to-control chain: the segmenter emits a binary
lane mask, but the geometry module (:mod:`src.geometry.curvature`) and, downstream,
the controller work on ordered ``(x, y)`` polylines. This module turns the mask
back into those polylines and derives the drivable centerline the controller
tracks.

Lanes in a forward-facing view run roughly top-to-bottom (vanishing point to
bumper), so each connected component is reduced to one point per image row via the
row-wise centroid of its pixels. That yields a polyline ordered by row, which is
exactly the ``(N, 2)`` shape :func:`src.geometry.curvature.lane_curvature` expects.

Everything here is in **image pixels**. Metric ground coordinates come after the
inverse-perspective projection (roadmap step two); curvature and lateral offset in
image space are perspective-distorted and only a proxy until then.
"""

from __future__ import annotations

import cv2
import numpy as np

FloatArray = np.ndarray

# A component must span at least this many rows to be a lane rather than a speck.
DEFAULT_MIN_ROWS = 8
# ...and carry at least this many foreground pixels.
DEFAULT_MIN_PIXELS = 40


def extract_lane_polylines(
    mask: np.ndarray,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
) -> list[FloatArray]:
    """Reduce a binary lane mask to one ordered polyline per lane.

    Each connected component is treated as a lane and collapsed to its row-wise
    centroid: for every image row the component occupies, the mean column of its
    foreground pixels becomes one ``(x, y)`` point. Points are ordered by row
    (top of image first), so the result is monotonic in ``y``.

    Args:
        mask: Binary mask of shape ``(H, W)``; nonzero is lane.
        min_rows: Discard components spanning fewer rows than this.
        min_pixels: Discard components with fewer foreground pixels than this.

    Returns:
        A list of ``(N, 2)`` float arrays of ``(x, y)`` points, one per lane,
        sorted left-to-right by the polyline's bottom-most (largest-``y``) column.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    polylines: list[FloatArray] = []
    for label in range(1, n_labels):  # 0 is background
        ys, xs = np.where(labels == label)
        if xs.size < min_pixels:
            continue
        rows = np.unique(ys)
        if rows.size < min_rows:
            continue
        # Row-wise mean column. np.bincount over rows gives sum(x) and count per row.
        sums = np.bincount(ys, weights=xs)
        counts = np.bincount(ys)
        cx = sums[rows] / counts[rows]
        poly = np.column_stack([cx, rows.astype(np.float64)])
        polylines.append(poly)

    polylines.sort(key=lambda p: p[np.argmax(p[:, 1]), 0])
    return polylines


def _x_at_bottom(poly: FloatArray) -> float:
    """Column of a polyline's largest-``y`` (nearest-to-camera) point."""
    return float(poly[np.argmax(poly[:, 1]), 0])


# How far a boundary may be extended past its observed extent, in image rows. Lane
# boundaries are locally straight, so a bounded extension is a fair reconstruction; an
# unbounded one would invent geometry.
DEFAULT_MAX_EXTEND_ROWS = 45
# Rows at the end of a boundary used to fit the direction it is extended along.
DEFAULT_FIT_TAIL_ROWS = 25


def resample_boundary(
    poly: FloatArray,
    rows: FloatArray,
    max_extend_rows: int = DEFAULT_MAX_EXTEND_ROWS,
    fit_tail_rows: int = DEFAULT_FIT_TAIL_ROWS,
) -> FloatArray:
    """Sample a boundary at the given image rows, extending a bounded amount.

    Inside the boundary's observed row span this interpolates. Beyond either end it
    continues along a line fitted to that end's last ``fit_tail_rows``, for at most
    ``max_extend_rows``; rows past that are ``nan``.

    This exists because the ego centreline is otherwise defined only over the row range
    the two boundaries happen to share, so one short boundary truncates it. The
    truncation moves frame to frame, which makes the centreline's near end jump in depth
    and reads as the whole line swinging.

    Args:
        poly: ``(N, 2)`` boundary as ``(x, y)``, ordered by increasing row.
        rows: Image rows to sample at.
        max_extend_rows: Maximum rows to continue past the observed span.
        fit_tail_rows: Rows at each end used to fit the extension direction.

    Returns:
        Sampled columns, one per requested row, ``nan`` where unavailable.
    """
    poly = np.asarray(poly, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.full(rows.shape, np.nan)

    y, x = poly[:, 1], poly[:, 0]
    y_lo, y_hi = float(y.min()), float(y.max())
    out = np.interp(rows, y, x, left=np.nan, right=np.nan)

    def extend(mask: np.ndarray, tail: np.ndarray, anchor: float) -> None:
        if not np.any(mask) or tail.shape[0] < 2 or np.ptp(tail[:, 1]) < 1e-9:
            return
        slope, intercept = np.polyfit(tail[:, 1], tail[:, 0], 1)
        within = mask & (np.abs(rows - anchor) <= max_extend_rows)
        out[within] = slope * rows[within] + intercept

    extend(rows < y_lo, poly[y <= y_lo + fit_tail_rows], y_lo)   # further away
    extend(rows > y_hi, poly[y >= y_hi - fit_tail_rows], y_hi)   # nearer the vehicle
    return out


def ego_lane_pair(
    polylines: list[FloatArray], image_width: int
) -> tuple[FloatArray, FloatArray] | None:
    """The two lanes bracketing the camera axis, nearest first on each side.

    Args:
        polylines: Lane polylines from :func:`extract_lane_polylines`.
        image_width: Frame width in pixels; its half is the ego reference column.

    Returns:
        ``(left_lane, right_lane)``, or ``None`` if the ego column is not bracketed.
    """
    if len(polylines) < 2:
        return None
    center_x = image_width / 2.0
    left = [p for p in polylines if _x_at_bottom(p) < center_x]
    right = [p for p in polylines if _x_at_bottom(p) >= center_x]
    if not left or not right:
        return None
    # Nearest lane on each side of the ego column.
    return max(left, key=_x_at_bottom), min(right, key=_x_at_bottom)


def ego_centerline(
    polylines: list[FloatArray],
    image_width: int,
    num_points: int = 50,
    image_height: int | None = None,
    max_extend_rows: int = DEFAULT_MAX_EXTEND_ROWS,
) -> FloatArray | None:
    """Centerline of the ego lane: midpoint of the two lanes bracketing the view.

    Both boundaries are resampled onto one row grid that runs from the further of their
    two far ends down towards the vehicle, each extended by at most
    ``max_extend_rows`` past what it actually covers. Taking only the rows the two
    boundaries share, as an earlier version did, let a single short boundary truncate
    the centreline, and because that truncation moved from frame to frame the
    centreline's near end jumped in depth by metres between neighbouring frames.

    Args:
        polylines: Lane polylines from :func:`extract_lane_polylines`.
        image_width: Frame width in pixels; its half is the ego reference column.
        num_points: Number of points to sample along the row range.
        image_height: Frame height, used as the nearest row to reach towards. Defaults
            to just below the lower boundary extent when not given.
        max_extend_rows: Bound on how far each boundary may be extended.

    Returns:
        An ``(num_points, 2)`` centerline ordered top-to-bottom, or ``None`` if no
        left/right pair brackets the ego column or too little overlap survives.
    """
    pair = ego_lane_pair(polylines, image_width)
    if pair is None:
        return None
    left_lane, right_lane = pair

    top = max(left_lane[:, 1].min(), right_lane[:, 1].min())
    observed_bottom = min(left_lane[:, 1].max(), right_lane[:, 1].max())
    # Reach towards the vehicle by the permitted extension, capped at the frame edge.
    target_bottom = observed_bottom + max_extend_rows
    if image_height is not None:
        target_bottom = min(target_bottom, float(image_height - 1))
    if target_bottom <= top:
        return None

    rows = np.linspace(top, target_bottom, num_points)
    xl = resample_boundary(left_lane, rows, max_extend_rows)
    xr = resample_boundary(right_lane, rows, max_extend_rows)
    centre = (xl + xr) / 2.0
    valid = np.isfinite(centre)
    if valid.sum() < 3:
        return None
    return np.column_stack([centre[valid], rows[valid]])


def image_lateral_offset(centerline: FloatArray, image_width: int) -> float:
    """Signed image-space offset of the ego column from the centerline base.

    Positive means the lane center sits to the right of the camera axis. This is a
    perspective-distorted proxy for the metric lateral offset the controller needs;
    the true value comes from the ground-plane projection (roadmap step two).

    Args:
        centerline: Ego centerline from :func:`ego_centerline`.
        image_width: Frame width in pixels.

    Returns:
        ``centerline_x_at_bottom - image_width / 2`` in pixels.
    """
    base_x = float(centerline[np.argmax(centerline[:, 1]), 0])
    return base_x - image_width / 2.0
