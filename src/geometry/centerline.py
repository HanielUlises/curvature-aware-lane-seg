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
) -> FloatArray | None:
    """Centerline of the ego lane: midpoint of the two lanes bracketing the view.

    Averages the columns of the bracketing lane pair over the row range they share,
    on a uniform grid of rows so the two lanes, sampled at different rows, still
    combine cleanly.

    Args:
        polylines: Lane polylines from :func:`extract_lane_polylines`.
        image_width: Frame width in pixels; its half is the ego reference column.
        num_points: Number of points to sample along the shared row range.

    Returns:
        An ``(num_points, 2)`` centerline ordered top-to-bottom, or ``None`` if no
        left/right pair brackets the ego column.
    """
    pair = ego_lane_pair(polylines, image_width)
    if pair is None:
        return None
    left_lane, right_lane = pair

    y0 = max(left_lane[:, 1].min(), right_lane[:, 1].min())
    y1 = min(left_lane[:, 1].max(), right_lane[:, 1].max())
    if y1 <= y0:
        return None

    ys = np.linspace(y0, y1, num_points)
    # np.interp needs increasing x; the polylines are ordered by y already.
    xl = np.interp(ys, left_lane[:, 1], left_lane[:, 0])
    xr = np.interp(ys, right_lane[:, 1], right_lane[:, 0])
    return np.column_stack([(xl + xr) / 2.0, ys])


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
