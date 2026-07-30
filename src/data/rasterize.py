"""Rasterize CurveLanes polylines to binary lane masks.

The invariant that governs this module: **rasterize at native resolution, then
resize the mask to the target size with nearest-neighbour interpolation.** Doing
it the other way round (resize the polyline coordinates, then draw) fragments
thin, near-horizontal lines into dashes and biases curvature. OpenCV's line
drawing clips off-frame points and handles near-vertical segments natively, so
no special-casing is needed for polylines that exit the frame.

Stroke width is specified in *target* pixels and scaled up to native before
drawing, so the rendered line has a resolution-independent thickness and thin
lanes survive the downscale.

All frames handed here are expected to share the target aspect ratio; the
odd-aspect CurveLanes frames (1570x660) are excluded upstream. :func:`rasterize_frame`
enforces that contract loudly via :func:`assert_target_aspect` rather than
silently applying a distorting non-uniform resize.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from src.data.curvelanes import FrameAnnotation, Lane

Mask = npt.NDArray[np.uint8]

# (width, height) in pixels.
DEFAULT_TARGET_SIZE: tuple[int, int] = (512, 288)
DEFAULT_STROKE_PX: int = 5
DEFAULT_ASPECT_TOL: float = 0.02

_FOREGROUND: int = 255


def is_target_aspect(
    width: int,
    height: int,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    tol: float = DEFAULT_ASPECT_TOL,
) -> bool:
    """Return whether a native ``(width, height)`` matches the target aspect.

    Args:
        width: Native image width in pixels.
        height: Native image height in pixels.
        target_size: Target ``(width, height)`` whose aspect ratio is the reference.
        tol: Maximum allowed relative difference in aspect ratio.

    Returns:
        ``True`` if ``width / height`` is within ``tol`` (relative) of the target
        aspect ratio. Used at the subset stage to filter out odd-aspect frames.
    """
    target_w, target_h = target_size
    native_ratio = width / height
    target_ratio = target_w / target_h
    return abs(native_ratio - target_ratio) / target_ratio <= tol


def assert_target_aspect(
    frame: FrameAnnotation,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    tol: float = DEFAULT_ASPECT_TOL,
) -> None:
    """Raise if ``frame`` does not share the target aspect ratio.

    Guards the exclusion contract: only frames whose aspect matches the target
    may be rasterized, because resizing a mismatched frame to ``target_size``
    applies a non-uniform scale that distorts curvature.

    Raises:
        ValueError: If the frame's native aspect ratio is outside ``tol``.
    """
    if not is_target_aspect(frame.width, frame.height, target_size, tol):
        target_ratio = target_size[0] / target_size[1]
        raise ValueError(
            f"{frame.image_path.name}: native {frame.width}x{frame.height} "
            f"(ratio {frame.width / frame.height:.3f}) does not match target aspect "
            f"{target_ratio:.3f} within tol {tol}. Odd-aspect frames must be excluded "
            f"upstream, not resized here."
        )


def _native_stroke(stroke_px: int, native_width: int, target_width: int) -> int:
    """Scale a target-space stroke width to native pixels (>= 1)."""
    scaled = round(stroke_px * native_width / target_width)
    return max(1, scaled)


def rasterize_lanes(
    lanes: list[Lane],
    native_size: tuple[int, int],
    stroke_native_px: int,
) -> Mask:
    """Draw lanes onto a native-resolution binary canvas.

    Args:
        lanes: Lane polylines in native pixel coordinates.
        native_size: Native ``(width, height)`` of the canvas.
        stroke_native_px: Line thickness in native pixels.

    Returns:
        A ``uint8`` mask of shape ``(height, width)`` with lanes at ``255`` on a
        ``0`` background. Off-frame polyline points are clipped by OpenCV.
    """
    width, height = native_size
    mask: Mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        # cv2.polylines wants int32 (N, 1, 2) point arrays; rounding to the
        # native grid is the actual rasterization step.
        pts = np.round(lane.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [pts],
            isClosed=False,
            color=_FOREGROUND,
            thickness=stroke_native_px,
            lineType=cv2.LINE_8,
        )
    return mask


def rasterize_frame(
    frame: FrameAnnotation,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    stroke_px: int = DEFAULT_STROKE_PX,
    aspect_tol: float = DEFAULT_ASPECT_TOL,
) -> Mask:
    """Rasterize a frame's lanes to a target-resolution binary mask.

    Draws at native resolution and then resizes to ``target_size`` with
    nearest-neighbour interpolation, preserving thin structures. The frame must
    share the target aspect ratio (see :func:`assert_target_aspect`).

    Args:
        frame: Parsed frame annotation with native resolution and lanes.
        target_size: Output ``(width, height)`` in pixels.
        stroke_px: Line thickness in *target* pixels; scaled to native for drawing.
        aspect_tol: Relative aspect-ratio tolerance for the exclusion guard.

    Returns:
        A ``uint8`` mask of shape ``(target_height, target_width)``, lanes at
        ``255`` on a ``0`` background.

    Raises:
        ValueError: If the frame's aspect ratio does not match the target.
    """
    assert_target_aspect(frame, target_size, aspect_tol)

    target_w, target_h = target_size
    native_mask = rasterize_native(frame, target_size, stroke_px, aspect_tol)

    if (frame.width, frame.height) == (target_w, target_h):
        return native_mask

    resized = cv2.resize(
        native_mask,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.uint8)


def rasterize_native(
    frame: FrameAnnotation,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    stroke_px: int = DEFAULT_STROKE_PX,
    aspect_tol: float = DEFAULT_ASPECT_TOL,
) -> Mask:
    """Rasterize a frame's lanes at **native** resolution (no resize).

    Returned so callers can apply a native-resolution sky crop *before* the final
    resize (see :mod:`src.data.transforms`); resizing first would blur the crop
    seam and shrink thin lines twice. Stroke width is scaled from ``target_size``
    so the post-resize thickness is resolution-independent.

    Args:
        frame: Parsed frame annotation.
        target_size: Target ``(w, h)`` — used only to scale the stroke and to
            enforce the aspect guard.
        stroke_px: Line thickness in target pixels.
        aspect_tol: Aspect-ratio tolerance for the exclusion guard.

    Returns:
        A native-resolution ``uint8`` mask of shape ``(frame.height, frame.width)``.

    Raises:
        ValueError: If the frame's aspect ratio does not match the target.
    """
    assert_target_aspect(frame, target_size, aspect_tol)
    stroke_native = _native_stroke(stroke_px, frame.width, target_size[0])
    return rasterize_lanes(frame.lanes, (frame.width, frame.height), stroke_native)
