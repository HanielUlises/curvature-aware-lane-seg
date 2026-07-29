"""Inverse-perspective mapping: image pixels to a metric ground plane.

Roadmap step two. Curvature and lateral offset measured in the image plane are
perspective-distorted (a straight road curves toward the vanishing point), so the
controller cannot use them directly. Assuming locally flat ground, a homography
maps image points ``(u, v)`` to a bird's-eye ground plane ``(x, z)`` where ``x`` is
lateral position and ``z`` is distance ahead, both in metres.

The homography is solved with the Direct Linear Transform (a pure-NumPy SVD, so the
math ports to Eigen in ``deploy/`` under the same golden-vector contract as the
curvature estimator) from four point correspondences.

Because neither CurveLanes nor TuSimple ships camera calibration, the default
image-to-ground correspondence is a **placeholder**: a trapezoid on the road plane
mapped to a rectangle of assumed lane width and look-ahead. It gives geometrically
consistent BEV output, but absolute metres are only correct once real extrinsics
replace it (see :func:`build_ground_homography`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def homography_from_points(src: FloatArray, dst: FloatArray) -> FloatArray:
    """Solve the 3x3 homography mapping ``src`` points to ``dst`` (DLT).

    Args:
        src: Source points, shape ``(N, 2)`` with ``N >= 4``.
        dst: Destination points, shape ``(N, 2)``, paired with ``src``.

    Returns:
        The ``3x3`` homography ``H`` (normalized so ``H[2, 2] == 1``) with
        ``dst ~ H @ [src, 1]`` in homogeneous coordinates.

    Raises:
        ValueError: If fewer than four correspondences are given.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 4 or dst.shape[0] != src.shape[0]:
        raise ValueError("need >= 4 paired correspondences")

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)
    # Homography is the right singular vector of the smallest singular value.
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def apply_homography(h: FloatArray, points: FloatArray) -> FloatArray:
    """Apply a homography to ``(N, 2)`` points.

    Args:
        h: A ``3x3`` homography.
        points: Points to transform, shape ``(N, 2)``.

    Returns:
        Transformed points, shape ``(N, 2)``; the homogeneous coordinate is
        divided out (with a small floor to avoid division by zero at the horizon).
    """
    pts = np.asarray(points, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = homog @ h.T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


@dataclass(frozen=True)
class GroundPlane:
    """A configured image-to-ground mapping.

    Attributes:
        h: Image-to-ground ``3x3`` homography.
        h_inv: Ground-to-image inverse, for drawing BEV results back on the frame.
    """

    h: FloatArray
    h_inv: FloatArray

    def to_ground(self, image_points: FloatArray) -> FloatArray:
        """Map image ``(u, v)`` pixels to ground ``(x, z)`` metres."""
        return apply_homography(self.h, image_points)

    def to_image(self, ground_points: FloatArray) -> FloatArray:
        """Map ground ``(x, z)`` metres back to image ``(u, v)`` pixels."""
        return apply_homography(self.h_inv, ground_points)


def build_ground_homography(
    image_size: tuple[int, int],
    src_trapezoid: tuple[tuple[float, float], ...],
    lane_width_m: float = 3.7,
    look_ahead_m: float = 30.0,
) -> GroundPlane:
    """Build a placeholder flat-ground mapping from a road-plane trapezoid.

    The four ``src_trapezoid`` points (fractions of image width/height, ordered
    top-left, top-right, bottom-right, bottom-left of the road region) are mapped to
    a ground rectangle ``lane_width_m`` wide and ``look_ahead_m`` deep, centred on
    the camera axis. This encodes "these four road points form a rectangle on flat
    ground", the standard uncalibrated IPM assumption.

    Args:
        image_size: ``(width, height)`` in pixels.
        src_trapezoid: Four ``(fx, fy)`` fractional image points on the road plane.
        lane_width_m: Width the trapezoid's base spans, in metres.
        look_ahead_m: Distance ahead the trapezoid's top reaches, in metres.

    Returns:
        A :class:`GroundPlane` with forward and inverse homographies.

    Raises:
        ValueError: If ``src_trapezoid`` does not have exactly four points.
    """
    if len(src_trapezoid) != 4:
        raise ValueError("src_trapezoid needs exactly 4 points")
    w, h = image_size
    src = np.array([[fx * w, fy * h] for fx, fy in src_trapezoid], dtype=np.float64)
    half = lane_width_m / 2.0
    # Ground frame: x lateral (right +), z ahead (far +). Match trapezoid ordering:
    # top-left, top-right, bottom-right (near), bottom-left (near).
    dst = np.array(
        [
            [-half, look_ahead_m],
            [half, look_ahead_m],
            [half, 0.0],
            [-half, 0.0],
        ],
        dtype=np.float64,
    )
    hmat = homography_from_points(src, dst)
    return GroundPlane(h=hmat, h_inv=np.linalg.inv(hmat))
