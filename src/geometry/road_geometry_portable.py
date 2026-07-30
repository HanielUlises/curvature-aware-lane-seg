"""Portable road geometry — the deployment reference for offset, heading, curvature.

:mod:`src.geometry.road_geometry` is the training-side implementation and reaches
curvature through scipy/FITPACK. This module computes the same three control
quantities from the same ground-plane centreline using only the portable
cubic spline of :mod:`src.geometry.curvature_portable`, so it can be mirrored in C++
(``deploy/src/road_geometry.cpp``) and pinned by golden vectors.

The near-field line fit that produces lateral offset and heading is shared with the
training implementation, not duplicated: it is plain least squares with no library
dependency worth avoiding.

Ground convention matches the training module: ``x`` lateral (right positive), ``z``
ahead, and curvature signed **right-positive**, the negation of the counter-clockwise
convention the curvature formula returns under this axis layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.curvature_portable import (
    sample_positions_portable,
    signed_curvature_along_portable,
)
from src.geometry.road_geometry import (
    DEFAULT_OFFSET_DISTANCE_M,
    DEFAULT_PREVIEW_M,
    fit_near_line,
)

FloatArray = np.ndarray

DEFAULT_NUM_SAMPLES = 100


@dataclass(frozen=True)
class PortableRoadGeometry:
    """Control quantities read off a ground-plane centreline.

    Attributes:
        lateral_offset_m: Lane-centre lateral position at ``offset_distance_m``.
        offset_distance_m: Distance ahead at which the offset was evaluated.
        heading_error_rad: Angle of the near-field centreline from ``+z``.
        curvature_1pm: Median signed curvature along the centreline, right positive.
        preview_distances_m: Look-ahead distances for ``preview_curvature_1pm``.
        preview_curvature_1pm: Signed curvature at each preview distance; ``nan``
            where the distance falls outside the reconstructed centreline.
    """

    lateral_offset_m: float
    offset_distance_m: float
    heading_error_rad: float
    curvature_1pm: float
    preview_distances_m: FloatArray
    preview_curvature_1pm: FloatArray


def portable_road_geometry(
    ground_centerline: FloatArray,
    preview_distances_m: tuple[float, ...] = DEFAULT_PREVIEW_M,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    offset_distance_m: float = DEFAULT_OFFSET_DISTANCE_M,
) -> PortableRoadGeometry | None:
    """Read offset, heading, and curvature from a ground-plane centreline.

    Args:
        ground_centerline: ``(N, 2)`` centreline ``(x, z)`` in metres, any order in
            ``z``; it is sorted near-to-far internally.
        preview_distances_m: Look-ahead distances at which to report curvature.
        num_samples: Samples along the spline for the curvature estimate.
        offset_distance_m: Distance ahead at which to report lateral offset.

    Returns:
        A :class:`PortableRoadGeometry`, or ``None`` if fewer than three points are
        given or the near-field fit is degenerate.
    """
    ground = np.asarray(ground_centerline, dtype=np.float64)
    if ground.shape[0] < 3:
        return None
    ground = ground[np.argsort(ground[:, 1])]

    fit = fit_near_line(ground)
    if fit is None:
        return None
    intercept, slope = fit
    previews = np.asarray(preview_distances_m, dtype=np.float64)

    # Negate the counter-clockwise convention so that right turns are positive.
    kappa = -signed_curvature_along_portable(ground, num_samples)
    positions = sample_positions_portable(ground, num_samples)
    if kappa.size == 0 or positions.shape[0] != kappa.size:
        preview_kappa = np.full(previews.shape, np.nan)
        representative = 0.0
    else:
        # Curvature is indexed by spline parameter, so interpolate it against the
        # depth of the same samples to answer "curvature at z metres ahead".
        order = np.argsort(positions[:, 1])
        preview_kappa = np.interp(
            previews, positions[order, 1], kappa[order], left=np.nan, right=np.nan
        )
        representative = float(np.median(kappa))

    return PortableRoadGeometry(
        lateral_offset_m=intercept + slope * offset_distance_m,
        offset_distance_m=float(offset_distance_m),
        heading_error_rad=float(np.arctan(slope)),
        curvature_1pm=representative,
        preview_distances_m=previews,
        preview_curvature_1pm=preview_kappa,
    )
