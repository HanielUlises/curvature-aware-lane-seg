"""Metric road geometry for the controller: offset, heading, and curvature.

Roadmap step three. Steps one and two produce an ego centerline in image pixels and
a homography to a flat ground plane. This module composes them and reads the
quantities a kinematic MPC consumes, all in metric ground coordinates:

- **lateral offset** ``e_y``: where the lane centre sits relative to the vehicle axis;
- **heading error** ``e_psi``: the angle of the centreline relative to straight ahead;
- **curvature** ``kappa(z)``: the signed tightness of the lane at preview distances.

Ground convention: ``x`` is lateral (right positive), ``z`` is distance ahead
(positive). The vehicle sits at the origin looking along ``+z``. Curvature is
**signed with right turns positive**, which is the negation of the mathematical
counter-clockwise convention returned by
:func:`src.geometry.curvature.signed_curvature_along` under this axis layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev

from src.geometry.centerline import ego_centerline, extract_lane_polylines
from src.geometry.curvature import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SMOOTHING,
    fit_arclength_spline,
    signed_curvature_along,
)
from src.geometry.ipm import GroundPlane

FloatArray = np.ndarray

# Look-ahead distances (metres) at which curvature is reported to the controller.
DEFAULT_PREVIEW_M = (5.0, 10.0, 20.0)
# Depth span over which the near-field line is fitted for offset and heading.
NEAR_FIT_SPAN_M = 12.0
# Distance ahead at which lateral offset is reported. Nothing is observed at the vehicle
# plane itself: the recovered centreline typically starts about 12 m ahead, so a value
# quoted at z = 0 is extrapolated rather than measured.
DEFAULT_OFFSET_DISTANCE_M = 5.0


@dataclass(frozen=True)
class RoadGeometry:
    """Control-relevant geometry of the ego lane, in metric ground coordinates.

    Attributes:
        ground_centerline: ``(M, 2)`` centreline ``(x, z)`` in metres, near to far.
        lateral_offset_m: Lane-centre lateral position at ``offset_distance_m`` ahead;
            positive means the centre lies to the right of the vehicle axis.
        offset_distance_m: Distance ahead at which the offset was evaluated.
        heading_error_rad: Angle of the centreline tangent from ``+z``, from the
            near-field fit; positive turns to the right.
        curvature_1pm: Representative signed curvature ahead, in ``1/m``; positive
            curves right.
        preview_distances_m: Look-ahead distances for ``preview_curvature_1pm``.
        preview_curvature_1pm: Signed curvature at each preview distance. Entries are
            ``nan`` where the distance falls outside the reconstructed centreline.
    """

    ground_centerline: FloatArray
    lateral_offset_m: float
    offset_distance_m: float
    heading_error_rad: float
    curvature_1pm: float
    preview_distances_m: FloatArray
    preview_curvature_1pm: FloatArray


def fit_near_line(
    ground: FloatArray, span_m: float = NEAR_FIT_SPAN_M
) -> tuple[float, float] | None:
    """Least-squares line ``x = slope * z + intercept`` over the near centreline.

    Offset and heading are both read from this fit rather than from the two nearest
    points. Two points define a line exactly, so any angular noise in them is passed
    straight through, and because the centreline usually begins about 12 m ahead that
    noise is then extrapolated back over a long lever arm. Fitting the near span
    averages it down instead.

    Args:
        ground: ``(N, 2)`` centreline ``(x, z)`` sorted by increasing ``z``.
        span_m: Depth span from the nearest point to include in the fit.

    Returns:
        ``(intercept, slope)``, or ``None`` if fewer than two usable points remain.
    """
    if ground.shape[0] < 2:
        return None
    near = ground[ground[:, 1] <= ground[0, 1] + span_m]
    if near.shape[0] < 3:
        near = ground[: min(3, ground.shape[0])]
    if near.shape[0] < 2 or np.ptp(near[:, 1]) < 1e-9:
        return None
    slope, intercept = np.polyfit(near[:, 1], near[:, 0], 1)
    return float(intercept), float(slope)


def road_geometry(
    image_centerline: FloatArray,
    ground_plane: GroundPlane,
    preview_distances_m: tuple[float, ...] = DEFAULT_PREVIEW_M,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
    offset_distance_m: float = DEFAULT_OFFSET_DISTANCE_M,
) -> RoadGeometry | None:
    """Metric road geometry from an image-space ego centreline.

    Args:
        image_centerline: ``(M, 2)`` centreline in image pixels (from
            :func:`src.geometry.centerline.ego_centerline`).
        ground_plane: Image-to-ground mapping (:class:`src.geometry.ipm.GroundPlane`).
        preview_distances_m: Look-ahead distances at which to report curvature.
        num_samples: Samples along the ground curve for the curvature estimate.
        smoothing: Spline smoothing passed to the curvature estimator.
        offset_distance_m: Distance ahead at which to report lateral offset.

    Returns:
        A :class:`RoadGeometry`, or ``None`` if the centreline has fewer than three
        points after projection or is too degenerate to fit.
    """
    ground = ground_plane.to_ground(np.asarray(image_centerline, dtype=np.float64))
    if ground.shape[0] < 3:
        return None
    # Order near (small z) to far (large z).
    ground = ground[np.argsort(ground[:, 1])]

    fit = fit_near_line(ground)
    if fit is None:
        return None
    intercept, slope = fit
    lateral_offset = intercept + slope * offset_distance_m
    heading = float(np.arctan(slope))

    # Negate the counter-clockwise convention so that right turns are positive.
    kappa = -signed_curvature_along(ground, num_samples=num_samples, smoothing=smoothing)
    tck = fit_arclength_spline(ground, smoothing)
    previews = np.asarray(preview_distances_m, dtype=np.float64)

    if tck is None or kappa.size == 0:
        return RoadGeometry(
            ground_centerline=ground,
            lateral_offset_m=lateral_offset,
            offset_distance_m=float(offset_distance_m),
            heading_error_rad=heading,
            curvature_1pm=0.0,
            preview_distances_m=previews,
            preview_curvature_1pm=np.full(previews.shape, np.nan),
        )

    # Curvature samples share the uniform parameter grid with the position samples,
    # so sample z on the same grid and interpolate curvature against distance ahead.
    u = np.linspace(0.0, 1.0, kappa.size)
    _, z_of_u = splev(u, tck, der=0)
    z_of_u = np.asarray(z_of_u, dtype=np.float64)
    order = np.argsort(z_of_u)
    z_sorted, kappa_sorted = z_of_u[order], kappa[order]

    preview_kappa = np.interp(
        previews, z_sorted, kappa_sorted, left=np.nan, right=np.nan
    )
    representative = float(np.median(kappa))

    return RoadGeometry(
        ground_centerline=ground,
        lateral_offset_m=lateral_offset,
        offset_distance_m=float(offset_distance_m),
        heading_error_rad=heading,
        curvature_1pm=representative,
        preview_distances_m=previews,
        preview_curvature_1pm=preview_kappa,
    )


def road_geometry_from_mask(
    mask: np.ndarray,
    ground_plane: GroundPlane,
    preview_distances_m: tuple[float, ...] = DEFAULT_PREVIEW_M,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
    offset_distance_m: float = DEFAULT_OFFSET_DISTANCE_M,
) -> RoadGeometry | None:
    """Metric road geometry straight from a lane mask.

    Convenience wrapper: extract lanes, take the ego centreline, project, and read
    the control quantities. Returns ``None`` if no ego lane can be formed.

    Args:
        mask: Binary lane mask, shape ``(H, W)``.
        ground_plane: Image-to-ground mapping.
        preview_distances_m: Look-ahead distances at which to report curvature.
        num_samples: Samples for the curvature estimate.
        smoothing: Spline smoothing for the curvature estimate.
    """
    polylines = extract_lane_polylines(mask)
    center = ego_centerline(polylines, mask.shape[1], image_height=mask.shape[0])
    if center is None:
        return None
    return road_geometry(
        center, ground_plane, preview_distances_m, num_samples, smoothing,
        offset_distance_m,
    )
