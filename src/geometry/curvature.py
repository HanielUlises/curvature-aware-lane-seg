"""Per-frame curvature estimation from CurveLanes annotation polylines.

Curvature is the quantity the downstream controller ultimately steers on, and it
is what the evaluation stratifies by. This module estimates a single scalar
curvature summary per frame from the raw annotation polylines, in a form that is
**resolution-invariant** so that frames captured at different native resolutions
are comparable.

Method
------
Each lane polyline is treated as a parametric curve ``(x(u), y(u))`` fitted with
a cubic B-spline over a normalized-arclength parameter ``u in [0, 1]``. Curvature
is evaluated with the parameterization-invariant formula

.. math::

    \\kappa = \\frac{|x' y'' - y' x''|}{(x'^2 + y'^2)^{3/2}}

where derivatives are taken with respect to the spline parameter. The formula is
invariant to the choice of regular parameterization, so using ``u`` (rather than
true arclength) yields the correct geometric curvature while keeping the fit
well conditioned.

Because ``y = f(x)`` is never used, vertical tangents and lanes that double back
relative to the image axes are handled without special cases — the reason the
project mandates arclength parameterization.

Resolution invariance
----------------------
Curvature has units of inverse length, so a value in raw pixels depends on image
scale. Coordinates are normalized by the native image **width** before fitting
(an isotropic scaling that preserves aspect), giving a dimensionless
``kappa_norm = kappa_pixel * width`` that is comparable across resolutions. A
circle of radius ``R`` pixels in a ``W``-wide image has ``kappa_norm = W / R``.

Aggregation
-----------
Per lane, the curvature summary is a high percentile (default 90th) of
``|kappa(u)|`` sampled along the lane — this targets the *tightest* part of the
lane while staying robust to endpoint noise. Per frame, the summary is the
**maximum** over lanes: a frame is "curved" if any lane bends sharply. This
choice deliberately biases the stratification toward the high-curvature regime
the project exists to serve.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep

from src.data.curvelanes import FrameAnnotation

FloatArray = npt.NDArray[np.float64]

# Minimum points to define curvature at all (< 3 => at most a straight segment).
_MIN_POINTS_FOR_CURVATURE = 3
# Denominator floor guarding the curvature ratio at near-stationary points.
_SPEED_EPS = 1e-12

DEFAULT_PERCENTILE = 90.0
DEFAULT_NUM_SAMPLES = 100
# Light smoothing for real (width-normalized) annotations; 0.0 interpolates and
# is used by the analytic tests. In normalized coordinates the scene spans ~[0, 1],
# so this is a gentle denoising of second derivatives, not a shape change.
DEFAULT_SMOOTHING = 1e-4


def _dedup_consecutive(points: FloatArray) -> FloatArray:
    """Drop consecutive duplicate points (zero-length segments break the fit)."""
    if points.shape[0] <= 1:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0.0, axis=1)
    return points[keep]


def _normalized_arclength(points: FloatArray) -> FloatArray:
    """Cumulative arclength normalized to ``[0, 1]`` for use as spline parameter."""
    seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0.0:
        return cum
    return cum / total


def fit_arclength_spline(
    points: FloatArray,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple | None:
    """Fit a parametric B-spline over normalized arclength.

    Args:
        points: Polyline of shape ``(N, 2)`` in whatever coordinate space the
            caller wants curvature reported in (raw pixels or normalized).
        smoothing: ``s`` passed to :func:`scipy.interpolate.splprep`. ``0.0``
            interpolates; small positive values denoise second derivatives.

    Returns:
        The spline representation ``tck`` from ``splprep``, or ``None`` if the
        polyline is too short or degenerate to define curvature.
    """
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    n = pts.shape[0]
    if n < _MIN_POINTS_FOR_CURVATURE:
        return None

    degree = min(3, n - 1)
    u = _normalized_arclength(pts)
    if u[-1] <= 0.0:
        return None

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, k=degree, s=smoothing)
    except (ValueError, TypeError):
        return None
    return tck


def curvature_along(
    points: FloatArray,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
) -> FloatArray:
    """Sample geometric curvature ``|kappa(u)|`` uniformly along a polyline.

    Args:
        points: Polyline of shape ``(N, 2)``.
        num_samples: Number of evaluation points along the curve.
        smoothing: Spline smoothing ``s`` (see :func:`fit_arclength_spline`).

    Returns:
        Array of non-negative curvature magnitudes. Empty if curvature is
        undefined (fewer than three unique points).
    """
    tck = fit_arclength_spline(points, smoothing)
    if tck is None:
        return np.zeros(0, dtype=np.float64)

    u = np.linspace(0.0, 1.0, num_samples)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    numerator = np.abs(dx * ddy - dy * ddx)
    speed_sq = dx * dx + dy * dy
    denom = np.power(np.maximum(speed_sq, _SPEED_EPS), 1.5)
    return np.asarray(numerator / denom, dtype=np.float64)


def lane_curvature(
    points: FloatArray,
    percentile: float = DEFAULT_PERCENTILE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
) -> float:
    """Scalar curvature summary for one lane: a high percentile of ``|kappa|``.

    Args:
        points: Polyline of shape ``(N, 2)``.
        percentile: Percentile of ``|kappa(u)|`` to report (default 90).
        num_samples: Samples along the curve.
        smoothing: Spline smoothing ``s``.

    Returns:
        The requested percentile of curvature magnitude, or ``0.0`` if curvature
        is undefined for this polyline.
    """
    kappa = curvature_along(points, num_samples=num_samples, smoothing=smoothing)
    if kappa.size == 0:
        return 0.0
    return float(np.percentile(kappa, percentile))


def frame_curvature(
    frame: FrameAnnotation,
    percentile: float = DEFAULT_PERCENTILE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
) -> float:
    """Resolution-invariant scalar curvature for a frame.

    Coordinates are normalized by native width before fitting, so the result is
    dimensionless and comparable across resolutions. The frame summary is the
    maximum per-lane summary (see module docstring).

    Args:
        frame: Parsed frame annotation.
        percentile: Per-lane curvature percentile.
        num_samples: Samples along each lane.
        smoothing: Spline smoothing ``s``.

    Returns:
        ``max_lane percentile(|kappa_norm|)``; ``0.0`` for a frame with no lane
        that defines curvature.
    """
    if frame.width <= 0:
        raise ValueError(f"{frame.image_path.name}: non-positive width {frame.width}")

    best = 0.0
    for lane in frame.lanes:
        normalized = lane.points / float(frame.width)
        best = max(best, lane_curvature(normalized, percentile, num_samples, smoothing))
    return best
