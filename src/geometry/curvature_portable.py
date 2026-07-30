"""Portable cubic-spline curvature — the deployment reference algorithm.

This is the numerical **specification** the C++ deployment port (``deploy/``)
mirrors, implemented in pure NumPy with no scipy/FITPACK dependency. It exists
because the training-time curvature in :mod:`src.geometry.curvature` uses
``scipy.splprep`` (FITPACK), which has no drop-in C++ equivalent and whose
smoothing-spline internals are not portable.

Both implementations recover the true geometric curvature on curves with a
closed-form answer (line, circle), so they agree there. On arbitrary noisy
polylines they legitimately differ (different spline formulations), which is why
the C++ port is validated against *this* reference for real-frame cases rather
than against FITPACK. The two curvature notions are computed on different inputs
anyway — FITPACK on the annotation polyline for stratification, this on the
detected BEV points at deployment — so they need not agree bit-for-bit.

The math here is identical to ``deploy/src/curvature.cpp``: a parametric
cubic spline over normalized arclength, curvature via the parameterization-
invariant formula, summarized by a percentile of ``|kappa|``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_SPEED_EPS = 1e-12
_MIN_POINTS = 3


def _dedup_consecutive(points: FloatArray) -> FloatArray:
    if points.shape[0] <= 1:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0.0, axis=1)
    return points[keep]


def _normalized_arclength(points: FloatArray) -> FloatArray:
    seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    return cum / total if total > 0.0 else cum


def _spline_moments(u: FloatArray, f: FloatArray) -> FloatArray:
    """Second-derivative moments of the cubic spline through ``(u, f)``.

    End conditions are **not-a-knot**: the third derivative is continuous across the
    first and last interior knots, which is equivalent to the first two (and last
    two) segments sharing one cubic. The earlier natural condition ``M_0 = M_{n-1} = 0``
    forced curvature to zero at the polyline ends, and the controller reads curvature
    at a 5 m look-ahead, which on a recovered centreline sits close to the near end.
    On a circular arc of radius 50 m the natural condition returned 0.0032 1/m there
    against a true 0.02; not-a-knot recovers the correct value.

    With only three points there is a single interior knot and not-a-knot is
    undefined, so the natural condition is used for that case.
    """
    n = u.shape[0]
    if n < _MIN_POINTS:
        return np.zeros(n, dtype=np.float64)

    h = np.diff(u)
    a = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1]
        a[i, i] = 2.0 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i]
        d[i] = 6.0 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])

    if n >= 4:
        a[0, 0], a[0, 1], a[0, 2] = h[1], -(h[0] + h[1]), h[0]
        a[n - 1, n - 3] = h[n - 2]
        a[n - 1, n - 2] = -(h[n - 3] + h[n - 2])
        a[n - 1, n - 1] = h[n - 3]
    else:
        a[0, 0] = 1.0
        a[n - 1, n - 1] = 1.0

    return np.linalg.solve(a, d)


def curvature_along_portable(points: FloatArray, num_samples: int = 100) -> FloatArray:
    """Sample ``|kappa|`` along the portable cubic spline through ``points``."""
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    n = pts.shape[0]
    if n < _MIN_POINTS or num_samples < 1:
        return np.zeros(0, dtype=np.float64)

    u = _normalized_arclength(pts)
    if u[-1] <= 0.0:
        return np.zeros(0, dtype=np.float64)

    mx = _spline_moments(u, pts[:, 0])
    my = _spline_moments(u, pts[:, 1])

    kappa = np.empty(num_samples, dtype=np.float64)
    seg = 0
    for s in range(num_samples):
        uu = 0.0 if num_samples == 1 else s / (num_samples - 1)
        while seg < n - 2 and uu > u[seg + 1]:
            seg += 1
        dx1, dx2 = _segment_derivs(uu, u, pts[:, 0], mx, seg)
        dy1, dy2 = _segment_derivs(uu, u, pts[:, 1], my, seg)
        numer = abs(dx1 * dy2 - dy1 * dx2)
        speed_sq = max(dx1 * dx1 + dy1 * dy1, _SPEED_EPS)
        kappa[s] = numer / speed_sq**1.5
    return kappa


def _segment_derivs(
    uu: float, u: FloatArray, f: FloatArray, m: FloatArray, seg: int
) -> tuple[float, float]:
    h = u[seg + 1] - u[seg]
    a = (u[seg + 1] - uu) / h
    b = (uu - u[seg]) / h
    d1 = (f[seg + 1] - f[seg]) / h + h / 6.0 * (
        (3.0 * b * b - 1.0) * m[seg + 1] - (3.0 * a * a - 1.0) * m[seg]
    )
    d2 = a * m[seg] + b * m[seg + 1]
    return float(d1), float(d2)


def _segment_value(
    uu: float, u: FloatArray, f: FloatArray, m: FloatArray, seg: int
) -> float:
    """Value of the moment-form cubic on segment ``seg`` at parameter ``uu``.

    Unlike :func:`_segment_derivs`, which is written in terms of the normalized
    local coordinates, the value form takes ``a`` and ``b`` unnormalized; dividing
    them by ``h`` here silently rescales the curve.
    """
    h = u[seg + 1] - u[seg]
    a = u[seg + 1] - uu
    b = uu - u[seg]
    value = (m[seg] * a**3 + m[seg + 1] * b**3) / (6.0 * h)
    value += (f[seg] / h - m[seg] * h / 6.0) * a
    value += (f[seg + 1] / h - m[seg + 1] * h / 6.0) * b
    return float(value)


def _prepared(points: FloatArray) -> tuple[FloatArray, FloatArray] | None:
    """Deduplicated points with their normalized-arclength parameter, or ``None``."""
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < _MIN_POINTS:
        return None
    u = _normalized_arclength(pts)
    if u[-1] <= 0.0:
        return None
    return pts, u


def _uniform_parameters(num_samples: int) -> FloatArray:
    if num_samples == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(0.0, 1.0, num_samples)


def signed_curvature_along_portable(
    points: FloatArray, num_samples: int = 100
) -> FloatArray:
    """Sample **signed** ``kappa`` along the portable cubic spline.

    Identical to :func:`curvature_along_portable` but keeps the sign of the cross
    product (positive is a counter-clockwise turn in the frame the points are given
    in). The controller needs the sign; the stratification does not, which is why the
    unsigned form remains the one pinned by the curvature golden vectors.
    """
    prep = _prepared(points)
    if prep is None or num_samples < 1:
        return np.zeros(0, dtype=np.float64)
    pts, u = prep
    n = pts.shape[0]
    mx = _spline_moments(u, pts[:, 0])
    my = _spline_moments(u, pts[:, 1])

    kappa = np.empty(num_samples, dtype=np.float64)
    seg = 0
    for s, uu in enumerate(_uniform_parameters(num_samples)):
        while seg < n - 2 and uu > u[seg + 1]:
            seg += 1
        dx1, dx2 = _segment_derivs(uu, u, pts[:, 0], mx, seg)
        dy1, dy2 = _segment_derivs(uu, u, pts[:, 1], my, seg)
        speed_sq = max(dx1 * dx1 + dy1 * dy1, _SPEED_EPS)
        kappa[s] = (dx1 * dy2 - dy1 * dx2) / speed_sq**1.5
    return kappa


def sample_positions_portable(points: FloatArray, num_samples: int = 100) -> FloatArray:
    """Evaluate the spline itself on the same uniform parameter grid.

    The curvature samples are indexed by the spline parameter, not by distance, so a
    consumer that wants curvature at a metric look-ahead needs the positions on the
    identical grid to interpolate against. Returns ``(num_samples, 2)``, or an empty
    array when the spline is undefined.
    """
    prep = _prepared(points)
    if prep is None or num_samples < 1:
        return np.zeros((0, 2), dtype=np.float64)
    pts, u = prep
    n = pts.shape[0]
    mx = _spline_moments(u, pts[:, 0])
    my = _spline_moments(u, pts[:, 1])

    out = np.empty((num_samples, 2), dtype=np.float64)
    seg = 0
    for s, uu in enumerate(_uniform_parameters(num_samples)):
        while seg < n - 2 and uu > u[seg + 1]:
            seg += 1
        out[s, 0] = _segment_value(uu, u, pts[:, 0], mx, seg)
        out[s, 1] = _segment_value(uu, u, pts[:, 1], my, seg)
    return out


def lane_curvature_portable(
    points: FloatArray, percentile: float = 90.0, num_samples: int = 100
) -> float:
    """Percentile of ``|kappa|`` for a lane, using the portable cubic spline."""
    kappa = curvature_along_portable(points, num_samples)
    if kappa.size == 0:
        return 0.0
    return float(np.percentile(kappa, percentile))
