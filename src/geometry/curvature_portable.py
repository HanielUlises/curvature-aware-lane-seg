"""Portable natural-cubic-spline curvature — the deployment reference algorithm.

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

The math here is identical to ``deploy/src/curvature.cpp``: a natural parametric
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


def _natural_moments(u: FloatArray, f: FloatArray) -> FloatArray:
    """Second-derivative moments of the natural cubic spline through (u, f)."""
    n = u.shape[0]
    moments = np.zeros(n, dtype=np.float64)
    interior = n - 2
    if interior <= 0:
        return moments

    a = np.zeros((interior, interior), dtype=np.float64)
    d = np.zeros(interior, dtype=np.float64)
    for k in range(interior):
        i = k + 1
        h_prev = u[i] - u[i - 1]
        h_next = u[i + 1] - u[i]
        denom = h_prev + h_next
        a[k, k] = 2.0
        if k > 0:
            a[k, k - 1] = h_prev / denom
        if k < interior - 1:
            a[k, k + 1] = h_next / denom
        d[k] = 6.0 / denom * ((f[i + 1] - f[i]) / h_next - (f[i] - f[i - 1]) / h_prev)

    moments[1 : n - 1] = np.linalg.solve(a, d)
    return moments


def curvature_along_natural(points: FloatArray, num_samples: int = 100) -> FloatArray:
    """Sample ``|kappa|`` along a natural parametric cubic spline through ``points``."""
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    n = pts.shape[0]
    if n < _MIN_POINTS or num_samples < 1:
        return np.zeros(0, dtype=np.float64)

    u = _normalized_arclength(pts)
    if u[-1] <= 0.0:
        return np.zeros(0, dtype=np.float64)

    mx = _natural_moments(u, pts[:, 0])
    my = _natural_moments(u, pts[:, 1])

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


def lane_curvature_natural(
    points: FloatArray, percentile: float = 90.0, num_samples: int = 100
) -> float:
    """Percentile of ``|kappa|`` for a lane, using the portable natural spline."""
    kappa = curvature_along_natural(points, num_samples)
    if kappa.size == 0:
        return 0.0
    return float(np.percentile(kappa, percentile))
