# Geometry port specification (C++ deployment)

The end consumer of this system is a real-time kinematic MPC controller running
on-vehicle. That loop is C++, not Python. This document is the **numerical
contract** a C++ port of `src/geometry/` must satisfy so the port is a verified
transcription rather than a reimplementation people have to trust by eye.

It currently covers **curvature**, the only geometry primitive implemented so
far. IPM/homography and spline fitting (roadmap steps 3–6) will extend this same
contract-plus-golden-vectors pattern.

## Why a spec instead of a line-by-line port

The Python training path (`src/geometry/curvature.py`) fits the annotation
polyline with `scipy.interpolate.splprep`, i.e. **FITPACK** — a Fortran
smoothing-spline library with bespoke automatic knot placement. There is no
drop-in C++ equivalent, and replicating FITPACK's internals in C++ is neither
feasible nor desirable for a real-time target. So the port does **not** target
FITPACK. It targets a portable, fully specified algorithm, and correctness is
established **numerically** against shared golden vectors.

## The two curvature implementations

| | Implementation | Used for |
|---|---|---|
| Training reference | `src/geometry/curvature.py` (FITPACK smoothing spline) | Curvature-stratified subset + evaluation. Produced the shipped manifests. |
| Portable reference | `src/geometry/curvature_portable.py` (natural cubic spline, pure NumPy) | The **spec** the C++ mirrors. |
| Deployment port | `deploy/src/curvature.cpp` (natural cubic spline, Eigen) | The on-vehicle loop. |

The portable reference and the C++ port implement **identical math** and agree to
floating-point precision (`rel_err ~1e-16` on the golden vectors). The FITPACK
reference and the portable reference **agree on closed-form geometry** (a circle
of radius `R` gives `kappa = 1/R` from both, within ~0.1%) but **legitimately
differ on noisy polylines** — different spline formulations respond differently
to annotation jitter. This is expected and acceptable: the two are computed on
different inputs anyway (FITPACK on image-space annotation polylines for
stratification; the port on detected BEV points at inference), so they are not
required to agree bit-for-bit.

## Algorithm (natural parametric cubic spline)

Input: an ordered polyline `(x_i, y_i)`, `i = 0..N-1`, in the coordinate space
curvature is wanted in (normalize by image width first for a resolution-invariant
result — see `frame_curvature`).

1. **Dedup** consecutive identical points (zero-length segments are singular).
2. If fewer than 3 unique points remain, curvature is undefined → return 0.
3. **Parameter** `u_i` = cumulative Euclidean arclength, normalized to `[0, 1]`.
4. Fit **natural cubic splines** `x(u)`, `y(u)` independently: second-derivative
   moments `M` solve the standard tridiagonal system with natural boundary
   conditions `M_0 = M_{N-1} = 0`.
5. At each of `num_samples` uniform `u` in `[0, 1]`, evaluate first/second
   derivatives of the moment-form cubic and the **parameterization-invariant**
   curvature

   ```
   kappa = |x'·y'' − y'·x''| / (x'² + y'²)^{3/2}
   ```

   with the speed denominator floored at `1e-12`.
6. **Lane summary** = the `percentile` (default 90th) of `|kappa|` over the
   samples, using NumPy's `linear` percentile method (rank
   `p/100·(m−1)`, linear interpolation between order statistics).

Using `(x(u), y(u))` rather than `y = f(x)` is mandatory: vertical tangents and
lanes that double back — the high-curvature cases this project exists for — have
no functional form.

## Tolerances a port must meet

Validated by `deploy/test/test_curvature.cpp` against `deploy/test/golden/*.txt`:

- **Closed-form cases** (`line`, `circle_R*`): assert against the *analytic*
  curvature. Circles within **5%** (the golden run shows ~0.1%); line `|kappa| <
  1e-3`.
- **Reference cases** (`clothoid`, `real_*`): assert against the portable NumPy
  reference within **2%** (the golden run shows machine precision, since the math
  is identical).

If a future port swaps the spline (e.g. tinyspline, Eigen's `SplineFitting`),
regenerate expectations from the portable reference, or keep the natural-cubic
math and expect the same tight agreement.

## Regenerating the golden vectors

```
python -m scripts.export_golden_vectors
```

Writes `tests/golden/curvature/*.json` (canonical) and
`deploy/test/golden/*.txt` (flat, parsed by the C++ test without a JSON
dependency). `tests/test_golden_curvature.py` guards the Python side; the C++
side is guarded by `ctest` in `deploy/`.

## Known limitations carried into deployment

- **Image-space curvature.** Curvature from the annotation polyline is in the
  image plane; perspective inflates apparent curvature near the vanishing point.
  True road curvature requires BEV projection (IPM), which is a later roadmap
  step and will feed the port BEV points instead of image points.
- **Natural boundary conditions** zero the curvature at the polyline endpoints.
  The percentile summary is robust to this, but pointwise `kappa` near the ends
  is not trustworthy — trim the interior if a port needs per-sample values.
