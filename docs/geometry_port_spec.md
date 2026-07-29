# Geometry Port Specification (C++ Deployment)

The end consumer of this system is a real-time kinematic MPC controller running
on-vehicle, in C++ rather than Python. This document is the **numerical contract** a
C++ port of the geometry module must satisfy, so that the port is a verified
transcription rather than a reimplementation trusted by eye.

It currently specifies **curvature**, the only geometry primitive implemented so far.
The inverse-perspective projection and the ground-plane spline fit extend the same
contract-plus-golden-vectors pattern.

## 1. Rationale

The training path fits each annotation polyline with a FITPACK smoothing spline (a
Fortran library with bespoke automatic knot placement, reached through SciPy). It has
no drop-in C++ equivalent, and reproducing its internals on a real-time target is
neither feasible nor desirable. The port therefore does **not** target FITPACK. It
targets a fully specified, portable algorithm, and correctness is established
numerically against shared golden vectors.

## 2. Implementations

| Role | Method | Purpose |
|---|---|---|
| Training reference | [FITPACK smoothing spline (SciPy)](../src/geometry/curvature.py) | Curvature-stratified subset and evaluation; produced the shipped manifests. |
| Portable reference | [Natural cubic spline, pure NumPy](../src/geometry/curvature_portable.py) | The specification the C++ mirrors. |
| Deployment port | [Natural cubic spline, Eigen](../deploy/src/curvature.cpp) | The on-vehicle loop. |

The portable reference and the C++ port implement identical mathematics and agree to
floating-point precision, with relative error on the order of $10^{-16}$ on the golden
vectors. The FITPACK reference and the portable reference agree on closed-form geometry
(for a circle of radius $R$ both yield $\kappa = 1/R$ to within $\sim 0.1\%$) but
legitimately differ on noisy polylines, since the two spline formulations respond
differently to annotation jitter. This is acceptable: they operate on different inputs,
the FITPACK reference on image-space annotation polylines for stratification, the port
on detected ground-plane points at inference, so bit-for-bit agreement is not required.

## 3. Algorithm

Let the input be an ordered polyline

```math
P = \{(x_i, y_i)\}_{i=0}^{N-1}
```

in the coordinate space in which curvature is desired.

**3.1 Deduplication.** Remove consecutive coincident points, which produce zero-length
segments and a singular parameterization. If fewer than three distinct points remain,
curvature is undefined and the estimate is $0$.

**3.2 Arclength parameter.** With segment lengths
$\ell_i = \lVert (x_{i+1}, y_{i+1}) - (x_i, y_i) \rVert$, define the cumulative arclength
and its normalization to the unit interval:

```math
s_i = \sum_{k=1}^{i} \ell_{k-1}, \qquad u_i = \frac{s_i}{s_{N-1}} \in [0, 1].
```

**3.3 Natural cubic splines.** Fit $x(u)$ and $y(u)$ independently as natural cubic
splines. For a scalar function $f$ with nodal values $f_i$ and spacings
$h_i = u_{i+1} - u_i$, the second-derivative moments $M_i = f''(u_i)$ solve the
tridiagonal system

```math
h_{i-1} M_{i-1} + 2(h_{i-1} + h_i)\, M_i + h_i M_{i+1}
  = 6\left( \frac{f_{i+1} - f_i}{h_i} - \frac{f_i - f_{i-1}}{h_{i-1}} \right),
\quad 1 \le i \le N-2,
```

under the natural boundary conditions

```math
M_0 = M_{N-1} = 0.
```

On the segment $u \in [u_i, u_{i+1}]$, writing $a = u_{i+1} - u$ and $b = u - u_i$, the
interpolant and its derivatives are

```math
f(u) = \frac{M_i a^3 + M_{i+1} b^3}{6 h_i}
      + \left( \frac{f_i}{h_i} - \frac{M_i h_i}{6} \right) a
      + \left( \frac{f_{i+1}}{h_i} - \frac{M_{i+1} h_i}{6} \right) b,
```

```math
f'(u) = \frac{-M_i a^2 + M_{i+1} b^2}{2 h_i}
       + \frac{f_{i+1} - f_i}{h_i}
       - \frac{(M_{i+1} - M_i) h_i}{6},
\qquad
f''(u) = \frac{M_i a + M_{i+1} b}{h_i}.
```

**3.4 Curvature.** For the parametric curve $(x(u), y(u))$, the
parameterization-invariant curvature is

```math
\kappa(u) = \frac{\bigl\lvert x'(u)\, y''(u) - y'(u)\, x''(u) \bigr\rvert}
                 {\bigl( x'(u)^2 + y'(u)^2 \bigr)^{3/2}},
```

with the speed denominator floored at $\varepsilon = 10^{-12}$ to guard near-stationary
points. The parametric form is required: vertical tangents and lanes that double back,
the high-curvature cases this project targets, have no graph form $y = f(x)$.

**3.5 Lane summary.** Evaluate $\kappa$ at $m$ uniformly spaced parameters
$u \in [0, 1]$ and reduce to the $p$-th percentile ($p = 90$ by default) using the
linear method: with the sorted samples $\kappa_{(0)} \le \dots \le \kappa_{(m-1)}$ and
the fractional rank

```math
r = \frac{p}{100}\,(m - 1), \qquad
\kappa_p = \kappa_{(\lfloor r \rfloor)}
         + \bigl( r - \lfloor r \rfloor \bigr)
           \bigl( \kappa_{(\lceil r \rceil)} - \kappa_{(\lfloor r \rfloor)} \bigr).
```

**3.6 Resolution normalization.** Curvature has units of inverse length, so scaling the
coordinates by $1/W$ (image width) scales curvature by $W$. Normalizing the input by
width yields the resolution-invariant value

```math
\tilde{\kappa} = W \,\kappa_{\text{pixel}}.
```

## 4. Tolerances

A port is validated against the flat golden fixtures. Two regimes:

- **Closed-form cases** ($\text{line}$, $\text{circle}_R$) assert against the analytic
  curvature. A straight line must give $\lvert \kappa \rvert < 10^{-3}$; a circle of
  radius $R$ must give $\kappa = 1/R$ within $5\%$ (the reference run shows
  $\sim 0.1\%$).
- **Reference cases** ($\text{clothoid}$, real lanes) assert against the portable NumPy
  reference within $2\%$ (the reference run shows machine precision, the mathematics
  being identical).

Sanity anchors: for a straight line $\kappa \equiv 0$; for a circle of radius $R$,
$\kappa \equiv 1/R$; for a clothoid, $\kappa(s)$ is affine in arclength $s$.

If a future port substitutes a different spline, either regenerate the expectations from
the portable reference or retain the natural-cubic mathematics and expect the same tight
agreement.

## 5. Golden vectors

The export routine writes a canonical JSON set (consumed by the Python guard) and a flat
text set (parsed by the C++ test without a JSON dependency). Each fixture carries the
input polyline and the expected summary. The Python side is guarded by its golden test,
the C++ side by the deployment test suite. Regenerate the fixtures after any change to
the curvature mathematics.

## 6. Limitations carried into deployment

- **Image-space curvature.** Curvature from an annotation polyline lives in the image
  plane, where perspective inflates apparent curvature near the vanishing point. True
  road curvature requires the ground-plane projection, which will feed the port
  ground-plane points in place of image points.
- **Natural boundary conditions** force $\kappa = 0$ at the polyline endpoints. The
  percentile summary is robust to this, but pointwise $\kappa$ near the ends is not
  trustworthy; trim to the interior if a port needs per-sample values.
