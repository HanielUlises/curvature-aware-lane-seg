# Geometry Port Specification (C++ Deployment)

The end consumer of this system is a real-time kinematic MPC controller running
on-vehicle, in C++ rather than Python. This document is the **numerical contract** a
C++ port of the geometry module must satisfy, so that the port is a verified
transcription rather than a reimplementation trusted by eye.

It specifies three stages of the perception-to-control chain: **curvature**
(sections 3 to 6), the **inverse-perspective projection** (section 7), and the metric
**road-geometry read-out** the controller consumes (section 8). Each is held to the
same contract-plus-golden-vectors pattern. The lane-mask decomposition that precedes
them and the controller itself are not yet ported.

## 1. Rationale

The training path fits each annotation polyline with a FITPACK smoothing spline (a
Fortran library with bespoke automatic knot placement, reached through SciPy). It has
no drop-in C++ equivalent, and reproducing its internals on a real-time target is
neither feasible nor desirable. The port therefore does **not** target FITPACK. It
targets a fully specified, portable algorithm, and correctness is established
numerically against shared golden vectors.

## 2. Implementations

| Stage | Training reference | Portable reference | Deployment port |
|---|---|---|---|
| Curvature | [FITPACK smoothing spline](../src/geometry/curvature.py) | [Cubic spline, pure NumPy](../src/geometry/curvature_portable.py) | [Cubic spline, Eigen](../deploy/src/curvature.cpp) |
| Projection | [DLT homography](../src/geometry/ipm.py) | same module (no library gap) | [DLT homography, Eigen](../deploy/src/ipm.cpp) |
| Road geometry | [FITPACK-backed read-out](../src/geometry/road_geometry.py) | [Portable read-out](../src/geometry/road_geometry_portable.py) | [Eigen read-out](../deploy/src/road_geometry.cpp) |

The curvature stage is the only one with a library gap. The projection is a small SVD
with the same mathematics on both sides, so the port is held to floating-point
agreement rather than a loose tolerance. The road-geometry stage inherits the curvature
split: its portable reference exists so the port has something to be pinned against
that does not call FITPACK.

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

**3.3 Cubic splines.** Fit $x(u)$ and $y(u)$ independently as interpolating cubic
splines. For a scalar function $f$ with nodal values $f_i$ and spacings
$h_i = u_{i+1} - u_i$, the second-derivative moments $M_i = f''(u_i)$ solve the
tridiagonal system

```math
h_{i-1} M_{i-1} + 2(h_{i-1} + h_i)\, M_i + h_i M_{i+1}
  = 6\left( \frac{f_{i+1} - f_i}{h_i} - \frac{f_i - f_{i-1}}{h_{i-1}} \right),
\quad 1 \le i \le N-2,
```

closed by **not-a-knot** end conditions, which require the third derivative to be
continuous across the first and last interior knots:

```math
h_1 M_0 - (h_0 + h_1) M_1 + h_0 M_2 = 0,
\qquad
h_{N-2} M_{N-3} - (h_{N-3} + h_{N-2}) M_{N-2} + h_{N-3} M_{N-1} = 0.
```

Together with the $N-2$ interior equations this is a full $N \times N$ system for all
moments, including the endpoints. With $N = 3$ there is a single interior knot and the
not-a-knot conditions are undefined; that case falls back to $M_0 = M_{N-1} = 0$.

The natural condition $M_0 = M_{N-1} = 0$ was used previously and is **not**
acceptable here. It forces $\kappa = 0$ at the polyline ends, and the controller reads
curvature at a $5\,\text{m}$ look-ahead which, on a centreline recovered from a
detection, sits close to the near end. Measured on a circular arc of radius
$50\,\text{m}$, the natural condition returned $\kappa = 0.0032\ \text{m}^{-1}$ at that
look-ahead against a true $0.02\ \text{m}^{-1}$; not-a-knot returns
$0.02000\ \text{m}^{-1}$ at every look-ahead. The percentile summary of section 3.5 was
insensitive to the difference, which is why the error surfaced only once the
road-geometry stage began reading pointwise values.

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
the portable reference or retain the mathematics of section 3 and expect the same tight
agreement.

## 5. Golden vectors

Two export routines, one for curvature and one for the projection and road-geometry
stages, each write a canonical JSON set (consumed by the Python guard) and a flat text
set (parsed by the C++ tests without a JSON dependency). Each fixture carries its input
and the expected output. The Python side is guarded by its golden tests, the C++ side by
the two deployment test suites. Regenerate the fixtures after any change to the
mathematics, and expect the values to move: the not-a-knot correction of section 3.3
changed every fixture that reads pointwise curvature.

## 6. Limitations carried into deployment

- **Image-space curvature.** Curvature from an annotation polyline lives in the image
  plane, where perspective inflates apparent curvature near the vanishing point. True
  road curvature requires the ground-plane projection, which will feed the port
  ground-plane points in place of image points.
- **Endpoint conditioning.** Not-a-knot removes the forced $\kappa = 0$ that the
  natural condition imposed at the polyline ends (section 3.3), but the ends remain the
  least constrained part of the fit: they are extrapolated from one interior cubic
  rather than bracketed by data on both sides. Pointwise $\kappa$ within a segment or
  two of either end is accordingly the least trustworthy.
- **Interpolation, not smoothing.** The portable spline passes exactly through every
  input point, so detection jitter enters the curvature estimate undamped. The training
  path applies FITPACK smoothing; the deployment path relies instead on the temporal
  filter downstream. A port fed raw per-frame detections without that filter will see
  curvature noise.

## 7. Projection (inverse-perspective mapping)

Given $N \ge 4$ correspondences $(x_i, y_i) \mapsto (u_i, v_i)$, the homography
$H$ satisfying $[u, v, 1]^\top \sim H\,[x, y, 1]^\top$ is the Direct Linear Transform
solution. Each correspondence contributes two rows to $A \in \mathbb{R}^{2N \times 9}$:

```math
\begin{bmatrix}
-x_i & -y_i & -1 & 0 & 0 & 0 & u_i x_i & u_i y_i & u_i \\
0 & 0 & 0 & -x_i & -y_i & -1 & v_i x_i & v_i y_i & v_i
\end{bmatrix},
```

and $\mathrm{vec}(H)$ is the right singular vector of $A$ belonging to the smallest
singular value, rescaled so $H_{33} = 1$. That rescaling also removes the sign
ambiguity of the singular vector, so the two implementations must agree elementwise and
not merely up to scale.

Applying $H$ to a point divides out the homogeneous coordinate, with the divisor
floored at $10^{-12}$ in magnitude so that points at the horizon, where the divisor
passes through zero, do not produce infinities.

**Tolerance.** Elementwise agreement with the reference $H$, and agreement on mapped
points, within $10^{-9}$ relative. The round trip $H^{-1} H$ must return the probe
points to within $10^{-6}$ pixels; this is what draws bird's-eye results back onto the
camera frame, where an inversion error would show up only as a mis-drawn overlay.

## 8. Road geometry

Input: the ego centreline in ground coordinates, $x$ lateral (right positive), $z$
ahead, sorted by increasing $z$. Output: the three quantities the kinematic lateral MPC
consumes.

**8.1 Near-field line.** Take the points within $12\,\text{m}$ of the nearest one (or
the three nearest, if fewer than three qualify) and fit $x = m z + c$ by ordinary least
squares. Lateral offset and heading error are read from this fit:

```math
e_y = c + m\, d, \qquad e_\psi = \arctan m,
```

where $d = 5\,\text{m}$ is the look-ahead at which offset is reported. Offset is **not**
quoted at $z = 0$: a centreline recovered from a detection typically begins about
$12\,\text{m}$ ahead, so a value at the vehicle plane would be extrapolated rather than
measured. Two nearest points would define the line exactly and pass their angular noise
straight through that extrapolation, which is why the span is fitted instead.

**8.2 Curvature.** Sample signed curvature along the spline of section 3, retaining the
sign of the cross product rather than its magnitude, and negate it: the formula returns
counter-clockwise-positive values, and in these axes a right-hand turn is clockwise,
whereas the controller convention is right-positive. The scalar summary is the median
over the samples.

**8.3 Preview curvature.** Curvature samples are indexed by the spline parameter, not
by distance, so evaluate the spline itself on the identical uniform parameter grid,
sort the pairs $(z_j, \kappa_j)$ by depth, and linearly interpolate $\kappa$ at each
requested look-ahead. Look-aheads outside the sampled depth range yield $\mathrm{NaN}$
rather than a clamped value: the controller must be able to tell "no curvature
information at $20\,\text{m}$" from "straight at $20\,\text{m}$".

**Tolerance.** $10^{-6}$ relative against the portable reference on all four outputs,
with $\mathrm{NaN}$ required to match $\mathrm{NaN}$ exactly. The closed-form anchors
are a straight lane ($e_\psi = 0$, $\kappa = 0$), a lane at constant slope
($e_\psi = \arctan m$ exactly), and a circular arc of radius $R$ ($\kappa = \pm 1/R$ at
every preview distance, sign following the turn direction).
