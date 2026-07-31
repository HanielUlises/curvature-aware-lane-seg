# Geometry Port Specification (C++ Deployment)

The end consumer of this system is a real-time kinematic MPC controller running
on-vehicle, in C++ rather than Python. This document is the **numerical contract** a
C++ port of the geometry module must satisfy, so that the port is a verified
transcription rather than a reimplementation trusted by eye.

It specifies the whole chain, from the segmenter's mask to a steering command:
**curvature** (sections 3 to 6), the **inverse-perspective projection** (section 7), the
metric **road-geometry read-out** (section 8), the **mask decomposition** that starts the
chain (section 9), the **boundary tracker** that gives it temporal identity (section 10),
the **temporal filter** (section 11), and the **controller** itself (section 12). Each is
held to the same contract-plus-golden-vectors pattern.

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
| Mask decomposition | [OpenCV components](../src/geometry/centerline.py) | same module (no library gap) | [Run-based labelling](../deploy/src/centerline.cpp) |
| Boundary tracking | [Tracker](../src/geometry/lane_tracker.py) | same module | [Tracker](../deploy/src/lane_tracker.cpp) |
| Temporal filter | [Kalman filters](../src/geometry/temporal.py) | same module | [Kalman filters](../deploy/src/temporal.cpp) |
| Controller | [Lateral MPC](../src/control/mpc.py) | same module | [Lateral MPC](../deploy/src/mpc.cpp) |

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

## 9. Mask decomposition

The segmenter's binary mask becomes one polyline per lane. Each 8-connected component is
collapsed to its row-wise centroid: for every image row the component occupies, the mean
column of its foreground pixels is one point. Components spanning fewer than 8 rows or
carrying fewer than 40 pixels are discarded as specks. The polylines are ordered
left to right by the column of their bottom-most point.

**9.1 Labelling.** A port need not reproduce OpenCV's label numbering, only its partition
of the foreground; the ordering above is what makes the output canonical. The reference
implementation in this port labels **horizontal runs** rather than pixels. Two runs on
adjacent rows are 8-connected when

```math
a_{x_0} \le b_{x_1} \ \wedge\  b_{x_0} \le a_{x_1}
```

for half-open spans $[x_0, x_1)$, which is an overlap test after widening one span by a
pixel. Since both rows are sorted by column this is a two-pointer merge, and the row-wise
centroid a lane polyline needs has a closed form over a run,

```math
\sum_{x \in [x_0, x_1)} x = \tfrac{1}{2}\,(x_0 + x_1 - 1)(x_1 - x_0),
```

so no pass over individual pixels is required at all. Measured on 512×288 frames this is
the difference between 335 and 38 microseconds, and the mask stage otherwise dominates
the cost of the entire chain.

**9.2 Ego centreline.** The two polylines bracketing the camera axis, nearest on each
side by bottom-most column, are resampled onto a common grid of image rows running from
the further of their two far ends down to the nearer end plus a bounded extension, capped
at the frame edge. Each boundary may be extended at most 45 rows past what it covers,
along a line fitted to that end's last 25 rows. The centreline is the midpoint where both
are defined.

## 10. Boundary tracking

The tracker holds one column profile per boundary over a fixed row grid and carries it
between frames. Its full state, defaults and rationale are in
[the reference implementation](../src/geometry/lane_tracker.py); a port must reproduce
five behaviours, each of which cost a real defect to establish:

1. **Association before averaging.** Each boundary is matched to the one the track was
   already following, by median column distance over shared rows, subject to a gate.
   Re-choosing the pair per frame by proximity to the camera axis lets the next marking
   out take the place of a lost one, moving the centreline by half a lane.
2. **Ordering.** The matched left must lie left of the matched right. A gate alone lets
   the right track match the left boundary when the two are close.
3. **Width plausibility, row by row.** Expected lane width in pixels varies steeply with
   row, from about 105 px at row 60 to 377 px at row 180 on this camera, so a pair is
   compared against the width *profile* being tracked, never against a scalar.
4. **Guards on the drawn line.** Reject where the boundaries have crossed; break where
   the midpoint steps sideways faster than three times its own typical bend; keep only
   the longest contiguous run.
5. **Extent as a median.** The near end is the median of the last three frames, clipped
   to what the current frame supports, and applied **only when it leaves at least three
   points**. Applying it unconditionally erases the line on frames where the median falls
   before the run begins.

**Tolerances.** Validated over a recorded sequence of real predicted masks rather than
synthetic input, since the stage is stateful and its failure modes only appear over time.
Across 1200 consecutive frames the port and the reference agree on every frame that
yields a centreline, on the number of points in it, and on each point to within
$1.3 \times 10^{-11}$ px.

Two numerical details are load-bearing at that tolerance. Row grids must be built the way
`numpy.linspace` builds them, step first with the endpoint pinned, because the last bits
reach the extent cutoff and decide whether a row falls inside it. And the sideways-step
guard is computed over every row including the undefined ones, so that the row following a
gap has an infinite step and is dropped, rather than being skipped and given a step of
zero.

## 11. Temporal filter

Each of the three control quantities is tracked by a scalar Kalman filter over
$[\text{value}, \text{rate}]$ with a constant-rate model. Over a step $\Delta t$,

```math
A = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix},
\qquad
Q = \sigma_a^2 \begin{bmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{bmatrix},
```

the standard discretization of continuous white-noise acceleration. Three behaviours are
part of the contract, and all three are stateful, so a port is validated over a **sequence**
rather than a step: a single-frame fixture agrees with anything.

1. **Seeding.** The first measurement sets the state directly rather than being blended up
   from zero, with $P = \mathrm{diag}(\sigma_m^2, \sigma_a^2)$.
2. **Gating.** A measurement whose residual exceeds $g\sqrt{P_{00} + \sigma_m^2}$ is
   rejected, with $g = 4$ by default. The state still advances on the motion model, so a
   rejected frame is not a frozen frame.
3. **Coasting.** A frame with no geometry predicts forward and increments a counter. The
   counter resets only when the **offset** is accepted, that being the quantity the
   controller is most sensitive to; heading or curvature being gated does not reset it.

**Tolerance.** $10^{-9}$ relative against the reference on all three values, with exact
agreement on the measured, accepted and coasting flags, over a 40-step sequence containing
three dropouts, one gross outlier and a lane change.

## 12. Controller

Linearized lateral error dynamics of a kinematic bicycle at speed $v$ with wheelbase $L$,
forward-Euler discretized:

```math
x_{k+1} = A x_k + B u_k + d_k, \quad
A = \begin{bmatrix} 1 & v\,\Delta t \\ 0 & 1 \end{bmatrix}, \;
B = \begin{bmatrix} 0 \\ v\,\Delta t / L \end{bmatrix}, \;
d_k = \begin{bmatrix} 0 \\ -v\,\Delta t\,\kappa_k \end{bmatrix}.
```

Stacking the horizon gives $X = S_x x_0 + S_u U + S_d D$, and the quadratic cost makes the
input sequence the solution of an unconstrained least-squares problem,

```math
U^\star = -\left( S_u^\top \bar{Q} S_u + \bar{R} \right)^{-1} S_u^\top \bar{Q}
          \left( S_x x_0 + S_d D \right),
```

re-solved every step on the current state. The first element is the command; the steering
limit is applied by saturation, so it is respected at the plant but not inside the
optimization.

**Sign conventions.** Internally cross-track is positive left of the path and steering
positive to the left; the perception stack reports right-positive. The mapping is
$e = \text{offset}$, $\psi = \text{heading}$, $\kappa \to -\kappa$, with the returned
steer negated back. A port that omits the negation produces a controller that steers
confidently into the ditch, which no tolerance would catch, so the fixtures include both
left and right curves.

**Tolerance.** $10^{-9}$ relative on the command and on the unsaturated command, with
exact agreement on the saturation flag, across nine cases spanning offsets, headings,
curvatures of both signs, speeds from 5 to 30 m/s, and a deliberately saturating state.

**Closed-form anchor.** On a constant-curvature path with no tracking error the steady-state
steer is the Ackermann value $\delta = L\kappa$. This holds independently of the fixtures
and is asserted directly at four curvatures, so a port that regenerated the vectors from
its own output is still caught.
