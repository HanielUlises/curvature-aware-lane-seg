# Curvature-Aware Lane Segmentation for Control

Monocular lane perception evaluated on the geometry a controller consumes, rather than on
segmentation overlap. The intended consumer is a kinematic Model Predictive Controller,
which needs the vehicle's lateral offset in its lane and the curvature of the lane ahead,
so those are the quantities measured here.

<p align="center">
  <img src="docs/assets/lane_demo.gif" alt="Lane predictions on unseen highway footage" width="720">
</p>

<p align="center"><em>Baseline U-Net/ResNet-18 (0.616 val IoU) run frame by frame on held-out
TuSimple clips, a different camera than the CurveLanes training data. Predicted lane mask
in red. The prediction stays stable across frames with no temporal modelling.</em></p>

## Data and training

Training uses a curvature-stratified subset of CurveLanes rather than the raw split.
Overlap metrics averaged over the natural distribution are insensitive to the frames the
controller cares most about, which are the tight curves. Per-frame curvature is the 90th
percentile of |κ| along each annotated lane, normalized by image width and reduced to the
maximum over lanes. Frames are binned on this value with fixed, percentile-derived edges
and sampled to an equal count per bin.

Figure 1 shows the effect. CurveLanes is already a curve-curated dataset, so after
excluding odd-aspect frames the image-space curvature concentrates in the middle bins,
and the near-straight and tightest bins each hold about five percent of the data by
construction of the percentile edges. Drawing 1,600 frames per bin (8,000 total) flattens
the distribution and leaves the two extreme bins roughly four times over-represented
relative to their natural frequency.

![Curvature stratification of the training subset](docs/assets/fig_stratification.png)

*Figure 1. Natural vs. stratified training distribution over curvature bins (log scale).
The subset is flat by construction, so the tails are lifted relative to nature.*

The model is a U-Net with an ImageNet-pretrained ResNet-18 encoder (14.3 M parameters).
It trains on 512×288 crops with an equal-weight Dice + BCE objective, Adam at 1e-3 on a
cosine schedule, batch size 8, mixed precision. Preprocessing applies an aspect-preserving
sky crop followed by an isotropic resize, so the resize does not distort curvature and the
stratification label stays valid. Validation runs on a separately stratified 2,000-frame
set. On a 6 GB RTX 3060 an epoch takes about 2.5 minutes and uses 2.4 GB, well under the
memory budget.

Validation IoU (Figure 2) climbs quickly and plateaus around epoch 40. The best
checkpoint is epoch 48 at 0.616. The early kink is a resume after a power interruption,
not a property of the schedule.

![Validation IoU over training](docs/assets/fig_learning_curve.png)

*Figure 2. Foreground IoU on the validation set across 50 epochs.*

## Accuracy across curvature

The best checkpoint reaches 0.616 IoU and 0.762 Dice on the lane class. Background is
excluded as uninformative. Broken out by bin:

| | overall | near-straight | gentle | moderate | sharp | tightest |
|---|---|---|---|---|---|---|
| IoU  | 0.616 | 0.588 | 0.620 | 0.634 | 0.620 | 0.630 |
| Dice | 0.762 | 0.740 | 0.766 | 0.776 | 0.765 | 0.773 |

Accuracy does not fall off with curvature. The near-straight bin is the weakest and the
curved bins are marginally better (Figure 3). This is the stratified sampling working as
intended: with the tight-curve regime lifted to equal weight during training, the model
does not treat it as a rare corner case. The near-straight bin lags for a geometric
reason, not a modelling one. A near-straight lane runs thin all the way to the vanishing
point and carries few foreground pixels, so with a 5-pixel annotation stroke the IoU
denominator is small and boundary error dominates the ratio. A single global IoU would
have hidden this, which is the argument for stratifying the metric.

![Per-bin IoU across curvature](docs/assets/fig_per_bin_iou.png)

*Figure 3. Per-bin IoU on the held-in validation set and on a held-out split. The dashed
line is the overall validation IoU. The tightest bin was not sampled in the
natural-distribution held-out draw (n=0).*

## Generalization

To confirm these numbers are not an artifact of the validation set, we evaluated on 250
frames from the CurveLanes `valid` split that belong to neither the training subset nor
the validation subset. They are unseen by both training and checkpoint selection. Overall
IoU there is 0.640, slightly above the validation figure, so the model is not overfit.
The per-bin pattern holds (Figure 3, orange). The tightest bin is empty in this draw
because the sample follows the natural distribution, where such frames are rare. That gap
is itself the reason stratified training was necessary, even when stratified evaluation
cannot populate every bin.

## Control-relevant error, and why IoU was misleading

Overlap is not the quantity the controller consumes. To measure what is, the mask is
reduced to an ego centreline, projected to a flat ground plane, and fitted, and the same
pipeline is run on the ground-truth mask; the two geometries are then compared in
control units. Reported below over the full validation split (1,767 frames, of which
1,280 yielded a reference geometry and a prediction), stratified by the same curvature
bins.

Offset is reported 5 m ahead rather than at the vehicle, because nothing is observed at
the vehicle plane: the recovered centreline typically begins about 12 m out, so a figure
quoted at the bumper is extrapolated rather than measured. Offset and heading both come
from a least-squares line fitted over the near 12 m.

| bin | detection rate | offset MAE at 5 m | heading MAE | κ MAE at 5 m | at 10 m | at 20 m |
|---|---|---|---|---|---|---|
| near-straight | 82.1% | 1.507 | 4.04° | 0.186 | 0.066 | 0.034 |
| gentle | 75.3% | 1.065 | 3.32° | 0.181 | 0.101 | 0.034 |
| moderate | 71.5% | 0.667 | 3.45° | 0.320 | 0.103 | 0.024 |
| sharp | 68.9% | 0.716 | 4.07° | 0.351 | 0.101 | 0.038 |
| tightest | 60.5% | 0.429 | 3.86° | 0.544 | 0.088 | 0.029 |
| **overall** | **73.5%** | **0.976** | **3.72°** | **0.268** | **0.091** | **0.032** |

Offsets are in metres and curvatures in 1/m under the placeholder ground-plane mapping,
so the absolute scale is provisional; the comparison across bins is the meaningful
reading, and the next section establishes which of these columns survive that caveat.

Two of these columns reverse the conclusion drawn from IoU. **Detection rate falls
monotonically with curvature**, from 82% on near-straight frames to 61% on the tightest
ones: on the curves this project exists for, the model fails to yield a usable ego lane
two times in five. **Near-field curvature error degrades** over the same range, from 0.186
to 0.544, a factor of 2.9. Neither is visible in the per-bin IoU table, where the curved
bins scored slightly *better* than the straight ones. The reason is that IoU counts pixels,
and the pixels are dominated by the wide, unambiguous lane markings near the bumper,
whereas the controller depends on the geometry further ahead and on the lane being found
at all.

Heading error does **not** support that conclusion, and an earlier version of this table
claimed it did. Reading offset and heading from two points instead of a fitted line put
heading MAE at 8.25° in the sharp bin against 5.05° near-straight, which looked like
curvature-dependent degradation. Two points define a line exactly, so their noise passes
through undamped. Fitting the near span drops overall heading error from 5.63° to 3.72° and
flattens it across bins to between 3.32° and 4.07°, leaving no curvature trend. The
apparent trend was an estimator artifact.

One column remains unexplained: offset error *improves* with curvature, from 1.51 m to
0.43 m. It survives both the estimator fix and both ground mappings, so it is not either of
those. Ambiguous ego-lane selection on wide, near-parallel markings pairing the wrong two
boundaries is the remaining candidate. Nothing is claimed from that column.

The practical conclusion is that another point of IoU was the wrong thing to chase.
Detection reliability on curves is what limits this model as a control front end.

## Metric calibration, and what the numbers can support

The table above uses a hand-chosen trapezoid asserted to be a rectangle on flat ground.
That is stable but it is not a camera, so it was replaced with a pinhole model, where the
ground plane maps to the image by `H = K R M` with `M = [[1,0,0],[0,0,h],[0,1,0]]`. On
synthetic scenes the model recovers a known pitch to 0.05 degrees and a projected arc of
radius 80 m to 1e-4.

Its parameters cannot be fitted to CurveLanes. The vanishing point is unusable because the
annotations stop around row 146 while the horizon sits near row 82, leaving a long
extrapolation that scatters by about 68 pixels. Fitting pitch by lane parallelism instead
also fails to identify anything: no interior optimum, 63 percent of frames pegged at the
search bound, and no pitch admitting a plausible camera height. The cause is that
CurveLanes aggregates many vehicles and cameras, so no single calibration exists.

TuSimple does have one, being a single fleet with lanes annotated across the horizon
region. Every failed diagnostic passes there:

| | CurveLanes | TuSimple |
|---|---|---|
| parallelism optimum | at search bound | interior, 7.6 degrees |
| parallelism residual | 0.180 | **0.0139** |
| vanishing point vertical IQR | 68 px | **11 px** |
| recovered lane width | 3.43 m, IQR 2.60 to 4.70 | **3.64 m, IQR 3.56 to 3.80** |
| implied camera height | 2.37 m at zero pitch | **1.62 m** |

The fitted camera is 7.47 degrees of pitch, 1.00 degree of yaw and 1.62 m of height. Two
independent estimators agree on it: parallelism gives 7.47 degrees, and the vanishing
point, which the fit never uses, gives 7.71 degrees. The height suits TuSimple being
trucking footage.

Two caveats. This is TuSimple's camera, so it makes the driving demo metric but leaves the
CurveLanes error magnitudes as relative comparisons; `scripts.eval_control` warns on
cross-dataset use. And re-running the evaluation under the pinhole mapping leaves detection
rates **bit-for-bit identical** while error magnitudes shift three to four fold, so the
headline finding does not depend on calibration but the absolute magnitudes do.

## Qualitative behaviour

<p align="center">
  <img src="docs/assets/validation_sweep.gif" alt="Validation frames ordered from straight to tight curve" width="620">
</p>

<p align="center"><em>Validation frames ordered by curvature, near-straight first and
tightest last. Red is the prediction, green the ground-truth contour, and the caption gives
each frame's curvature, bin and IoU. Unlike the demos above this is CurveLanes with labels,
so prediction and truth can be compared directly.</em></p>

The overlays agree with the numbers. The best frames, around 0.85 IoU, span the full
curvature range, and so do the worst frames. Success and failure are not sorted by
curvature. Where the model fails it is almost always a night scene or faint markings: it
recovers the lane location but lays down a slightly thick, broken stroke. The failure axis
is illumination, not geometry, which is what the flat per-bin table predicts.

Watching the sweep also shows what the per-bin IoU could not. The prediction stays glued to
the ground truth across the whole curvature range, which is exactly why IoU looked flat,
while the frames where the model picks up a barrier or a kerb alongside the real markings
are the ones that later cost an ego-lane detection.

## Control chain

<p align="center">
  <img src="docs/assets/control_demo.gif" alt="Perception to control chain on TuSimple footage" width="820">
</p>

<p align="center"><em>Top left: predicted lane mask (red), lane polylines recovered from it
(cyan), and the ego centreline (yellow). Top right: the same geometry after the calibrated
ground projection, on a metric grid, with the quantities the controller consumes; crosses
mark the 5, 10 and 20 m preview distances. Bottom: raw per-frame estimates in grey against
the temporally filtered signal in green. Five seconds of continuous held-out TuSimple
footage at 20 Hz, 99 of 100 frames yielding an ego lane.</em></p>

This is the whole chain in one view: mask, then polylines, then centreline, then metric
ground geometry, then the three control inputs. It also serves as a visual check on the
calibration, because the lane boundaries come back close to vertical and evenly spaced in
the bird's-eye panel. Perspective makes them converge sharply in the camera view, so
recovering them as parallel is the property a correct inverse-perspective map has to
satisfy, and it is the same property the fit was scored on.

The readout is the interface to the controller. Lateral offset is signed positive to the
right of the camera axis, heading is the centreline bearing from straight ahead, and
curvature is signed so a right turn is positive, quoted alongside the radius it implies. On
this stretch the values sit around 0.2 m of offset, a couple of degrees of heading, and
radii in the hundreds of metres, which is what a highway lane should give.

Where the panel reads "no ego lane" the mask carried distant lane markings but not the two
boundaries either side of the vehicle, so no centreline could be formed. Those frames are
the detection failures counted in the control metric above, and seeing them here is the
point: an overlap score would have logged the distant markings as a partial success, while
the controller gets nothing.

<p align="center">
  <img src="docs/assets/control_montage.gif" alt="The same pipeline across many different scenes" width="700">
</p>

<p align="center"><em>The same pipeline over 240 frames drawn from many separate clips,
227 of them yielding an ego lane. TuSimple clips are one second each and mostly minutes
apart, so the hard cuts are clip boundaries rather than tracking failures. Where the
continuous run above shows how the estimate behaves in time, this shows how it holds up
across scenes.</em></p>

Two things are easier to see here than in a single stretch of road. The lane count changes
constantly as the mask picks up barriers, kerbs and reflective strips alongside the real
markings, which is the association weakness that costs detections. And the readout stays in
a plausible band across scenes it has never seen, which is the case for the calibration
generalizing beyond the frames it was fitted on.

The trace strip is the argument for filtering at all, and it is not flattering to the raw
signal. Frames here are 50 ms apart, so any large change between neighbours is noise rather
than motion; a vehicle cannot move half a metre sideways in that time.

Chasing that noise found two real defects, and the first two explanations were both wrong,
which is worth recording. The visible symptom was the centreline sliding sideways and
snapping back.

The first suspect was lane association, the ego pair switching between boundaries. The data
did not support it: offset changed by the same amount whether or not the selected boundaries
had switched. The actual cause was the **estimator**. Offset and heading were read from the
two nearest centreline points, and since the centreline began a median of 12.6 m ahead, that
noise was extrapolated back over a long lever arm. A least-squares fit over the near 12 m
replaced it, and that is also what corrected the false heading trend above.

The second suspect was lateral movement of the drawn line, and that was wrong too. At a
fixed depth the line is steady to about 0.9 px per frame. What moved was its **near end**,
which jumped a median of 8.9 m in depth per frame, because the centreline was defined only
over the rows the two boundaries happened to share and one short boundary truncated it.
Resampling both boundaries onto a common row grid, each extended by at most 45 rows past
what it covers, cut that to 2.25 m and brought the near end from 17.5 m to 8.1 m.

| | original | after fit | after fit and extent | filtered |
|---|---|---|---|---|
| offset jitter | 0.779 m | 0.427 m | 0.366 m | **0.120 m** |
| heading jitter | 3.48° | 1.63° | 1.27° | **0.46°** |
| near-end depth wander | 8.9 m | 8.9 m | **2.25 m** | n/a |

One thing that was tried and rejected: drawing the centreline reconstructed from the
filtered state rather than the raw one. Measured at 10 m ahead it was 7.7 times *less*
steady than the raw line, because rebuilding a curve from three filtered scalars amplifies
heading and curvature noise quadratically with distance. Reverted.

Curvature is untouched by either fix, since it comes from the spline rather than the
near-field line, and it remains the weakest signal: raw curvature still flips sign on 48 of
98 frame transitions, carrying almost no information about which way the road bends until
filtered. Association is also still imperfect, with the lane count changing on 71 of 99
transitions as the mask fires on barriers and reflective strips. The drawn end still moves
by about 32 px per frame, since bounded extension cannot invent the 200 rows that would be
needed to pin it to the frame edge. And 0.12 m of filtered offset variation per frame is
still more than a controller should track. The geometry is markedly better than it was and
still not good enough to drive on.

## Roadmap toward the controller

The segmenter produces a lane mask in the image plane. The controller needs lateral
offset and preview curvature in metric ground coordinates. The steps between them, in
order:

1. **Centerline extraction.** Reduce the binary mask to ordered lane points and a drivable
   centerline (skeletonize or column-wise peak-picking, then associate points into lanes).
   This turns pixels into the polylines the geometry module already expects.
2. **Ground-plane projection (IPM).** Apply an inverse-perspective homography from camera
   calibration to map lane points into a metric bird's-eye view. This removes the
   perspective inflation of curvature near the vanishing point, which is the known
   limitation of image-space κ set out in the
   [geometry port specification](docs/geometry_port_spec.md).
3. **Geometry in BEV.** Fit the arclength spline in ground coordinates and read off
   curvature κ(s) and lateral offset. The estimator already exists in the geometry module,
   with a portable NumPy reference and a validated C++ port in the deployment module
   against shared golden vectors.
4. **Control-relevant evaluation.** Replace, or at least augment, IoU with the errors the
   MPC consumes: lateral offset error and curvature error at fixed preview distances. This
   makes the top-line thesis measurable, since a model can win on IoU and still misplace
   the centerline the controller tracks. Implemented; see the control-error section above.
5. **Temporal stability.** A constant-velocity Kalman filter per quantity, so the command
   is smooth rather than re-estimated independently each frame. It also answers the
   detection-failure finding directly: a frame with no ego lane advances on the motion
   model instead of returning nothing, and the filter reports how long it has been
   coasting so a supervisor can hand over before the extrapolation is trusted too far.
   Measurements far outside the predicted distribution are gated out, which stops one
   badly placed centreline from stepping the steering.
6. **Kinematic MPC.** Linearized lateral error dynamics of a kinematic bicycle, with
   previewed curvature entering as a known disturbance so a curve is handled by
   feedforward rather than by letting error build. Stacking the horizon makes the solve a
   closed-form least-squares problem, re-solved every step, with the steering limit applied
   by saturation; a genuinely constrained solve would need a QP, which this is not. In
   steady state on a constant-curvature path it reproduces the Ackermann relation
   `delta = L * kappa`, and in closed-loop simulation it drives a 1.5 m offset to under
   2 cm and holds a constant curve to under 5 cm.

All six steps are implemented and unit-tested, with step 2 calibrated from TuSimple and
cross-validated by two independent estimators. Steps 2 and 3 are now ported to C++
alongside the curvature estimator, pinned by their own golden vectors; porting them
exposed a real defect, since the natural spline end condition the port had been using
forces curvature to zero at the ends of a polyline and so returned 0.0032 1/m instead of
0.02 for the 5 m preview on a 50 m arc. Not-a-knot end conditions fixed it in both
languages. What remains is integration rather than new components: running the filter and
controller over recorded sequences to measure closed-loop behaviour against real
perception noise instead of the simulated plant, and porting the mask-to-centreline stage
and the controller itself. Extending the
metric evaluation onto TuSimple, where the units are physical, would let the control errors
be quoted in real metres rather than as relative comparisons.
