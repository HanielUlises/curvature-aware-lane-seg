# Curvature-Aware Lane Segmentation for Downstream Control

Lane perception from monocular RGB, optimized against **downstream control error on
high-curvature trajectories** rather than raw segmentation overlap. The end consumer is a
kinematic Model Predictive Controller, so every design choice in this repository is
judged by its effect on the geometry the controller actually consumes: the lateral offset
of the vehicle in its lane and the curvature of the lane ahead.

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
memory budget. That headroom matters for the capacity experiment below.

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

## What does not improve it

We tried the obvious levers for a stronger baseline. None beat 0.616, and the pattern of
failure is the useful result.

Sweeping the decision threshold from 0.30 to 0.80 leaves IoU flat, with a peak at exactly
0.50. The probability maps are already well calibrated, so thresholding gives nothing.
Doubling encoder capacity to ResNet-34, with a larger batch and a Dice-weighted loss,
plateaus at 0.611, just below ResNet-18. Capacity is not the constraint, which matches
the memory headroom above. A Lovász-hinge term, which is a direct IoU surrogate,
destabilizes training from scratch and mildly degrades the model as a low-weight
fine-tune. The one change that helps is horizontal-flip test-time augmentation, which
lifts validation IoU to 0.623. Lanes are left/right symmetric, so the averaged prediction
is slightly cleaner at no training cost.

The model is at the practical ceiling for this supervision. The binding constraint is the
thin-stroke geometry: a one-pixel error on a five-pixel mask is a large relative IoU
penalty, and neither extra capacity nor loss engineering changes that. Moving past about
0.62 requires changing what the network predicts, for example thicker or distance-weighted
supervision, or a bird's-eye-view projection so that far lanes are not compressed into a
few pixels. A bigger backbone is not the answer.

## Control-relevant error, and why IoU was misleading

Overlap is not the quantity the controller consumes. To measure what is, the mask is
reduced to an ego centreline, projected to a flat ground plane, and fitted, and the same
pipeline is run on the ground-truth mask; the two geometries are then compared in
control units. Reported below over the full validation split (1,767 frames, of which
1,280 yielded a reference geometry and a prediction), stratified by the same curvature
bins.

| bin | detection rate | lateral offset MAE | heading MAE | κ MAE at 5 m | at 10 m | at 20 m |
|---|---|---|---|---|---|---|
| near-straight | 80.1% | 1.384 | 5.05° | 0.235 | 0.088 | 0.040 |
| gentle | 74.6% | 0.882 | 4.18° | 0.168 | 0.095 | 0.042 |
| moderate | 70.8% | 0.679 | 4.97° | 0.362 | 0.108 | 0.028 |
| sharp | 67.9% | 0.677 | 8.25° | 0.380 | 0.100 | 0.033 |
| tightest | 59.6% | 0.482 | 7.41° | 0.535 | 0.098 | 0.034 |
| **overall** | **72.4%** | **0.895** | **5.63°** | **0.297** | **0.097** | **0.037** |

Offsets are in metres and curvatures in 1/m under the placeholder ground-plane mapping,
so the absolute scale is provisional; the comparison across bins is the meaningful
reading, and the next section establishes which of these columns survive that caveat.

Two of these columns reverse the conclusion drawn from IoU. **Detection rate falls
monotonically with curvature**, from 80% on near-straight frames to 60% on the tightest
ones: on the curves this project exists for, the model fails to yield a usable ego lane
two times in five. **Heading error and near-field curvature error also degrade**, the
latter from 0.235 to 0.535 across the same range, a factor of 2.3. None of this is
visible in the per-bin IoU table, where the curved bins scored slightly *better* than
the straight ones. The reason is that IoU counts pixels, and the pixels are dominated by
the wide, unambiguous lane markings near the bumper, whereas the controller depends on
the geometry further ahead and on the lane being found at all.

One column moves the other way and remains unexplained: lateral offset error *improves*
with curvature, from 1.38 m to 0.48 m. Ambiguous ego-lane selection on wide, near-parallel
markings could pair the wrong two lanes, and extrapolating the centreline to the vehicle
plane is poorly conditioned. Both are measurement artifacts rather than model behaviour,
so nothing is claimed from that column.

The practical conclusion is that the segmentation ceiling discussed above was the wrong
thing to optimise. Detection reliability on curves, not another point of IoU, is what
limits this model as a control front end.

## Metric calibration, and what the numbers can support

The mapping used above asserts that a hand-chosen image trapezoid is a rectangle on flat
ground. That is a stable mapping but not a camera, so it was worth replacing with a real
pinhole model: the ground plane relates to the image by `H = K R M` for rotation `R` and
`M = [[1,0,0],[0,0,h],[0,1,0]]`, which is exact for flat ground. That model is
implemented and verified against synthetic scenes, where it recovers a known pitch to
within 0.05 degrees, returns parallel 3.7 m lanes as parallel and 3.7 m apart to 1e-6,
and recovers the curvature of a projected arc of radius 80 m to 1e-4.

Fitting its parameters to CurveLanes does not work, and the reasons are worth recording
because they bound what the control metric can claim, and because they point to the
dataset that does work.

The textbook route is the vanishing point, which fixes pitch and yaw directly. It is
unusable here: the lane annotations stop around row 146 of the 288-row preprocessed
frame while the horizon sits near row 82, so the vanishing point is a long extrapolation
beyond the data and its per-frame estimate scatters with an interquartile range of about
68 pixels. (Fixing a real bug along the way: the intrinsics must be carried through the
sky crop, since cropping the top moves the principal point up to row 82 rather than the
frame centre at 144.)

The alternative is to fit pitch by making the ego lanes parallel on the ground, which
depends only on the near-field lanes that are observed. Measured as perpendicular
distance rather than lateral difference at equal depth, since the latter overestimates
width on exactly the curves this dataset over-samples, that objective still fails to
identify a calibration. It has no interior optimum, rising monotonically from the search
bound; 63 percent of frames peg their individual optimum at that bound; per-frame optima
disagree across roughly 8 degrees; and no pitch admits a physically plausible camera,
with the implied height 2.37 m at zero pitch against an expected 1.2 to 1.6 m. Matching a
plausible height would need a field of view near 100 degrees, where lens distortion
invalidates an undistorted pinhole model anyway.

The explanation is that CurveLanes aggregates footage from many vehicles and cameras, so
there is no single calibration to recover. That is a property of the dataset rather than a
missing implementation, and it suggests the remedy: calibrate on a source filmed by one
vehicle.

TuSimple is that source. It comes from a single fleet, and it annotates lanes at fixed
rows from 240 to 710 of the native 1280 by 720 frame, which spans the horizon region
instead of stopping short of it. Every diagnostic that failed on CurveLanes now passes:

| | CurveLanes | TuSimple |
|---|---|---|
| parallelism optimum | at search bound | interior, 7.6 degrees |
| parallelism residual | 0.180 | **0.0139** |
| vanishing point vertical IQR | 68 px | **11 px** |
| recovered lane width | 3.43 m, IQR 2.60 to 4.70 | **3.64 m, IQR 3.56 to 3.80** |
| implied camera height | 2.37 m at zero pitch | **1.62 m** |

The fitted camera is 7.47 degrees of pitch, 1.00 degree of yaw, and 1.62 m of height. The
strongest evidence that this is a real measurement rather than a fitted artifact is that
two independent estimators agree: lane parallelism gives 7.47 degrees and the vanishing
point, which the fit never uses, gives 7.71 degrees, a difference of 0.24 degrees. The
recovered lane width lands within 0.5 percent of the 3.7 m standard with an interquartile
range of 0.24 m, against 2.1 m on CurveLanes. The 1.62 m height is higher than a
passenger car, which is consistent with TuSimple being autonomous-trucking footage.

The calibration is fitted in native pixels and converted into preprocessed-frame pixels,
since cropping and scaling change the intrinsic matrix but do not move the camera. One
caveat is worth stating plainly: this describes TuSimple's camera, so it makes the
TuSimple driving demo metric, and it does **not** retroactively put the CurveLanes
control-error table into physical units. Those remain relative comparisons, and
`scripts.eval_control` warns if a calibration is used across datasets.

What this means for the results above is settled empirically rather than by argument.
Re-running the evaluation under the pinhole mapping instead of the trapezoid leaves the
detection rates **bit-for-bit identical** (83.5, 77.1, 76.8, 68.9, 59.3 percent by bin)
while error magnitudes shift by a factor of three to four. Detection rate involves no
metric geometry at all, so the headline finding, that lane detection degrades
monotonically with curvature, does not depend on calibration. The heading and curvature
orderings are preserved across both mappings. The absolute error magnitudes are not, and
should be read only as relative comparisons.

## Qualitative behaviour

The rendered validation overlays, and the held-out TuSimple driving montage, agree with
the numbers. The best frames, around 0.85 IoU, span
the full curvature range, and so do the worst frames. Success and failure are not sorted
by curvature. Where the model fails it is almost always a night scene or faint markings:
it recovers the lane location but lays down a slightly thick, broken stroke. The failure
axis is illumination, not geometry, which is what the flat per-bin table predicts.

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
5. **Temporal stability.** Track the centerline across frames (a simple state filter on the
   spline coefficients) so the control input is smooth rather than re-estimated
   independently per frame.
6. **Kinematic MPC.** Feed lateral offset, heading error, and previewed κ into a kinematic
   bicycle MPC. Close the loop in simulation first, then port the perception-to-geometry
   path to the C++ deployment target.

Steps 1 to 4 are implemented and unit-tested, and step 2 now has a calibrated projection
whose parameters are measured from TuSimple, cross-validated by two independent estimators.
The remaining work is temporal filtering and then the controller itself. Extending the
metric evaluation onto TuSimple, where the units are physical, would let the control errors
be quoted in real metres rather than as relative comparisons.