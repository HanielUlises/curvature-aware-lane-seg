# Curvature-Aware Lane Segmentation for Downstream Control

Lane perception from monocular RGB, optimized against **downstream control error on
high-curvature trajectories** rather than segmentation overlap. The end consumer of this
system is a kinematic Model Predictive Controller, not a benchmark leaderboard. Every
design decision in this repository is argued against its effect on the geometry the
controller actually consumes — the lateral offset and the curvature of the lane ahead —
and not against how many pixels happen to agree with a mask.

