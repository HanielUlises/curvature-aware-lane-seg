"""Camera calibration for the ground-plane projection.

Roadmap step five. The placeholder mapping in :mod:`src.geometry.ipm` asserts that a
hand-chosen image trapezoid is a rectangle on flat ground. That is enough to exercise
the pipeline but it is not a camera model, so the metric scale is arbitrary and any
error in the trapezoid shows up as a systematic geometry error. This module replaces it
with an actual pinhole model whose pitch and yaw are **estimated from the data**.

## Frames and model

The road frame has ``X`` lateral (right positive), ``Z`` forward, ``Y`` up, with the
origin on the ground directly below the camera centre. The camera sits at height ``h``
and uses the usual image convention: ``x`` right, ``y`` down, ``z`` along the optical
axis. A ground point ``(X, Z)`` therefore has unrotated camera coordinates
``(X, h, Z)``, and since

```
(X, h, Z) = M (X, Z, 1),      M = [[1, 0, 0], [0, 0, h], [0, 1, 0]]
```

the ground plane maps to the image by a homography

```
H_ground->image = K R M,      R = Rx(pitch) Ry(yaw)
```

with ``pitch`` positive downwards and ``yaw`` positive when the camera turns left. The
projection is exact for flat ground; the inverse is the mapping the controller needs.

## Estimating the extrinsics

Taking ``Z -> inf`` gives the road vanishing point, which depends only on the rotation:

```
u_vp = cx + fx tan(yaw) / cos(pitch),      v_vp = cy - fy tan(pitch)
```

so a vanishing point measured from the detected lanes inverts directly to pitch and
yaw. That leaves focal length and height, which no monocular view can determine on its
own: focal length comes from an assumed field of view, and height sets the overall
metric scale and can be recovered from a known lane width (see
:func:`refine_height_from_lane_width`), because ground coordinates scale linearly in
``h``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.geometry.ipm import GroundPlane

FloatArray = np.ndarray

# Typical forward-facing dashcam horizontal field of view, degrees.
DEFAULT_HFOV_DEG = 70.0
# Typical camera mounting height for a passenger car, metres.
DEFAULT_HEIGHT_M = 1.35
# Standard highway lane width, metres (US Interstate / TuSimple footage).
DEFAULT_LANE_WIDTH_M = 3.7


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics plus the extrinsics that matter for a flat road.

    Attributes:
        fx: Focal length in pixels, horizontal.
        fy: Focal length in pixels, vertical.
        cx: Principal point column, pixels.
        cy: Principal point row, pixels.
        height_m: Camera height above the road, metres.
        pitch_rad: Downward tilt of the optical axis, radians (positive looks down).
        yaw_rad: Rotation about the vertical axis, radians (positive turns left).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = DEFAULT_HEIGHT_M
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def intrinsic_matrix(self) -> FloatArray:
        """The ``3x3`` intrinsic matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def rotation_matrix(self) -> FloatArray:
        """The camera rotation ``R = Rx(pitch) Ry(yaw)``."""
        cp, sp = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
        cy_, sy = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        ry = np.array([[cy_, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy_]])
        return rx @ ry

    def vanishing_point(self) -> tuple[float, float]:
        """Image location of the road vanishing point implied by the extrinsics."""
        u = self.cx + self.fx * math.tan(self.yaw_rad) / math.cos(self.pitch_rad)
        v = self.cy - self.fy * math.tan(self.pitch_rad)
        return float(u), float(v)


def intrinsics_from_fov(
    image_size: tuple[int, int], hfov_deg: float = DEFAULT_HFOV_DEG
) -> tuple[float, float, float, float]:
    """Approximate intrinsics from an assumed horizontal field of view.

    Assumes square pixels (``fy == fx``) and a centred principal point, which is a
    reasonable default for consumer cameras and is all a single view supports.

    Args:
        image_size: ``(width, height)`` in pixels.
        hfov_deg: Horizontal field of view, degrees.

    Returns:
        ``(fx, fy, cx, cy)`` in pixels.

    Raises:
        ValueError: If ``hfov_deg`` is not in ``(0, 180)``.
    """
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
    width, height = image_size
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return float(fx), float(fx), width / 2.0, height / 2.0


def intrinsics_for_preprocessed_frame(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    sky_frac: float,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> tuple[float, float, float, float]:
    """Intrinsics valid in the coordinates of a sky-cropped, resized frame.

    The model runs on frames produced by
    :func:`src.data.transforms.preprocess_geometry`, which removes the top
    ``sky_frac`` of the image, trims the longer axis to the target aspect, then scales
    isotropically. Cropping the top moves the principal point **up** in the cropped
    frame and the resize scales the focal length, so intrinsics computed as if the
    principal point sat at the centre of the preprocessed frame are wrong by the height
    of the crop. This applies the same crop-and-scale to the intrinsics.

    For a source that already matches the target aspect the result is independent of
    source resolution, which is what lets one calibration serve a mixed-resolution
    dataset.

    Args:
        source_size: Native ``(width, height)`` before preprocessing.
        target_size: Preprocessed ``(width, height)``.
        sky_frac: Fraction of height removed from the top.
        hfov_deg: Assumed horizontal field of view of the native camera, degrees.

    Returns:
        ``(fx, fy, cx, cy)`` in preprocessed-frame pixels.

    Raises:
        ValueError: If ``sky_frac`` is outside ``[0, 1)``.
    """
    if not 0.0 <= sky_frac < 1.0:
        raise ValueError(f"sky_frac must be in [0, 1), got {sky_frac}")
    width, height = source_size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    # Mirror preprocess_geometry's crop decisions exactly.
    top = min(int(round(height * sky_frac)), height - 1)
    crop_h = height - top
    x0, y0, crop_w = 0, top, width
    if width / crop_h > target_ratio:
        crop_w = int(round(crop_h * target_ratio))
        x0 = (width - crop_w) // 2
    elif width / crop_h < target_ratio:
        new_h = int(round(width / target_ratio))
        y0 = top + (crop_h - new_h)

    scale = target_w / crop_w
    fx_native, _, cx_native, cy_native = intrinsics_from_fov(source_size, hfov_deg)
    return (
        float(fx_native * scale),
        float(fx_native * scale),
        float((cx_native - x0) * scale),
        float((cy_native - y0) * scale),
    )


def calibration_for_preprocessed_frame(
    calib: CameraCalibration,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    sky_frac: float,
) -> CameraCalibration:
    """Move a native-resolution calibration into preprocessed-frame coordinates.

    Cropping and scaling an image changes where the principal point sits and how long
    the focal length is in pixels, but it does not move or rotate the camera, so the
    extrinsics carry over unchanged. Use this to apply a calibration measured on native
    frames (where lane annotations live) to the preprocessed frames the model runs on.

    Args:
        calib: Calibration valid in native ``source_size`` pixels.
        source_size: Native ``(width, height)``.
        target_size: Preprocessed ``(width, height)``.
        sky_frac: Fraction of height removed from the top by preprocessing.

    Returns:
        The same camera expressed in preprocessed-frame pixels.
    """
    width, height = source_size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    # Mirror preprocess_geometry's crop decisions exactly.
    top = min(int(round(height * sky_frac)), height - 1)
    crop_h = height - top
    x0, y0, crop_w = 0, top, width
    if width / crop_h > target_ratio:
        crop_w = int(round(crop_h * target_ratio))
        x0 = (width - crop_w) // 2
    elif width / crop_h < target_ratio:
        new_h = int(round(width / target_ratio))
        y0 = top + (crop_h - new_h)

    scale = target_w / crop_w
    return CameraCalibration(
        fx=calib.fx * scale,
        fy=calib.fy * scale,
        cx=(calib.cx - x0) * scale,
        cy=(calib.cy - y0) * scale,
        height_m=calib.height_m,
        pitch_rad=calib.pitch_rad,
        yaw_rad=calib.yaw_rad,
    )


def ground_plane_from_calibration(calib: CameraCalibration) -> GroundPlane:
    """Build the exact flat-ground mapping for a calibrated camera.

    Args:
        calib: Camera intrinsics and extrinsics.

    Returns:
        A :class:`src.geometry.ipm.GroundPlane` whose ``to_ground`` maps image pixels
        to ``(x, z)`` metres on the road.

    Raises:
        ValueError: If the camera height is not positive, which makes the ground-plane
            map singular.
    """
    if calib.height_m <= 0.0:
        raise ValueError(f"height_m must be positive, got {calib.height_m}")
    m = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, calib.height_m], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    ground_to_image = calib.intrinsic_matrix() @ calib.rotation_matrix() @ m
    return GroundPlane(h=np.linalg.inv(ground_to_image), h_inv=ground_to_image)


def project_ground_to_image(
    calib: CameraCalibration, ground_points: FloatArray
) -> FloatArray:
    """Forward-project ground ``(x, z)`` metres to image pixels.

    Provided so the projection can be tested against its own inverse rather than
    against hand-measured correspondences.

    Args:
        calib: Camera intrinsics and extrinsics.
        ground_points: ``(N, 2)`` ground points ``(x, z)`` in metres.

    Returns:
        ``(N, 2)`` image points ``(u, v)`` in pixels.
    """
    return ground_plane_from_calibration(calib).to_image(ground_points)


def calibration_from_vanishing_point(
    vanishing_point: tuple[float, float],
    image_size: tuple[int, int],
    hfov_deg: float = DEFAULT_HFOV_DEG,
    height_m: float = DEFAULT_HEIGHT_M,
    intrinsics: tuple[float, float, float, float] | None = None,
) -> CameraCalibration:
    """Recover pitch and yaw from a measured road vanishing point.

    Inverts the vanishing-point relations of the pinhole model. Focal length follows
    from ``hfov_deg`` and the scale from ``height_m``; neither is observable from a
    single view.

    Args:
        vanishing_point: Measured ``(u, v)`` of the road vanishing point, pixels.
        image_size: ``(width, height)`` in pixels.
        hfov_deg: Assumed horizontal field of view, degrees.
        height_m: Assumed camera height, metres.
        intrinsics: Explicit ``(fx, fy, cx, cy)``. Required when the vanishing point
            was measured on a cropped frame, where the principal point is not at the
            frame centre; see :func:`intrinsics_for_preprocessed_frame`.

    Returns:
        The corresponding :class:`CameraCalibration`.
    """
    fx, fy, cx, cy = (
        intrinsics if intrinsics is not None
        else intrinsics_from_fov(image_size, hfov_deg)
    )
    u_vp, v_vp = vanishing_point
    pitch = math.atan2(cy - v_vp, fy)
    yaw = math.atan2((u_vp - cx) * math.cos(pitch), fx)
    return CameraCalibration(
        fx=fx, fy=fy, cx=cx, cy=cy, height_m=height_m,
        pitch_rad=pitch, yaw_rad=yaw,
    )


def estimate_vanishing_point(polylines: list[FloatArray]) -> tuple[float, float] | None:
    """Least-squares intersection of the lane directions.

    Each lane polyline is fitted with a straight line (lanes are locally straight
    enough near the horizon, and the fit is over the whole lane so curvature averages
    out), and the point minimizing the squared distance to all such lines is returned.
    Two lanes are the minimum for a well-posed intersection.

    Args:
        polylines: Lane polylines in image pixels, as produced by
            :func:`src.geometry.centerline.extract_lane_polylines`.

    Returns:
        The estimated ``(u, v)``, or ``None`` if fewer than two usable lanes are given
        or the lines are too near-parallel to intersect stably.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float64)
        if pts.shape[0] < 2:
            continue
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        # Principal direction of the polyline; the normal defines the line.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        rows.append([normal[0], normal[1]])
        rhs.append(float(normal @ centroid))

    if len(rows) < 2:
        return None
    a = np.asarray(rows, dtype=np.float64)
    # Near-parallel lanes make the intersection ill-conditioned; reject rather than
    # return a vanishing point that is numerically meaningless.
    if np.linalg.matrix_rank(a, tol=1e-6) < 2:
        return None
    solution, *_ = np.linalg.lstsq(a, np.asarray(rhs, dtype=np.float64), rcond=None)
    return float(solution[0]), float(solution[1])


def _point_to_polyline_distance(point: FloatArray, polyline: FloatArray) -> float:
    """Shortest distance from a point to a polyline, measured to its segments."""
    starts = polyline[:-1]
    ends = polyline[1:]
    seg = ends - starts
    seg_len_sq = np.sum(seg * seg, axis=1)
    seg_len_sq = np.where(seg_len_sq < 1e-18, 1e-18, seg_len_sq)
    t = np.sum((point - starts) * seg, axis=1) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = starts + t[:, None] * seg
    return float(np.min(np.linalg.norm(point - closest, axis=1)))


def lane_width_profile(
    calib: CameraCalibration,
    left_lane: FloatArray,
    right_lane: FloatArray,
    num_samples: int = 12,
    trim_fraction: float = 0.15,
) -> FloatArray | None:
    """Perpendicular ego-lane width sampled along the left boundary.

    Width is the shortest distance from each sampled left-boundary point to the right
    boundary, **not** the difference in lateral coordinate at equal depth. The two agree
    on a straight road but the equal-depth version overestimates width on a curve, by a
    factor that grows with curvature, which would bias any fit made against a
    curve-heavy sample.

    The ends are trimmed because the nearest point to an endpoint is usually the other
    curve's endpoint rather than a perpendicular foot.

    Args:
        calib: Trial calibration.
        left_lane: Left boundary polyline in image pixels.
        right_lane: Right boundary polyline in image pixels.
        num_samples: Points sampled along the left boundary.
        trim_fraction: Fraction of the boundary dropped at each end.

    Returns:
        Widths in metres, or ``None`` if the trial calibration puts any lane point at
        or behind the camera plane, which makes the projection meaningless.
    """
    plane = ground_plane_from_calibration(calib)
    left = plane.to_ground(np.asarray(left_lane, dtype=np.float64))
    right = plane.to_ground(np.asarray(right_lane, dtype=np.float64))
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    if left[:, 1].min() <= 0.0 or right[:, 1].min() <= 0.0:
        return None
    if left.shape[0] < 2 or right.shape[0] < 2:
        return None

    lo = int(round(len(left) * trim_fraction))
    hi = len(left) - lo
    if hi - lo < 2:
        lo, hi = 0, len(left)
    candidates = left[lo:hi]
    if len(candidates) > num_samples:
        pick = np.linspace(0, len(candidates) - 1, num_samples).round().astype(int)
        candidates = candidates[pick]
    return np.array(
        [_point_to_polyline_distance(p, right) for p in candidates], dtype=np.float64
    )


def lane_parallelism_cost(
    calib: CameraCalibration, lane_pairs: list[tuple[FloatArray, FloatArray]]
) -> float:
    """How far a calibration is from making the ego lanes parallel.

    For each frame the ego-lane width is measured along the depth range the two lanes
    share; a correct ground projection makes that width constant. The cost is the
    median over frames of the relative spread ``std(width) / mean(width)``, which is
    scale free and so does not compete with the camera height.

    Args:
        calib: Trial calibration.
        lane_pairs: ``(left, right)`` lane polylines in image pixels, one per frame.

    Returns:
        The median relative width spread, or ``inf`` if too few frames project validly.
    """
    spreads: list[float] = []
    for left, right in lane_pairs:
        widths = lane_width_profile(calib, left, right)
        if widths is None:
            continue
        mean = float(np.mean(widths))
        if mean <= 1e-9:
            continue
        spreads.append(float(np.std(widths) / mean))
    if len(spreads) < max(5, len(lane_pairs) // 10):
        return float("inf")
    return float(np.median(spreads))


def calibrate_pitch_from_lane_parallelism(
    lane_pairs: list[tuple[FloatArray, FloatArray]],
    intrinsics: tuple[float, float, float, float],
    height_m: float = DEFAULT_HEIGHT_M,
    yaw_rad: float = 0.0,
    pitch_bounds_deg: tuple[float, float] = (-2.0, 25.0),
    coarse_steps: int = 108,
) -> CameraCalibration | None:
    """Estimate pitch by making the ego lanes parallel on the ground.

    Preferred over the vanishing-point route whenever the lane annotations stop well
    short of the horizon, since that leaves the vanishing point a long extrapolation
    beyond the data. Parallelism is determined by the near-field lanes that are
    actually observed.

    Pitch is the only extrinsic this can recover: yaw rotates the ground frame without
    affecting parallelism, and height scales it uniformly.

    Args:
        lane_pairs: ``(left, right)`` ego-lane polylines in image pixels, one per frame.
        intrinsics: ``(fx, fy, cx, cy)`` in the same pixel coordinates as the lanes.
        height_m: Camera height; only sets the scale, not the optimum.
        yaw_rad: Yaw to hold fixed during the search.
        pitch_bounds_deg: Inclusive search range for pitch, degrees.
        coarse_steps: Grid points in the initial scan before local refinement.

    Returns:
        The best-fitting calibration, or ``None`` if no trial pitch projects validly.
    """
    fx, fy, cx, cy = intrinsics

    def trial(pitch_deg: float) -> CameraCalibration:
        return CameraCalibration(
            fx=fx, fy=fy, cx=cx, cy=cy, height_m=height_m,
            pitch_rad=math.radians(pitch_deg), yaw_rad=yaw_rad,
        )

    lo, hi = pitch_bounds_deg
    grid = np.linspace(lo, hi, coarse_steps)
    costs = [lane_parallelism_cost(trial(p), lane_pairs) for p in grid]
    best_idx = int(np.argmin(costs))
    if not np.isfinite(costs[best_idx]):
        return None

    # Golden-section refinement inside the bracketing grid cell.
    step = (hi - lo) / (coarse_steps - 1)
    a, b = grid[best_idx] - step, grid[best_idx] + step
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(40):
        if lane_parallelism_cost(trial(c), lane_pairs) < lane_parallelism_cost(
            trial(d), lane_pairs
        ):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
        if abs(b - a) < 1e-4:
            break
    return trial((a + b) / 2.0)


def _median_lane_bearing(
    calib: CameraCalibration,
    lane_pairs: list[tuple[FloatArray, FloatArray]],
    num_samples: int = 12,
) -> float | None:
    """Median bearing of the ego-lane centreline from straight ahead, radians."""
    bearings: list[float] = []
    plane = ground_plane_from_calibration(calib)
    for left, right in lane_pairs:
        gl = plane.to_ground(np.asarray(left, dtype=np.float64))
        gr = plane.to_ground(np.asarray(right, dtype=np.float64))
        if not (np.isfinite(gl).all() and np.isfinite(gr).all()):
            continue
        if gl[:, 1].min() <= 0.0 or gr[:, 1].min() <= 0.0:
            continue
        z_lo = max(gl[:, 1].min(), gr[:, 1].min())
        z_hi = min(gl[:, 1].max(), gr[:, 1].max())
        if z_hi <= z_lo:
            continue
        zs = np.linspace(z_lo, z_hi, num_samples)
        gl_s, gr_s = gl[np.argsort(gl[:, 1])], gr[np.argsort(gr[:, 1])]
        centre_x = 0.5 * (
            np.interp(zs, gl_s[:, 1], gl_s[:, 0]) + np.interp(zs, gr_s[:, 1], gr_s[:, 0])
        )
        # Least-squares slope dx/dz of the centreline over the observed depth range.
        slope = float(np.polyfit(zs, centre_x, 1)[0])
        bearings.append(math.atan(slope))
    if not bearings:
        return None
    return float(np.median(bearings))


def calibrate_yaw_from_lane_bearing(
    calib: CameraCalibration,
    lane_pairs: list[tuple[FloatArray, FloatArray]],
    yaw_bounds_deg: tuple[float, float] = (-12.0, 12.0),
    steps: int = 97,
) -> CameraCalibration:
    """Estimate yaw by assuming the vehicle is on average aligned with its lane.

    Parallelism cannot determine yaw, since rotating the ground frame leaves parallel
    lanes parallel. The weakest additional assumption that pins it down is that over
    many frames the vehicle points along its lane rather than consistently across it,
    which holds for highway driving. Yaw is chosen so the median ego-lane bearing is
    zero.

    Args:
        calib: Calibration with pitch and height already set.
        lane_pairs: ``(left, right)`` ego-lane polylines in image pixels.
        yaw_bounds_deg: Inclusive search range for yaw, degrees.
        steps: Grid points in the scan.

    Returns:
        A calibration with the estimated yaw, or the input unchanged if no frame
        projects validly.
    """
    def trial(yaw_deg: float) -> CameraCalibration:
        return CameraCalibration(
            fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
            height_m=calib.height_m, pitch_rad=calib.pitch_rad,
            yaw_rad=math.radians(yaw_deg),
        )

    best_yaw, best_abs = None, float("inf")
    for yaw_deg in np.linspace(*yaw_bounds_deg, steps):
        bearing = _median_lane_bearing(trial(float(yaw_deg)), lane_pairs)
        if bearing is None:
            continue
        if abs(bearing) < best_abs:
            best_abs, best_yaw = abs(bearing), float(yaw_deg)
    if best_yaw is None:
        return calib
    return trial(best_yaw)


def refine_height_from_lane_width(
    calib: CameraCalibration,
    left_lane: FloatArray,
    right_lane: FloatArray,
    true_lane_width_m: float = DEFAULT_LANE_WIDTH_M,
) -> CameraCalibration:
    """Rescale the camera height so a known lane width is reproduced.

    Ground coordinates scale linearly in camera height, so the height that makes the
    observed lane separation equal ``true_lane_width_m`` is
    ``h * true_width / measured_width``. This fixes the one free scale parameter
    without any external measurement beyond the lane standard.

    Args:
        calib: Calibration whose height is to be refined.
        left_lane: Left lane polyline in image pixels.
        right_lane: Right lane polyline in image pixels.
        true_lane_width_m: Known lane width, metres.

    Returns:
        A calibration with the corrected height, or the input unchanged if the lane
        separation cannot be measured.
    """
    widths = lane_width_profile(calib, left_lane, right_lane)
    if widths is None or widths.size == 0:
        return calib
    measured = float(np.median(widths))
    if measured <= 1e-9:
        return calib

    scale = true_lane_width_m / measured
    return CameraCalibration(
        fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
        height_m=calib.height_m * scale,
        pitch_rad=calib.pitch_rad, yaw_rad=calib.yaw_rad,
    )
