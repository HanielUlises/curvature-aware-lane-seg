"""Estimate a dataset-level camera calibration from annotated lanes.

Roadmap step five. A camera has one calibration, so it is estimated once from the
dataset rather than per frame: a per-frame calibration would differ between the
prediction and the ground truth and would confound the control-error comparison.

The estimate uses the ground-truth lane masks, which are geometry we trust, and rests
on properties the observed near-field lanes actually determine:

1. Intrinsics come from an assumed field of view, carried through the sky crop and
   resize so the principal point is not mistakenly placed at the centre of the
   preprocessed frame.
2. **Pitch** is chosen to make the ego lanes parallel on the ground. This replaces the
   textbook vanishing-point route, which fails here because the lane annotations stop
   well short of the horizon and leave the vanishing point a long extrapolation beyond
   the data (its estimate scatters over tens of pixels).
3. **Yaw** is chosen so the median ego-lane bearing is zero, that is, the vehicle points
   along its lane on average.
4. **Height** is scaled so the median observed lane width equals the standard.

The vanishing point is still reported, as a diagnostic to compare against the fitted
pitch.

    python -m scripts.calibrate_camera
    python -m scripts.calibrate_camera calib_frames=800 ipm.calibrated.hfov_deg=60

Writes a calibration JSON that ``scripts.eval_control`` consumes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.data.tusimple import iter_label_frames
from src.geometry.calibration import (
    CameraCalibration,
    calibration_for_preprocessed_frame,
    intrinsics_from_fov,
    calibrate_pitch_from_lane_parallelism,
    calibrate_yaw_from_lane_bearing,
    estimate_vanishing_point,
    intrinsics_for_preprocessed_frame,
    lane_parallelism_cost,
    lane_width_profile,
    refine_height_from_lane_width,
)
from src.geometry.centerline import ego_lane_pair, extract_lane_polylines

# A representative 16:9 source; the preprocessed intrinsics do not depend on the source
# resolution once the aspect ratio matches, which is what lets one calibration serve the
# mixed-resolution dataset.
REFERENCE_SOURCE_SIZE = (2560, 1440)


def _lane_widths(calib: CameraCalibration, lane_pairs, limit: int = 300) -> np.ndarray:
    """Observed ego-lane widths under a calibration, for reporting.

    Uses the same perpendicular measure as the fit, so the reported width and the
    parallelism residual describe the same quantity.
    """
    widths = []
    for left, right in lane_pairs[:limit]:
        profile = lane_width_profile(calib, left, right)
        if profile is not None and profile.size:
            widths.append(float(np.median(profile)))
    return np.asarray(widths, dtype=np.float64)


def _collect_tusimple(cfg: DictConfig, limit: int):
    """Lane polylines and ego pairs from TuSimple labels, in native pixels."""
    settings = cfg.ipm.calibrated
    source_size = tuple(int(v) for v in settings.tusimple_source_size)
    frames = [f.lanes for f in iter_label_frames(Path(settings.tusimple_label_dir))]
    if len(frames) > limit:
        idx = np.linspace(0, len(frames) - 1, limit).round().astype(int)
        frames = [frames[i] for i in dict.fromkeys(idx.tolist())]
    pairs = [p for p in (ego_lane_pair(L, source_size[0]) for L in frames) if p is not None]
    return frames, pairs, source_size


def _collect_curvelanes(cfg: DictConfig, limit: int):
    """Lane polylines and ego pairs from cached CurveLanes masks, preprocessed pixels."""
    target_size = tuple(cfg.data.target_size)
    entries, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    masks_dir = Path(meta["mask_dir"])
    if limit < len(entries):
        # Entries are grouped by curvature bin, so stride rather than slice.
        idx = np.linspace(0, len(entries) - 1, limit).round().astype(int)
        entries = [entries[i] for i in dict.fromkeys(idx.tolist())]
    frames, pairs = [], []
    for entry in entries:
        gt = cv2.imread(str(masks_dir / f"{entry.image_path.stem}.png"),
                        cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        polylines = extract_lane_polylines((gt > 0).astype(np.uint8))
        if len(polylines) < 2:
            continue
        frames.append(polylines)
        pair = ego_lane_pair(polylines, target_size[0])
        if pair is not None:
            pairs.append(pair)
    return frames, pairs, target_size


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    target_size = tuple(cfg.data.target_size)
    settings = cfg.ipm.calibrated
    source = str(settings.get("source", "curvelanes"))

    limit = int(cfg.get("calib_frames", 500))
    if source == "tusimple":
        frames, lane_pairs, measure_size = _collect_tusimple(cfg, limit)
        # Native frames: the principal point is centred, no crop to account for.
        intrinsics = intrinsics_from_fov(measure_size, float(settings.hfov_deg))
        fit_pitch = True
    elif source == "curvelanes":
        frames, lane_pairs, measure_size = _collect_curvelanes(cfg, limit)
        intrinsics = intrinsics_for_preprocessed_frame(
            REFERENCE_SOURCE_SIZE, target_size, float(cfg.data.sky_frac),
            float(settings.hfov_deg),
        )
        fit_pitch = bool(settings.get("fit_pitch", False))
    else:
        raise ValueError(f"unknown calibration source {source!r}")

    print(f"source: {source}, measured in {measure_size[0]}x{measure_size[1]} px")
    if len(lane_pairs) < 20:
        raise RuntimeError(
            f"only {len(lane_pairs)} frames with an ego lane pair; cannot calibrate"
        )

    vanishing_points = [
        vp for vp in (estimate_vanishing_point(L) for L in frames)
        if vp is not None and np.isfinite(vp).all()
    ]

    fx, fy, cx, cy = intrinsics
    if fit_pitch:
        bounds = tuple(float(v) for v in settings.get("fit_pitch_bounds_deg", (-2.0, 25.0)))
        fitted = calibrate_pitch_from_lane_parallelism(
            lane_pairs, intrinsics, height_m=float(settings.height_m),
            pitch_bounds_deg=bounds,
        )
        if fitted is None:
            raise RuntimeError("no trial pitch projected validly; check the lane data")
        calib = fitted
        pitch_source = "fitted by lane parallelism"
    else:
        calib = CameraCalibration(
            fx=fx, fy=fy, cx=cx, cy=cy, height_m=float(settings.height_m),
            pitch_rad=math.radians(float(settings.pitch_deg)), yaw_rad=0.0,
        )
        pitch_source = "assumed (not identifiable from this data)"
    if bool(settings.get("fit_yaw", True)):
        calib = calibrate_yaw_from_lane_bearing(calib, lane_pairs)

    # Identifiability audit: a well-posed fit has an interior optimum and per-frame
    # optima that agree. Printed either way so the assumption above stays auditable.
    audit_lo, audit_hi = (float(v) for v in settings.get(
        "fit_pitch_bounds_deg", (-1.9, 20.0)))
    grid = np.linspace(audit_lo, audit_hi, 24)
    curve = [
        lane_parallelism_cost(
            CameraCalibration(fx=fx, fy=fy, cx=cx, cy=cy,
                              height_m=calib.height_m,
                              pitch_rad=math.radians(float(p)),
                              yaw_rad=calib.yaw_rad),
            lane_pairs,
        )
        for p in grid
    ]
    finite = np.isfinite(curve)
    argmin = int(np.argmin(np.where(finite, curve, np.inf)))
    interior = 0 < argmin < len(grid) - 1

    if bool(settings.refine_height):
        heights = [
            refine_height_from_lane_width(
                calib, left, right, float(settings.lane_width_m)
            ).height_m
            for left, right in lane_pairs
        ]
        calib = CameraCalibration(
            fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
            height_m=float(np.median(heights)),
            pitch_rad=calib.pitch_rad, yaw_rad=calib.yaw_rad,
        )

    fitted_vp = calib.vanishing_point()
    print(f"frames with ego lane pair : {len(lane_pairs)} of {len(frames)}")
    crop_note = ("native frame" if source == "tusimple"
                 else "sky crop accounted for")
    print(f"principal point           : ({calib.cx:.1f}, {calib.cy:.1f}) px ({crop_note})")
    print(f"focal length              : {calib.fx:.1f} px "
          f"(assumed {float(settings.hfov_deg):.0f} deg HFOV)")
    print(f"pitch                     : {math.degrees(calib.pitch_rad):+.2f} deg "
          f"(down positive) [{pitch_source}]")
    print(f"yaw    (lane bearing)     : {math.degrees(calib.yaw_rad):+.2f} deg "
          f"(left positive)")
    print(f"height (lane width)       : {calib.height_m:.3f} m")
    print(f"parallelism residual      : {lane_parallelism_cost(calib, lane_pairs):.4f} "
          f"(relative width spread; 0 is perfect)")
    print(f"implied horizon row       : {fitted_vp[1]:.1f} px")
    if vanishing_points:
        vps = np.asarray(vanishing_points)
        print(f"measured vanishing point  : ({np.median(vps[:, 0]):.1f}, "
              f"{np.median(vps[:, 1]):.1f}) px, IQR "
              f"({np.subtract(*np.percentile(vps[:, 0], [75, 25])):.1f}, "
              f"{np.subtract(*np.percentile(vps[:, 1], [75, 25])):.1f}) "
              f"[{'independent cross-check' if source == 'tusimple' else 'unreliable here'}]")

    print(f"identifiability           : parallelism cost minimal at "
          f"{grid[argmin]:+.1f} deg, "
          f"{'interior optimum' if interior else 'AT SEARCH BOUND (not identifiable)'}; "
          f"cost {np.min(np.where(finite, curve, np.inf)):.4f} to "
          f"{np.max(np.where(finite, curve, -np.inf)):.4f} over "
          f"{grid[0]:+.0f}..{grid[-1]:+.0f} deg")

    widths = _lane_widths(calib, lane_pairs)
    if widths.size:
        print(f"recovered lane width      : median {np.median(widths):.2f} m, IQR "
              f"{np.percentile(widths, 25):.2f}-{np.percentile(widths, 75):.2f} m "
              f"(target {float(settings.lane_width_m):.2f})")

    # Independent cross-check: pitch and yaw implied by the measured vanishing point.
    vp_pitch_deg = vp_yaw_deg = None
    if vanishing_points:
        vps_arr = np.asarray(vanishing_points)
        u_vp, v_vp = float(np.median(vps_arr[:, 0])), float(np.median(vps_arr[:, 1]))
        vp_pitch_deg = math.degrees(math.atan2(calib.cy - v_vp, calib.fy))
        vp_yaw_deg = math.degrees(
            math.atan2((u_vp - calib.cx) * math.cos(math.radians(vp_pitch_deg)), calib.fx)
        )
        print(f"independent VP estimate   : pitch {vp_pitch_deg:+.2f} deg, "
              f"yaw {vp_yaw_deg:+.2f} deg  "
              f"(agreement {abs(vp_pitch_deg - math.degrees(calib.pitch_rad)):.2f} deg)")

    # A calibration measured on native frames must be moved into preprocessed-frame
    # pixels before the pipeline, which runs on cropped frames, can use it.
    if source == "tusimple":
        pipeline_calib = calibration_for_preprocessed_frame(
            calib, measure_size, target_size, float(cfg.data.sky_frac)
        )
    else:
        pipeline_calib = calib

    out = Path(cfg.paths.output_root) / "calibration.json"
    out.write_text(json.dumps({
        "fx": pipeline_calib.fx, "fy": pipeline_calib.fy,
        "cx": pipeline_calib.cx, "cy": pipeline_calib.cy,
        "height_m": pipeline_calib.height_m,
        "pitch_rad": pipeline_calib.pitch_rad,
        "yaw_rad": pipeline_calib.yaw_rad,
        "image_size": list(target_size),
        "source": source,
        "measured_in_size": list(measure_size),
        "measured": {
            "fx": calib.fx, "fy": calib.fy, "cx": calib.cx, "cy": calib.cy,
            "pitch_deg": math.degrees(calib.pitch_rad),
            "yaw_deg": math.degrees(calib.yaw_rad),
        },
        "vanishing_point_pitch_deg": vp_pitch_deg,
        "vanishing_point_yaw_deg": vp_yaw_deg,
        "frames_used": len(lane_pairs),
        "hfov_deg_assumed": float(settings.hfov_deg),
        "parallelism_residual": lane_parallelism_cost(calib, lane_pairs),
        "pitch_source": pitch_source,
        "pitch_identifiable": bool(interior),
        "recovered_lane_width_median_m": float(np.median(widths)) if widths.size else None,
        "note": "yaw from zero median lane bearing, height scaled to the standard lane "
                "width, focal length from assumed HFOV carried through the sky crop. "
                "Pitch is assumed, not fitted: it is not identifiable from CurveLanes, "
                "which aggregates many cameras. The vanishing point is unusable here "
                "because lane annotations stop short of the horizon.",
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
