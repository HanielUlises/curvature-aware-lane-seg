"""Evaluate the segmenter on control-relevant errors, stratified by curvature.

Roadmap step four. Runs the model over the validation subset, recovers road geometry
from the predicted mask and from the cached ground-truth mask through the same
pipeline, and reports the errors a kinematic MPC is sensitive to: lateral offset,
heading, and curvature at fixed preview distances, per curvature bin.

    python -m scripts.eval_control
    python -m scripts.eval_control infer.ckpt=<ckpt> eval_limit=500

Frames where the prediction yields no ego lane are reported as detection failures
rather than dropped, since losing the lane is a control failure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform, preprocess_geometry
from src.eval.control_metrics import StratifiedControlMetric, control_errors
from src.geometry.calibration import (
    CameraCalibration,
    ground_plane_from_calibration,
    intrinsics_for_preprocessed_frame,
)
from src.geometry.ipm import build_ground_homography
from src.geometry.road_geometry import (
    DEFAULT_OFFSET_DISTANCE_M,
    road_geometry_from_mask,
)
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_video import _latest_run, _resolve_ckpt

BIN_NAMES = ("near-straight", "gentle", "moderate", "sharp", "tightest")


def _fmt(value: float, spec: str = ".3f") -> str:
    return "n/a" if value is None or np.isnan(value) else format(value, spec)


def _build_ground_plane(cfg: DictConfig, target_size: tuple[int, int]):
    """Ground mapping selected by ``ipm.mode``.

    ``trapezoid`` uses the hand-specified road rectangle; ``calibrated`` uses a pinhole
    model, reading the JSON written by ``scripts.calibrate_camera`` when present and
    falling back to the configured assumptions otherwise.
    """
    mode = str(cfg.ipm.mode)
    if mode == "trapezoid":
        t = cfg.ipm.trapezoid
        return build_ground_homography(
            target_size,
            src_trapezoid=tuple(tuple(p) for p in t.src_trapezoid),
            lane_width_m=float(t.lane_width_m),
            look_ahead_m=float(t.look_ahead_m),
        )
    if mode != "calibrated":
        raise ValueError(f"unknown ipm.mode {mode!r}; expected trapezoid or calibrated")

    path = Path(cfg.paths.output_root) / "calibration.json"
    if path.exists():
        d = json.loads(path.read_text())
        calib = CameraCalibration(
            fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
            height_m=d["height_m"], pitch_rad=d["pitch_rad"], yaw_rad=d["yaw_rad"],
        )
        print(f"ground plane: calibrated, from {path}")
        if d.get("source") == "tusimple":
            # The evaluation set is CurveLanes, so a TuSimple calibration describes a
            # different camera. The mapping stays self-consistent, and it is applied
            # identically to prediction and ground truth, but the metres are not this
            # dataset's metres.
            print("WARNING: this calibration was measured on TuSimple, while the "
                  "evaluation set is CurveLanes. Cross-bin comparisons remain valid; "
                  "absolute units do not transfer between cameras.")
    else:
        s = cfg.ipm.calibrated
        fx, fy, cx, cy = intrinsics_for_preprocessed_frame(
            (2560, 1440), target_size, float(cfg.data.sky_frac), float(s.hfov_deg)
        )
        calib = CameraCalibration(
            fx=fx, fy=fy, cx=cx, cy=cy, height_m=float(s.height_m),
            pitch_rad=math.radians(float(s.pitch_deg)), yaw_rad=0.0,
        )
        print("ground plane: calibrated, from configured assumptions "
              "(run scripts.calibrate_camera to write calibration.json)")
    return ground_plane_from_calibration(calib)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    target_size = tuple(cfg.data.target_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifests = Path(cfg.paths.output_root) / "manifests"
    entries, meta = read_manifest(manifests / "val.json")
    masks_dir = Path(meta["mask_dir"])
    bin_edges = meta["bin_edges"]

    previews = tuple(float(d) for d in cfg.ipm.preview_distances_m)
    ground = _build_ground_plane(cfg, target_size)

    ckpt = _resolve_ckpt(cfg)
    print(f"loading {ckpt}")
    model = (
        LaneSegmenter.load_from_checkpoint(
            str(ckpt), bin_edges=bin_edges, map_location=device
        )
        .eval()
        .to(device)
    )
    transform = build_eval_transform()
    metric = StratifiedControlMetric(bin_edges, num_previews=len(previews))

    limit = cfg.get("eval_limit", None)
    if limit and int(limit) < len(entries):
        # The manifest is grouped by curvature bin, so a prefix slice would only see
        # the first bin. Stride evenly instead to keep the subsample stratified.
        idx = np.linspace(0, len(entries) - 1, int(limit)).round().astype(int)
        selected = [entries[i] for i in dict.fromkeys(idx.tolist())]
    else:
        selected = entries
    truth_missing = 0

    with torch.no_grad():
        for entry in selected:
            gt = cv2.imread(str(masks_dir / f"{entry.image_path.stem}.png"),
                            cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            truth_geom = road_geometry_from_mask(
                (gt > 0).astype(np.uint8), ground, previews
            )
            if truth_geom is None:
                # No reference geometry: the annotation itself has no ego lane here, so
                # the frame cannot score a prediction either way.
                truth_missing += 1
                continue

            bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            image = preprocess_geometry(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), target_size,
                cfg.data.sky_frac, cv2.INTER_AREA,
            )
            logits = model(transform(image=image)["image"].unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)

            pred_geom = road_geometry_from_mask(pred, ground, previews)
            if pred_geom is None:
                metric.update_failure(entry.kappa)
                continue
            metric.update(control_errors(pred_geom, truth_geom), entry.kappa)

    overall, per_bin = metric.compute()
    preview_hdr = "  ".join(f"k@{d:g}m" for d in previews)
    print(f"\nframes scored: {overall.detected + overall.failed} "
          f"(reference geometry unavailable on {truth_missing})")
    print(f"\n{'bin':<14} {'n':>5} {'det%':>6} {'offset':>9} {'heading':>9}  {preview_hdr}")
    for i, s in enumerate(per_bin):
        name = BIN_NAMES[i] if i < len(BIN_NAMES) else f"bin{i}"
        kappa_cols = "  ".join(f"{_fmt(v, '.4f'):>7}" for v in s.curvature_mae_1pm)
        print(f"{name:<14} {s.detected:>5} {_fmt(s.detection_rate * 100, '.1f'):>6} "
              f"{_fmt(s.offset_mae_m):>9} {_fmt(s.heading_mae_deg, '.2f'):>9}  {kappa_cols}")
    kappa_cols = "  ".join(f"{_fmt(v, '.4f'):>7}" for v in overall.curvature_mae_1pm)
    print(f"{'OVERALL':<14} {overall.detected:>5} "
          f"{_fmt(overall.detection_rate * 100, '.1f'):>6} "
          f"{_fmt(overall.offset_mae_m):>9} {_fmt(overall.heading_mae_deg, '.2f'):>9}  "
          f"{kappa_cols}")
    print(f"\noffset = m (MAE) at {DEFAULT_OFFSET_DISTANCE_M:g} m ahead, heading = deg (MAE), "
          "k@Nm = signed curvature MAE in 1/m")
    print("Metric units depend on the placeholder IPM calibration; relative "
          "comparisons across bins and models are the meaningful readings.")

    out = _latest_run() / "control_metrics.json"
    out.write_text(json.dumps({
        "checkpoint": str(ckpt),
        "preview_distances_m": list(previews),
        "bin_edges": bin_edges,
        "overall": {
            "offset_mae_m": overall.offset_mae_m,
            "heading_mae_deg": overall.heading_mae_deg,
            "curvature_mae_1pm": overall.curvature_mae_1pm.tolist(),
            "detected": overall.detected,
            "failed": overall.failed,
            "detection_rate": overall.detection_rate,
        },
        "per_bin": [
            {
                "name": BIN_NAMES[i] if i < len(BIN_NAMES) else f"bin{i}",
                "offset_mae_m": s.offset_mae_m,
                "heading_mae_deg": s.heading_mae_deg,
                "curvature_mae_1pm": s.curvature_mae_1pm.tolist(),
                "detected": s.detected,
                "failed": s.failed,
                "detection_rate": s.detection_rate,
            }
            for i, s in enumerate(per_bin)
        ],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
