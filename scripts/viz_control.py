"""Visualize the whole perception-to-control chain, frame by frame.

Renders two panels side by side. The left panel is the camera view carrying the
predicted lane mask, the extracted lane polylines and the ego centreline. The right
panel is the metric bird's-eye view produced by the calibrated ground projection, with
a distance grid, the projected lane boundaries and centreline, preview markers, and the
three quantities a kinematic MPC consumes: lateral offset, heading error and signed
curvature.

The bird's-eye panel is only meaningful with a real calibration, so it reads
``calibration.json`` and refuses to guess. Because that calibration is measured on
TuSimple, point this at TuSimple footage.

    python -m scripts.viz_control infer.source=<clip-dir-or-video>
    python -m scripts.viz_control infer.source=<date-dir> infer.max_frames=300
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
from src.data.transforms import build_eval_transform
from src.geometry.calibration import CameraCalibration, ground_plane_from_calibration
from src.geometry.centerline import ego_lane_pair, extract_lane_polylines
from src.geometry.road_geometry import road_geometry
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_sequence import _center_crop_aspect, _frames_from_source, _predict
from scripts.infer_video import _latest_run, _resolve_ckpt

# Bird's-eye panel extent and styling.
BEV_WIDTH = 320
X_MAX_M = 7.0
Z_MAX_M = 35.0
COL_MASK = (232, 62, 62)      # prediction, red
COL_LANE = (0, 200, 255)      # extracted boundaries, cyan
COL_CENTER = (255, 225, 0)    # ego centreline, yellow
COL_GRID = (70, 70, 78)
COL_TEXT = (240, 240, 245)


def _load_calibration(cfg: DictConfig) -> CameraCalibration:
    path = Path(cfg.paths.output_root) / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.calibrate_camera` first; the "
            "bird's-eye panel is metric and needs a real calibration."
        )
    d = json.loads(path.read_text())
    return CameraCalibration(
        fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
        height_m=d["height_m"], pitch_rad=d["pitch_rad"], yaw_rad=d["yaw_rad"],
    )


def _bev_point(x_m: float, z_m: float, height: int) -> tuple[int, int]:
    """Ground metres to bird's-eye panel pixels (z up the panel, x across)."""
    u = (x_m + X_MAX_M) / (2.0 * X_MAX_M) * BEV_WIDTH
    v = height - (z_m / Z_MAX_M) * height
    return int(round(u)), int(round(v))


def _draw_bev_grid(panel: np.ndarray) -> None:
    h = panel.shape[0]
    for z in range(5, int(Z_MAX_M) + 1, 5):
        _, v = _bev_point(0.0, z, h)
        cv2.line(panel, (0, v), (BEV_WIDTH, v), COL_GRID, 1)
        cv2.putText(panel, f"{z}m", (4, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (150, 150, 158), 1, cv2.LINE_AA)
    for x in (-6, -4, -2, 0, 2, 4, 6):
        u, _ = _bev_point(float(x), 0.0, h)
        cv2.line(panel, (u, 0), (u, h), COL_GRID, 1)
    # Ego vehicle footprint at the origin.
    u0, v0 = _bev_point(0.0, 0.0, h)
    cv2.rectangle(panel, (u0 - 9, v0 - 16), (u0 + 9, v0), (110, 110, 120), -1)


def _draw_ground_polyline(panel: np.ndarray, pts: np.ndarray, colour, radius=1) -> None:
    h = panel.shape[0]
    for x, z in pts:
        if 0.0 <= z <= Z_MAX_M and abs(x) <= X_MAX_M:
            cv2.circle(panel, _bev_point(float(x), float(z), h), radius, colour, -1)


def _render(image_rgb, mask, polylines, centre_img, geom, ground, previews):
    """Compose the camera panel and the bird's-eye panel into one frame."""
    h, w = image_rgb.shape[:2]
    left = image_rgb.copy()
    red = np.zeros_like(left)
    red[..., 0], red[..., 1], red[..., 2] = COL_MASK
    m = mask.astype(bool)
    left[m] = (0.45 * red[m] + 0.55 * left[m]).astype(np.uint8)
    for poly in polylines:
        for x, y in poly.astype(int):
            cv2.circle(left, (x, y), 1, COL_LANE, -1)
    if centre_img is not None:
        for x, y in centre_img.astype(int):
            cv2.circle(left, (x, y), 2, COL_CENTER, -1)

    panel = np.full((h, BEV_WIDTH, 3), 22, dtype=np.uint8)
    _draw_bev_grid(panel)
    cv2.putText(panel, "bird's-eye (metric)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (170, 170, 178), 1, cv2.LINE_AA)

    if geom is None:
        cv2.putText(panel, "no ego lane", (BEV_WIDTH // 2 - 44, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 200), 1, cv2.LINE_AA)
    else:
        pair = ego_lane_pair(polylines, w)
        if pair is not None:
            for lane in pair:
                _draw_ground_polyline(panel, ground.to_ground(lane), COL_LANE)
        _draw_ground_polyline(panel, geom.ground_centerline, COL_CENTER, radius=2)
        # Preview markers where curvature is reported to the controller.
        for dist, kappa in zip(previews, geom.preview_curvature_1pm):
            if dist <= Z_MAX_M and not np.isnan(kappa):
                u, v = _bev_point(0.0, float(dist), h)
                cv2.drawMarker(panel, (u, v), (200, 200, 210), cv2.MARKER_TILTED_CROSS,
                               6, 1)
        rows = [
            f"offset  {geom.lateral_offset_m:+.2f} m",
            f"heading {math.degrees(geom.heading_error_rad):+.1f} deg",
            f"kappa   {geom.curvature_1pm:+.4f} 1/m",
        ]
        radius = 1.0 / abs(geom.curvature_1pm) if abs(geom.curvature_1pm) > 1e-4 else None
        rows.append(f"radius  {radius:.0f} m" if radius and radius < 1e4 else "radius  straight")
        # Dim the strip behind the readout so it stays legible over the lane markers.
        strip = panel[h - 62 :, :]
        panel[h - 62 :, :] = (strip * 0.25).astype(np.uint8)
        for i, text in enumerate(rows):
            cv2.putText(panel, text, (6, h - 46 + i * 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, COL_TEXT, 1, cv2.LINE_AA)

    return np.hstack([left, panel])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.infer.source:
        raise ValueError("set infer.source=<clip-dir-or-video>")
    source = Path(cfg.infer.source)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    target_size = tuple(cfg.data.target_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    previews = tuple(float(d) for d in cfg.ipm.preview_distances_m)

    calib = _load_calibration(cfg)
    ground = ground_plane_from_calibration(calib)
    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")

    ckpt = _resolve_ckpt(cfg)
    print(f"loading {ckpt}")
    model = LaneSegmenter.load_from_checkpoint(
        str(ckpt), bin_edges=meta["bin_edges"], map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else _latest_run() / "control_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem or 'sequence'}_control.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.infer.fps,
        (target_size[0] + BEV_WIDTH, target_size[1]),
    )

    max_frames = cfg.infer.get("max_frames", None)
    n = detected = 0
    for rgb in _frames_from_source(source, max_frames):
        rgb = _center_crop_aspect(rgb, target_size)
        image, pred = _predict(
            model, rgb, transform, target_size, cfg.data.sky_frac, cfg.infer.tta, device
        )
        polylines = extract_lane_polylines(pred)
        from src.geometry.centerline import ego_centerline

        centre_img = ego_centerline(polylines, target_size[0])
        geom = (
            road_geometry(centre_img, ground, previews)
            if centre_img is not None else None
        )
        detected += geom is not None
        frame = _render(image, pred, polylines, centre_img, geom, ground, previews)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    print(f"wrote {n} frames ({detected} with an ego lane) -> {out_path}")


if __name__ == "__main__":
    main()
