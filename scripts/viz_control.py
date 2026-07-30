"""Visualize the whole perception-to-control chain, frame by frame.

The top row places the camera view, carrying the predicted lane mask, the extracted lane
polylines and the ego centreline, beside the metric bird's-eye view produced by the
calibrated ground projection, with a distance grid, preview markers and the three
quantities a kinematic MPC consumes: lateral offset, heading error and signed curvature.

Underneath, a trace strip plots the raw per-frame estimate against the temporally
filtered signal (:mod:`src.geometry.temporal`) over a sliding window. Raw geometry is far
noisier than a vehicle can physically move, so the strip is the clearest statement of why
the filter exists, and it breaks the raw line at frames where no ego lane was recovered
while the filtered line coasts through them.

The bird's-eye panel is only meaningful with a real calibration, so it reads
``calibration.json`` and refuses to guess. Because that calibration is measured on
TuSimple, point this at TuSimple footage.

    python -m scripts.viz_control infer.source=<clip-dir-or-video>
    python -m scripts.viz_control infer.source=<date-dir> infer.max_frames=300
"""

from __future__ import annotations

import json
import math
from collections import deque
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
from src.geometry.road_geometry import DEFAULT_OFFSET_DISTANCE_M, road_geometry
from src.geometry.temporal import RoadGeometryFilter
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_sequence import _center_crop_aspect, _frames_from_source, _predict
from scripts.infer_video import _latest_run, _resolve_ckpt

# Bird's-eye panel extent and styling.
BEV_WIDTH = 320
X_MAX_M = 7.0
Z_MAX_M = 35.0
# Trace strip under the two panels, showing raw against filtered over time.
TRACE_HEIGHT = 96
TRACE_WINDOW = 100
COL_RAW = (120, 120, 132)
COL_FILT = (90, 220, 140)
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


def _draw_trace(panel, x0, width, raw, filt, label, unit):
    """One sparkline of raw against filtered history inside ``panel``."""
    h = panel.shape[0]
    top, bottom = 18, h - 6
    cv2.putText(panel, label, (x0 + 6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (170, 170, 178), 1, cv2.LINE_AA)

    values = [v for v in list(raw) + list(filt) if v is not None and np.isfinite(v)]
    if not values:
        return
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.12 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def to_px(i, v, n):
        u = x0 + 6 + int((width - 14) * i / max(n - 1, 1))
        y = bottom - int((bottom - top) * (v - lo) / (hi - lo))
        return u, y

    # Zero line, since the sign of offset and curvature is what matters.
    if lo < 0.0 < hi:
        _, yz = to_px(0, 0.0, len(filt))
        cv2.line(panel, (x0 + 6, yz), (x0 + width - 8, yz), (58, 58, 66), 1)

    for series, colour, thickness in ((raw, COL_RAW, 1), (filt, COL_FILT, 2)):
        pts = [
            to_px(i, v, len(series))
            for i, v in enumerate(series)
            if v is not None and np.isfinite(v)
        ]
        # Break the polyline at gaps so a missed detection reads as a gap, not a jump.
        prev_i = None
        for (i, v), p in zip(
            [(i, v) for i, v in enumerate(series) if v is not None and np.isfinite(v)],
            pts,
        ):
            if prev_i is not None and i - prev_i == 1:
                cv2.line(panel, prev_p, p, colour, thickness, cv2.LINE_AA)
            prev_i, prev_p = i, p

    latest = next((v for v in reversed(filt) if v is not None and np.isfinite(v)), None)
    if latest is not None:
        cv2.putText(panel, f"{latest:+.3f} {unit}", (x0 + width - 92, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_FILT, 1, cv2.LINE_AA)


def _draw_trace_strip(width, hist):
    """The full trace strip: offset on the left, curvature on the right."""
    panel = np.full((TRACE_HEIGHT, width, 3), 32, dtype=np.uint8)
    half = width // 2
    _draw_trace(panel, 0, half, hist["offset_raw"], hist["offset_filt"],
                "lateral offset   raw vs filtered", "m")
    _draw_trace(panel, half, width - half, hist["kappa_raw"], hist["kappa_filt"],
                "curvature   raw vs filtered", "1/m")
    cv2.line(panel, (half, 0), (half, TRACE_HEIGHT), (48, 48, 56), 1)
    return panel


def _render(image_rgb, mask, polylines, centre_img, geom, ground, previews,
            filtered=None, hist=None, offset_distance_m=DEFAULT_OFFSET_DISTANCE_M):
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
        # Fade the near end. How far the centreline reaches depends on where the mask
        # happens to detect boundaries, which varies frame to frame, so a hard endpoint
        # draws the eye to an extent that carries no information. The positions drawn
        # are unchanged; only their opacity near the end is.
        pts = centre_img.astype(int)
        n_pts = len(pts)
        fade_from = int(n_pts * 0.65)
        for i, (x, y) in enumerate(pts):
            if not (0 <= x < w and 0 <= y < h):
                continue
            alpha = 1.0
            if i >= fade_from and n_pts > fade_from:
                alpha = 1.0 - (i - fade_from) / max(n_pts - fade_from, 1)
                alpha = max(alpha, 0.0) ** 0.8
            if alpha <= 0.02:
                continue
            patch = left[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].astype(np.float32)
            blend = np.array(COL_CENTER, dtype=np.float32)
            left[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3] = (
                (1 - alpha) * patch + alpha * blend
            ).astype(np.uint8)

    panel = np.full((h, BEV_WIDTH, 3), 38, dtype=np.uint8)
    _draw_bev_grid(panel)
    cv2.putText(panel, "bird's-eye (metric)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (170, 170, 178), 1, cv2.LINE_AA)

    if geom is None:
        note = "no ego lane"
        if filtered is not None and filtered.coasting_frames:
            note += f" (coasting {filtered.coasting_frames})"
        cv2.putText(panel, note, (BEV_WIDTH // 2 - 62, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 210), 1, cv2.LINE_AA)
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
        # Dim the strip behind the readout so it stays legible over the lane markers.
        strip = panel[h - 62 :, :]
        panel[h - 62 :, :] = (strip * 0.25).astype(np.uint8)
        cv2.putText(panel, "raw", (128, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    COL_RAW, 1, cv2.LINE_AA)
        cv2.putText(panel, "filtered", (208, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    COL_FILT, 1, cv2.LINE_AA)
        rows = [
            (f"offset@{geom.offset_distance_m:g}m", f"{geom.lateral_offset_m:+.2f}",
             f"{filtered.lateral_offset_m:+.2f}" if filtered else "", "m"),
            ("heading", f"{math.degrees(geom.heading_error_rad):+.1f}",
             f"{math.degrees(filtered.heading_error_rad):+.1f}" if filtered else "", "deg"),
            ("kappa", f"{geom.curvature_1pm:+.4f}",
             f"{filtered.curvature_1pm:+.4f}" if filtered else "", "1/m"),
        ]
        for i, (name, raw_v, filt_v, unit) in enumerate(rows):
            y = h - 36 + i * 12
            cv2.putText(panel, f"{name} ({unit})", (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, COL_TEXT, 1, cv2.LINE_AA)
            cv2.putText(panel, raw_v, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        COL_RAW, 1, cv2.LINE_AA)
            cv2.putText(panel, filt_v, (208, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        COL_FILT, 1, cv2.LINE_AA)

    composed = np.hstack([left, panel])
    if hist is not None:
        composed = np.vstack([composed, _draw_trace_strip(composed.shape[1], hist)])
    # A border keeps the frame's extent readable when the page behind it is also dark,
    # which otherwise makes the dark panels look like the image is cropped.
    cv2.rectangle(composed, (0, 0), (composed.shape[1] - 1, composed.shape[0] - 1),
                  (110, 110, 120), 1)
    return composed


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
        (target_size[0] + BEV_WIDTH, target_size[1] + TRACE_HEIGHT),
    )
    filt = RoadGeometryFilter(dt=1.0 / max(float(cfg.infer.fps), 1.0))
    hist = {k: deque([None] * TRACE_WINDOW, maxlen=TRACE_WINDOW)
            for k in ("offset_raw", "offset_filt", "kappa_raw", "kappa_filt")}

    # Optionally also write each composed frame as PNG. Building the GIF from these
    # avoids inheriting the video codec's artifacts.
    frames_dir = None
    if bool(cfg.infer.get("dump_frames", False)):
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in frames_dir.glob("*.png"):
            stale.unlink()

    max_frames = cfg.infer.get("max_frames", None)
    n = detected = 0
    for rgb in _frames_from_source(source, max_frames):
        rgb = _center_crop_aspect(rgb, target_size)
        image, pred = _predict(
            model, rgb, transform, target_size, cfg.data.sky_frac, cfg.infer.tta, device
        )
        polylines = extract_lane_polylines(pred)
        from src.geometry.centerline import ego_centerline

        centre_img = ego_centerline(polylines, target_size[0],
                                    image_height=target_size[1])
        geom = (
            road_geometry(centre_img, ground, previews)
            if centre_img is not None else None
        )
        detected += geom is not None
        smoothed = filt.update(geom)
        hist["offset_raw"].append(geom.lateral_offset_m if geom else None)
        hist["kappa_raw"].append(geom.curvature_1pm if geom else None)
        hist["offset_filt"].append(smoothed.lateral_offset_m)
        hist["kappa_filt"].append(smoothed.curvature_1pm)
        frame = _render(image, pred, polylines, centre_img, geom, ground, previews,
                        smoothed, hist)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        if frames_dir is not None:
            cv2.imwrite(str(frames_dir / f"{n:04d}.png"), bgr)
        n += 1
    writer.release()
    print(f"wrote {n} frames ({detected} with an ego lane) -> {out_path}")


if __name__ == "__main__":
    main()
