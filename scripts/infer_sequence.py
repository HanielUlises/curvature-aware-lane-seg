"""Inference on an ordered image sequence or a video -> one consistent video.

Unlike :mod:`scripts.infer_video` (which sweeps the stratified val set by
curvature), this runs the model on **temporally ordered frames** — a folder of
sequential images such as a TuSimple clip, or a raw video file — and writes a
single output video with the predicted lane mask overlaid. No ground truth is
required; this is pure inference for a qualitative driving demo.

    # a TuSimple clip (20 sequential frames)
    python -m scripts.infer_sequence infer.source=data/raw/tusimple/clips/0601/1494452381594376146
    # a video file
    python -m scripts.infer_sequence infer.source=drive.mp4 infer.fps=20

Frames of any resolution are center-cropped to the model's target aspect before
the sky-crop + resize, so a non-16:9 source is handled without distortion.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform, preprocess_geometry
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_video import _latest_run, _overlay, _resolve_ckpt

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _center_crop_aspect(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Center-crop ``img`` to the target aspect ratio (no scaling)."""
    h, w = img.shape[:2]
    tw, th = target_size
    tar = tw / th
    cur = w / h
    if abs(cur - tar) < 1e-3:
        return img
    if cur > tar:  # too wide -> crop width
        nw = int(round(h * tar))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / tar))  # too tall -> crop height
    y0 = (h - nh) // 2
    return img[y0:y0 + nh, :]


def _ordered_images(source: Path) -> list[Path]:
    """Ordered image paths under ``source``.

    Handles both a single clip dir (images directly inside) and a parent dir whose
    children are clip dirs (e.g. a TuSimple date folder) — clips are visited in
    natural (timestamp) order, frames within each clip in natural order, so
    consecutive clips read as continuous driving.
    """
    direct = sorted((p for p in source.iterdir() if p.suffix.lower() in IMG_EXT),
                    key=_natural_key)
    if direct:
        return direct
    frames: list[Path] = []
    for clip in sorted((d for d in source.iterdir() if d.is_dir()), key=_natural_key):
        frames.extend(sorted((p for p in clip.iterdir() if p.suffix.lower() in IMG_EXT),
                             key=_natural_key))
    return frames


def _frames_from_source(source: Path, max_frames=None):
    """Yield RGB frames from an image directory (or nested clips) or a video file."""
    if source.is_dir():
        paths = _ordered_images(source)
        if max_frames:
            paths = paths[:max_frames]
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is not None:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(str(source))
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()


@torch.no_grad()
def _predict(model, image_rgb, transform, target_size, sky, tta, device,
             threshold: float = 0.5):
    image = preprocess_geometry(image_rgb, target_size, sky, cv2.INTER_AREA)
    x = transform(image=image)["image"].unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))
    if tta:
        pf = torch.sigmoid(model(torch.flip(x, dims=[3])))
        prob = 0.5 * (prob + torch.flip(pf, dims=[3]))
    # Thresholded here rather than downstream: a curvature-weighted model is more
    # conservative than the baseline and sits at a different point on its operating
    # curve, so the two are only comparable when each is given its own threshold.
    pred = (prob[0, 0].cpu().numpy() >= threshold).astype(np.uint8)
    return image, pred


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.infer.source:
        raise ValueError("set infer.source=<image-dir-or-video-file>")
    source = Path(cfg.infer.source)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    target_size = tuple(cfg.data.target_size)
    sky = cfg.data.sky_frac
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    ckpt = _resolve_ckpt(cfg)
    print(f"loading {ckpt}  (TTA={cfg.infer.tta})")
    model = LaneSegmenter.load_from_checkpoint(
        str(ckpt), bin_edges=meta["bin_edges"], map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else _latest_run() / "sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = target_size
    out_path = out_dir / f"{source.stem or 'sequence'}_pred.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             cfg.infer.fps, (w, h))
    n = 0
    max_frames = cfg.infer.get("max_frames", None)
    for rgb in _frames_from_source(source, max_frames):
        rgb = _center_crop_aspect(rgb, target_size)
        image, pred = _predict(model, rgb, transform, target_size, sky, cfg.infer.tta, device)
        vis = _overlay(image, pred, None, show_gt=False)
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    print(f"wrote {n} frames -> {out_path}")
    if n == 0:
        print("WARNING: no frames read from source")


if __name__ == "__main__":
    main()
