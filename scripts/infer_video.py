"""Qualitative inference: render predicted lane masks into a video.

Loads a trained :class:`~src.models.lane_segmenter.LaneSegmenter` checkpoint, runs
it over the validation subset, and encodes an MP4 that sweeps from straight to
tight-curve (frames ordered by the curvature label). Each frame overlays the
predicted lane mask (red) and, optionally, the ground-truth contour (green), with
a caption showing the frame's curvature and per-frame IoU — so the video shows,
qualitatively, how the model holds up as curvature increases.

    python -m scripts.infer_video
    python -m scripts.infer_video infer.ckpt=outputs/<run>/checkpoints/best.ckpt
    python -m scripts.infer_video infer.n_frames=200 infer.show_gt=false

Also writes the 6 best / 6 worst frames as stills for the results write-up.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform, preprocess_geometry
from src.models.lane_segmenter import LaneSegmenter


def _latest_run() -> Path:
    runs = sorted(Path("outputs").glob("*/"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("no outputs/<run> directories found")
    return runs[-1]


def _resolve_ckpt(cfg: DictConfig) -> Path:
    if cfg.infer.ckpt:
        return Path(cfg.infer.ckpt)
    ckpt_dir = _latest_run() / "checkpoints"
    # Prefer the monitored best (epoch-tagged) over last.ckpt.
    best = sorted(p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if best:
        # File name is "{epoch}-{val/iou}.ckpt"; pick the highest IoU.
        return max(best, key=lambda p: float(p.stem.split("-")[-1]))
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")


def _overlay(image_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray | None,
             show_gt: bool) -> np.ndarray:
    """Blend predicted mask (red) + optional GT contour (green) onto the image."""
    vis = image_rgb.copy()
    red = np.zeros_like(vis)
    red[..., 0] = 255
    m = pred.astype(bool)
    vis[m] = (0.45 * red[m] + 0.55 * vis[m]).astype(np.uint8)
    if show_gt and gt is not None:
        contours, _ = cv2.findContours(
            gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis


def _frame_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / (union + eps)) if union > 0 else float("nan")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.infer.seed)
    target_size = tuple(cfg.data.target_size)  # (W, H)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifests = Path(cfg.paths.output_root) / "manifests"
    entries, meta = read_manifest(manifests / "val.json")
    masks_dir = Path(meta["mask_dir"])
    bin_edges = meta["bin_edges"]

    ckpt = _resolve_ckpt(cfg)
    print(f"loading {ckpt}")
    model = LaneSegmenter.load_from_checkpoint(
        str(ckpt), bin_edges=bin_edges, map_location=device
    )
    model.eval().to(device)

    if cfg.infer.order == "curvature":
        entries = sorted(entries, key=lambda e: e.kappa)
    else:
        random.shuffle(entries)
    # Evenly subsample to n_frames so the sweep spans the whole curvature range.
    n = min(cfg.infer.n_frames, len(entries))
    idx = np.linspace(0, len(entries) - 1, n).round().astype(int)
    picks = [entries[i] for i in idx]

    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else _latest_run() / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = build_eval_transform()

    w, h = target_size
    writer = cv2.VideoWriter(
        str(out_dir / "lane_predictions.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.infer.fps,
        (w, h),
    )

    scored: list[tuple[float, np.ndarray, float]] = []  # (iou, vis_bgr, kappa)
    with torch.no_grad():
        for e in picks:
            bgr = cv2.imread(str(e.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = preprocess_geometry(image, target_size, cfg.data.sky_frac, cv2.INTER_AREA)
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            pred = (prob >= cfg.infer.threshold).astype(np.uint8)

            gt = cv2.imread(str(masks_dir / f"{e.image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt = (gt > 0).astype(np.uint8) if gt is not None else None
            iou = _frame_iou(pred, gt) if gt is not None else float("nan")

            vis = _overlay(image, pred, gt, cfg.infer.show_gt)
            label = f"kappa={e.kappa:.2f}  bin{e.bin_index}  IoU={iou:.2f}"
            cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            writer.write(vis_bgr)
            if not np.isnan(iou):
                scored.append((iou, vis_bgr, e.kappa))

    writer.release()
    print(f"video -> {out_dir/'lane_predictions.mp4'} ({len(picks)} frames)")

    # Best / worst stills for the write-up.
    scored.sort(key=lambda t: t[0])
    for tag, group in (("worst", scored[:6]), ("best", scored[-6:])):
        for i, (iou, vis_bgr, kappa) in enumerate(group):
            cv2.imwrite(str(out_dir / f"{tag}_{i}_iou{iou:.2f}_k{kappa:.1f}.png"), vis_bgr)
    print(f"stills -> {out_dir} (best/worst x6)")


if __name__ == "__main__":
    main()
