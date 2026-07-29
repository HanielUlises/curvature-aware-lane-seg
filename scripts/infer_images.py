"""Inference on previously-unseen frames (generalization check).

Samples frames from the CurveLanes ``valid`` split that are **not** in the eval
manifest — so they were seen neither during training (the training subset is drawn
from the ``train`` split) nor during checkpoint selection. Runs the trained model,
overlays predictions, and — because ``valid`` ships labels — scores IoU with the
same curvature bins used for stratification, answering two questions:

    does the reported accuracy hold on unseen data, and does the tail still not
    collapse?

    python -m scripts.infer_images
    python -m scripts.infer_images infer.unseen_n=500 infer.ckpt=<...>.ckpt
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.curvelanes import build_frame
from src.data.rasterize import is_target_aspect, rasterize_native
from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform, preprocess_geometry
from src.eval.metrics import assign_bins
from src.geometry.curvature import frame_curvature
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_video import _frame_iou, _latest_run, _overlay, _resolve_ckpt


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.infer.seed)
    target_size = tuple(cfg.data.target_size)
    sky = cfg.data.sky_frac
    stroke = int(cfg.data.stroke_px)
    tol = float(cfg.data.aspect_tol)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifests = Path(cfg.paths.output_root) / "manifests"
    val_entries, meta = read_manifest(manifests / "val.json")
    bin_edges = meta["bin_edges"]
    seen = {e.image_path.stem for e in val_entries}

    valid_root = Path(cfg.paths.data_root) / cfg.data.valid_subdir
    images_dir = valid_root / cfg.data.images_subdir
    labels_dir = valid_root / cfg.data.labels_subdir
    all_imgs = sorted(images_dir.glob("*.jpg"))
    unseen = [p for p in all_imgs if p.stem not in seen]
    random.shuffle(unseen)
    print(f"{len(all_imgs)} valid frames, {len(seen)} in eval manifest, "
          f"{len(unseen)} unseen candidates")

    ckpt = _resolve_ckpt(cfg)
    print(f"loading {ckpt}")
    model = LaneSegmenter.load_from_checkpoint(
        str(ckpt), bin_edges=bin_edges, map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

    nb = len(bin_edges) - 1
    bI, bU, bC = np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=int)
    gI = gU = 0.0
    records = []  # (kappa, iou, image_path)

    scanned = 0
    with torch.no_grad():
        for p in unseen:
            if len(records) >= cfg.infer.unseen_n:
                break
            scanned += 1
            label = labels_dir / f"{p.stem}.lines.json"
            if not label.exists():
                continue
            try:
                frame = build_frame(p, label)
                if not is_target_aspect(frame.width, frame.height, target_size, tol):
                    continue
                gt_native = rasterize_native(frame, target_size, stroke, tol)
            except ValueError:
                continue
            gt = preprocess_geometry(gt_native, target_size, sky, cv2.INTER_NEAREST)
            gt = (gt > 0).astype(np.uint8)

            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = preprocess_geometry(rgb, target_size, sky, cv2.INTER_AREA)
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            pred = (prob >= cfg.infer.threshold).astype(np.uint8)

            iou = _frame_iou(pred, gt)
            kappa = float(frame_curvature(frame))
            b = int(assign_bins(np.array([kappa]), bin_edges)[0])
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            gI += inter; gU += union
            bI[b] += inter; bU[b] += union; bC[b] += 1
            records.append((kappa, iou, image, pred, gt))

    n = len(records)
    print(f"\nscored {n} unseen frames (scanned {scanned})")
    print(f"OVERALL unseen IoU: {gI/(gU+1e-7):.4f}")
    print("per curvature bin:")
    names = ["straight", "gentle", "moderate", "sharp", "tightest"]
    for k in range(nb):
        iou = bI[k]/(bU[k]+1e-7) if bC[k] else float("nan")
        print(f"  bin{k} {names[k]:9s} n={bC[k]:3d}  IoU={iou:.4f}")

    # render video (ordered straight->curvy) + best/worst stills
    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else _latest_run() / "unseen"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_k = sorted(records, key=lambda r: r[0])
    nv = min(cfg.infer.unseen_video, len(by_k))
    idx = np.linspace(0, len(by_k) - 1, nv).round().astype(int)
    w, h = target_size
    writer = cv2.VideoWriter(str(out_dir / "unseen_predictions.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), cfg.infer.fps, (w, h))
    for i in idx:
        kappa, iou, image, pred, gt = by_k[i]
        vis = _overlay(image, pred, gt, cfg.infer.show_gt)
        lab = f"UNSEEN  kappa={kappa:.2f}  IoU={iou:.2f}"
        cv2.putText(vis, lab, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, lab, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    writer.release()

    by_iou = sorted([r for r in records if not np.isnan(r[1])], key=lambda r: r[1])
    for tag, grp in (("worst", by_iou[:6]), ("best", by_iou[-6:])):
        for j, (kappa, iou, image, pred, gt) in enumerate(grp):
            vis = _overlay(image, pred, gt, cfg.infer.show_gt)
            cv2.imwrite(str(out_dir / f"{tag}_{j}_iou{iou:.2f}_k{kappa:.1f}.png"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\nvideo + stills -> {out_dir}")


if __name__ == "__main__":
    main()
