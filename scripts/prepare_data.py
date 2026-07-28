"""Orchestrate CurveLanes preprocessing: parse -> curvature -> subset -> mask cache.

Hydra entry point. Produces, under ``paths.output_root`` (on the external disk):

- ``manifests/train.json`` and ``manifests/val.json`` — the stratified subsets.
- ``masks/{train,val}/<stem>.png`` — cached target-resolution lane masks.
- ``manifests/histogram.json`` — natural vs selected curvature histograms.

Run:
    python -m scripts.prepare_data
    python -m scripts.prepare_data subset.train_target=5000 data.num_workers=12
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.curvelanes import build_frame, index_split
from src.data.rasterize import rasterize_native
from src.data.subset import (
    Subset,
    build_subset,
    compute_frame_curvatures,
    format_histogram,
    write_manifest,
)
from src.data.transforms import preprocess_geometry


def _precompute_mask(args: tuple) -> str:
    image_path, label_path, masks_dir, target_size, stroke_px, aspect_tol, sky_frac = args
    frame = build_frame(Path(image_path), Path(label_path))
    native = rasterize_native(frame, tuple(target_size), stroke_px, aspect_tol)
    mask = preprocess_geometry(native, tuple(target_size), sky_frac, cv2.INTER_NEAREST)
    out = Path(masks_dir) / f"{Path(image_path).stem}.png"
    cv2.imwrite(str(out), mask)
    return out.name


def _cache_masks(subset: Subset, masks_dir: Path, cfg: DictConfig) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    target_size = list(cfg.data.target_size)
    work = [
        (
            str(e.image_path),
            str(e.label_path),
            str(masks_dir),
            target_size,
            int(cfg.data.stroke_px),
            float(cfg.data.aspect_tol),
            float(cfg.data.sky_frac),
        )
        for e in subset.entries
    ]
    num_workers = int(cfg.data.num_workers)
    print(f"[mask] rasterizing {len(work)} masks -> {masks_dir} ({num_workers} workers)")
    if num_workers <= 1:
        for i, a in enumerate(work):
            _precompute_mask(a)
    else:
        with Pool(processes=num_workers) as pool:
            for _ in pool.imap_unordered(_precompute_mask, work, chunksize=32):
                pass


def _split_dirs(cfg: DictConfig, split_subdir: str) -> tuple[Path, Path]:
    root = Path(cfg.paths.data_root) / split_subdir
    return root / cfg.data.images_subdir, root / cfg.data.labels_subdir


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_root = Path(cfg.paths.output_root)
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    target_size = tuple(cfg.data.target_size)
    cparams = cfg.data.curvature
    curvature_kwargs = dict(
        percentile=float(cparams.percentile),
        num_samples=int(cparams.num_samples),
        smoothing=float(cparams.smoothing),
        target_size=target_size,
        aspect_tol=float(cfg.data.aspect_tol),
        num_workers=int(cfg.data.num_workers),
    )

    # --- index both splits ---
    train_imgs, train_lbls = _split_dirs(cfg, cfg.data.train_subdir)
    valid_imgs, valid_lbls = _split_dirs(cfg, cfg.data.valid_subdir)
    train_idx = index_split(train_imgs, train_lbls)
    valid_idx = index_split(valid_imgs, valid_lbls)
    print(
        f"[index] train pairs={len(train_idx.pairs)} "
        f"(unmatched img={len(train_idx.images_without_label)}, "
        f"lbl={len(train_idx.labels_without_image)}) | "
        f"valid pairs={len(valid_idx.pairs)}"
    )

    # --- curvature (parallel) ---
    print("[curv] computing train curvatures ...")
    train_records, train_excl = compute_frame_curvatures(train_idx.pairs, **curvature_kwargs)
    print(f"[curv] train included={len(train_records)} excluded_aspect={train_excl}")
    print("[curv] computing valid curvatures ...")
    valid_records, valid_excl = compute_frame_curvatures(valid_idx.pairs, **curvature_kwargs)
    print(f"[curv] valid included={len(valid_records)} excluded_aspect={valid_excl}")

    # --- stratified subsets (val reuses train bin edges) ---
    n_bins = int(cfg.subset.n_bins)
    seed = int(cfg.subset.seed)
    train_subset = build_subset(
        train_records, int(cfg.subset.train_target), n_bins, seed, train_excl
    )
    valid_subset = build_subset(
        valid_records,
        int(cfg.subset.val_target),
        n_bins,
        seed,
        valid_excl,
        bin_edges=train_subset.bin_edges,
    )

    print("\n=== TRAIN curvature histogram (natural vs selected) ===")
    print(format_histogram(train_subset))
    print("\n=== VAL curvature histogram (natural vs selected) ===")
    print(format_histogram(valid_subset))

    # --- cache masks ---
    masks_train = out_root / "masks" / "train"
    masks_val = out_root / "masks" / "val"
    _cache_masks(train_subset, masks_train, cfg)
    _cache_masks(valid_subset, masks_val, cfg)

    # --- write manifests ---
    common_meta = {
        "target_size": list(target_size),
        "sky_frac": float(cfg.data.sky_frac),
        "stroke_px": int(cfg.data.stroke_px),
        "aspect_tol": float(cfg.data.aspect_tol),
        "curvature": OmegaConf.to_container(cparams, resolve=True),
    }
    write_manifest(
        train_subset,
        manifests_dir / "train.json",
        {**common_meta, "split": "train", "mask_dir": str(masks_train)},
    )
    write_manifest(
        valid_subset,
        manifests_dir / "val.json",
        {**common_meta, "split": "val", "mask_dir": str(masks_val)},
    )

    # --- histogram report ---
    histogram = {
        "bin_edges": train_subset.bin_edges,
        "train": {
            "natural": train_subset.natural_counts,
            "selected": train_subset.selected_counts,
        },
        "val": {
            "natural": valid_subset.natural_counts,
            "selected": valid_subset.selected_counts,
        },
    }
    (manifests_dir / "histogram.json").write_text(json.dumps(histogram, indent=2))

    print(
        f"\n[done] train={len(train_subset.entries)} val={len(valid_subset.entries)} "
        f"-> manifests in {manifests_dir}, masks in {out_root / 'masks'}"
    )


if __name__ == "__main__":
    main()
