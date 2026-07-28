"""Smoke test for the LightningDataModule against a synthetic manifest."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.subset import ManifestEntry, Subset, write_manifest
from src.data.datamodule import LaneDataModule


def _make_split(root: Path, name: str, n: int, native=(128, 72), target=(64, 36)) -> Path:
    images = root / name / "images"
    masks = root / name / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    entries = []
    for i in range(n):
        stem = f"{name}{i}"
        img = (np.random.rand(native[1], native[0], 3) * 255).astype(np.uint8)
        cv2.imwrite(str(images / f"{stem}.jpg"), img)
        mask = np.zeros((target[1], target[0]), dtype=np.uint8)
        cv2.line(mask, (2, target[1] - 2), (target[0] - 2, 2), 255, 2)
        cv2.imwrite(str(masks / f"{stem}.png"), mask)
        entries.append(
            ManifestEntry(
                image_path=images / f"{stem}.jpg",
                label_path=images / f"{stem}.lines.json",
                width=native[0],
                height=native[1],
                num_lanes=2,
                kappa=float(i),
                bin_index=i % 3,
            )
        )
    subset = Subset(entries, [0.0, 1.0, 2.0, float("inf")], [n, n, n], [n, n, n], 0, 0)
    manifest = root / f"{name}.json"
    write_manifest(subset, manifest)
    return manifest, masks


def test_datamodule_yields_batch_of_expected_shape(tmp_path: Path) -> None:
    target = (64, 36)
    train_manifest, train_masks = _make_split(tmp_path, "train", 6, target=target)
    val_manifest, val_masks = _make_split(tmp_path, "val", 4, target=target)

    dm = LaneDataModule(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        train_masks_dir=train_masks,
        val_masks_dir=val_masks,
        batch_size=2,
        num_workers=0,
        target_size=target,
        sky_frac=0.0,
        seed=0,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (2, 3, 36, 64)
    assert batch["mask"].shape == (2, 1, 36, 64)
    assert batch["mask"].max() <= 1.0 and batch["mask"].min() >= 0.0

    val_batch = next(iter(dm.val_dataloader()))
    assert val_batch["image"].shape == (2, 3, 36, 64)
