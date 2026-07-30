"""LightningDataModule wiring the CurveLanes subset to training.

Masks are precomputed and cached at the model input resolution by
``scripts/prepare_data.py``; the dataset loads the native image, applies the same
aspect-preserving sky-crop + isotropic resize used for the cached mask (so the
two stay pixel-aligned), then runs the augmentation pipeline jointly on both.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.subset import ManifestEntry, read_manifest
from src.data.transforms import (
    DEFAULT_SKY_FRAC,
    DEFAULT_TARGET_SIZE,
    build_eval_transform,
    build_train_transform,
    preprocess_geometry,
)


class LaneDataset(Dataset):
    """Image + cached-mask pairs for lane segmentation.

    Args:
        entries: Manifest entries (frames selected into the subset).
        masks_dir: Directory holding ``<stem>.png`` masks at target resolution.
        transform: albumentations pipeline applied jointly to image and mask.
        target_size: Model input ``(width, height)``.
        sky_frac: Sky-crop fraction (must match the mask cache).
    """

    def __init__(
        self,
        entries: list[ManifestEntry],
        masks_dir: Path,
        transform,
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
        sky_frac: float = DEFAULT_SKY_FRAC,
    ) -> None:
        self.entries = entries
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.target_size = target_size
        self.sky_frac = sky_frac

    def __len__(self) -> int:
        return len(self.entries)

    def _mask_path(self, entry: ManifestEntry) -> Path:
        return self.masks_dir / f"{entry.image_path.stem}.png"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]

        image_bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image {entry.image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Match the mask's baked-in geometry (sky-crop + isotropic resize).
        image = preprocess_geometry(image, self.target_size, self.sky_frac, cv2.INTER_AREA)

        mask_path = self._mask_path(entry)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read cached mask {mask_path}")

        out = self.transform(image=image, mask=mask)
        image_t = out["image"]
        # Binarize + add channel dim -> (1, H, W) float for Dice/BCE.
        mask_t = (out["mask"] > 0).float().unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "kappa": torch.tensor(entry.kappa)}


def _seed_worker(worker_id: int) -> None:
    """Deterministic per-worker seeding for reproducible augmentation."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class LaneDataModule(pl.LightningDataModule):
    """Wire train/val manifests + cached masks into dataloaders.

    Args:
        train_manifest: Path to the training subset manifest JSON.
        val_manifest: Path to the validation subset manifest JSON.
        train_masks_dir: Cached-mask directory for the train split.
        val_masks_dir: Cached-mask directory for the val split.
        batch_size: Per-step batch size (default 8, the VRAM budget).
        num_workers: Dataloader worker processes.
        target_size: Model input ``(width, height)``.
        sky_frac: Sky-crop fraction (must match the mask cache).
        seed: Base seed for worker reproducibility.
    """

    def __init__(
        self,
        train_manifest: Path,
        val_manifest: Path,
        train_masks_dir: Path,
        val_masks_dir: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
        sky_frac: float = DEFAULT_SKY_FRAC,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_manifest = Path(train_manifest)
        self.val_manifest = Path(val_manifest)
        self.train_masks_dir = Path(train_masks_dir)
        self.val_masks_dir = Path(val_masks_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.sky_frac = sky_frac
        self.seed = seed
        self._train: LaneDataset | None = None
        self._val: LaneDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train_entries, _ = read_manifest(self.train_manifest)
        val_entries, _ = read_manifest(self.val_manifest)
        self._train = LaneDataset(
            train_entries,
            self.train_masks_dir,
            build_train_transform(),
            self.target_size,
            self.sky_frac,
        )
        self._val = LaneDataset(
            val_entries,
            self.val_masks_dir,
            build_eval_transform(),
            self.target_size,
            self.sky_frac,
        )

    def _loader(self, dataset: LaneDataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None, "call setup() first"
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None, "call setup() first"
        return self._loader(self._val, shuffle=False)
