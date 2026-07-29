"""Train the baseline lane segmenter.

Hydra entry point. Reads the stratified manifests + cached masks produced by
``scripts.prepare_data``, trains a U-Net/ResNet-18 baseline, and reports
validation IoU/Dice **per curvature bin** so the high-curvature tail is visible.

    python -m scripts.train
    python -m scripts.train train.max_epochs=10 model.encoder_name=resnet34

Checkpoints and logs land under the Hydra run directory (``outputs/<timestamp>``).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.data.datamodule import LaneDataModule
from src.data.subset import read_manifest
from src.models.lane_segmenter import LaneSegmenter


def _manifest_paths(cfg: DictConfig) -> tuple[Path, Path]:
    manifests = Path(cfg.paths.output_root) / "manifests"
    return manifests / "train.json", manifests / "val.json"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    train_manifest, val_manifest = _manifest_paths(cfg)
    for path in (train_manifest, val_manifest):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing manifest {path}. Run `python -m scripts.prepare_data` first."
            )

    # Mask dirs + bin edges come from the manifests, so training uses exactly the
    # bins the subset was stratified on.
    _, train_meta = read_manifest(train_manifest)
    val_entries, val_meta = read_manifest(val_manifest)
    bin_edges = val_meta["bin_edges"]

    datamodule = LaneDataModule(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        train_masks_dir=Path(train_meta["mask_dir"]),
        val_masks_dir=Path(val_meta["mask_dir"]),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        target_size=tuple(cfg.data.target_size),
        sky_frac=cfg.data.sky_frac,
        seed=cfg.seed,
    )

    model = LaneSegmenter(
        bin_edges=bin_edges,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        dice_weight=cfg.model.dice_weight,
    )

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="{epoch}-{val/iou:.4f}",
        monitor=cfg.train.monitor,
        mode=cfg.train.monitor_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=str(run_dir),
        deterministic=True,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
    )

    print(OmegaConf.to_yaml(cfg))
    print(f"train frames: unknown (manifest), val frames: {len(val_entries)}")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
