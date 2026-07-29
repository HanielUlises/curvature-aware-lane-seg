"""Baseline lane-segmentation model: U-Net + ResNet-18, Dice + BCE.

This is the reference model every curvature-aware experiment is measured against.
It is deliberately plain — an ImageNet-pretrained ResNet-18 encoder (fits the 6 GB
budget at 512x288) and a standard Dice + BCE objective. The only project-specific
piece is validation: metrics are reported per curvature bin via
:class:`src.eval.metrics.StratifiedSegMetric`, so the high-curvature tail the
project targets is visible rather than averaged away.
"""

from __future__ import annotations

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn

from src.eval.metrics import StratifiedSegMetric


class LaneSegmenter(pl.LightningModule):
    """U-Net lane segmenter with curvature-stratified validation.

    Args:
        bin_edges: Curvature bin edges (from the manifest) for stratified metrics.
        encoder_name: ``segmentation_models_pytorch`` encoder identifier.
        encoder_weights: Pretrained-weights tag, or ``None`` for random init.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        dice_weight: Weight on the Dice term (BCE term is ``1 - dice_weight``).
    """

    def __init__(
        self,
        bin_edges: list[float],
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dice_weight: float = 0.5,
        lovasz_weight: float = 0.0,
    ) -> None:
        super().__init__()
        # bin_edges is data, not a hyperparameter to reconstruct the model from.
        self.save_hyperparameters(ignore=["bin_edges"])
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Lovász-hinge is a direct surrogate for IoU (vs Dice's soft overlap); adding
        # it sharpens thin-structure boundaries where IoU is most sensitive.
        self.lovasz_loss = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.val_metric = StratifiedSegMetric(bin_edges)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)

    def _loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base = self.dice_weight * self.dice_loss(logits, mask) + (
            1.0 - self.dice_weight
        ) * self.bce_loss(logits, mask)
        if self.lovasz_weight > 0:
            return (1.0 - self.lovasz_weight) * base + self.lovasz_weight * self.lovasz_loss(
                logits, mask
            )
        return base

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self._loss(logits, batch["mask"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self._loss(logits, batch["mask"])
        self.val_metric.update(torch.sigmoid(logits), batch["mask"], batch["kappa"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        overall, per_bin = self.val_metric.compute()
        self.log("val/iou", overall.iou, prog_bar=True)
        self.log("val/dice", overall.dice)
        # Per-bin IoU is the headline: watch the tail bins, not the average.
        for b, scores in enumerate(per_bin):
            if scores.count == 0:
                continue
            self.log(f"val/iou_bin{b}", scores.iou)
            self.log(f"val/dice_bin{b}", scores.dice)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs if self.trainer else 50
        )
        return {"optimizer": opt, "lr_scheduler": sched}
