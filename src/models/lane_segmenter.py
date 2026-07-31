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
from src.models.curvature_loss import (
    CurvatureHead,
    curvature_weights,
    far_field_weights,
    per_sample_bce,
    per_sample_dice,
)


class LaneSegmenter(pl.LightningModule):
    """U-Net lane segmenter with curvature-stratified validation.

    Args:
        bin_edges: Curvature bin edges (from the manifest) for stratified metrics.
        encoder_name: ``segmentation_models_pytorch`` encoder identifier.
        encoder_weights: Pretrained-weights tag, or ``None`` for random init.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        dice_weight: Weight on the Dice term (BCE term is ``1 - dice_weight``).
        curvature_weight: Strength of per-sample curvature weighting. ``0`` reproduces
            the baseline objective exactly; ``1`` weights the sharpest curvature bin
            twice the straightest.
        aux_curvature_weight: Weight on the auxiliary curvature-regression head. ``0``
            leaves the head out of the model entirely.
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
        curvature_weight: float = 0.0,
        aux_curvature_weight: float = 0.0,
        far_field_weight: float = 0.0,
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
        self.curvature_weight = curvature_weight
        self.aux_curvature_weight = aux_curvature_weight
        self.far_field_weight = far_field_weight
        self._row_weights: torch.Tensor | None = None
        # A buffer so it follows the model onto the device, but deliberately not
        # persistent: every load path already passes the manifest's edges explicitly, and
        # putting them in the state dict would make checkpoints trained before this
        # existed fail to load, and would let a checkpoint's edges silently disagree with
        # the manifest the metric is using.
        self.register_buffer(
            "bin_edges", torch.tensor(bin_edges, dtype=torch.float32), persistent=False
        )
        self.curvature_head = (
            CurvatureHead(self.net.encoder.out_channels[-1])
            if aux_curvature_weight > 0
            else None
        )
        self.val_metric = StratifiedSegMetric(bin_edges)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)

    def _loss(
        self, logits: torch.Tensor, mask: torch.Tensor, kappa: torch.Tensor | None = None
    ) -> torch.Tensor:
        rows = None
        if self.far_field_weight > 0:
            # Cached per height: the frame size does not change within a run.
            if self._row_weights is None or self._row_weights.shape[2] != logits.shape[2]:
                self._row_weights = far_field_weights(
                    logits.shape[2], self.far_field_weight, dtype=torch.float32
                )
            rows = self._row_weights.to(logits.device, logits.dtype)

        if (self.curvature_weight > 0 and kappa is not None) or rows is not None:
            # Per-sample so the weight can be applied before the batch is pooled.
            weights = (
                curvature_weights(kappa, self.bin_edges, self.curvature_weight)
                if (self.curvature_weight > 0 and kappa is not None)
                else torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
            )
            per_sample = self.dice_weight * per_sample_dice(logits, mask, rows) + (
                1.0 - self.dice_weight
            ) * per_sample_bce(logits, mask, rows)
            base = (weights * per_sample).mean()
        else:
            base = self.dice_weight * self.dice_loss(logits, mask) + (
                1.0 - self.dice_weight
            ) * self.bce_loss(logits, mask)
        if self.lovasz_weight > 0:
            return (1.0 - self.lovasz_weight) * base + self.lovasz_weight * self.lovasz_loss(
                logits, mask
            )
        return base

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        kappa = batch["kappa"]
        if self.curvature_head is None:
            logits = self(batch["image"])
        else:
            # Run the segmenter in two halves so the encoder's bottleneck is available
            # to the auxiliary head without a second forward pass.
            features = self.net.encoder(batch["image"])
            logits = self.net.segmentation_head(self.net.decoder(features))
        loss = self._loss(logits, batch["mask"], kappa)
        self.log("train/seg_loss", loss, on_step=False, on_epoch=True)
        if self.curvature_head is not None:
            aux = CurvatureHead.loss(self.curvature_head(features[-1]), kappa)
            self.log("train/aux_curvature", aux, on_step=False, on_epoch=True)
            loss = loss + self.aux_curvature_weight * aux
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self._loss(logits, batch["mask"], batch["kappa"])
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
