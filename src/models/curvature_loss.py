"""Curvature-aware training objectives.

The project stratifies its *data* and its *metrics* by curvature, but until now the
objective did not see curvature at all: the loss took logits and a mask, and the only
consumer of a frame's ``kappa`` was the validation metric. That gap is why the
control-relevant finding, detection rate falling from 82% on near-straight frames to 61%
on the tightest ones, went unaddressed by the model.

Two mechanisms are provided, both off by default so the baseline stays exactly what it
was:

- **Per-sample curvature weighting.** Curvature-stratified sampling equalizes how *often*
  a curved frame is seen, not how much it contributes once seen. Curved frames are the
  harder ones, so an equal-frequency diet still leaves them under-served relative to
  their difficulty. Weighting the per-frame loss by curvature closes that.
- **An auxiliary curvature head.** A small regression head on the encoder bottleneck
  predicting the frame's curvature. It does not change what the segmenter outputs; it
  constrains the representation to carry curvature, which is the property the project's
  name claims and the segmentation objective alone has no reason to produce.

Weights are normalized to mean one within each batch. Without that, turning weighting on
also scales the effective learning rate, and any change in the result could be either the
weighting or the step size.
"""

from __future__ import annotations

import torch
from torch import nn

def curvature_weights(
    kappa: torch.Tensor,
    bin_edges: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Per-sample loss weights rising with curvature bin, mean-normalized per batch.

    The weight is ``1 + strength * b / (K - 1)`` for a frame in bin ``b`` of ``K``, so
    the sharpest bin is weighted ``1 + strength`` times the straightest and the bins
    between are spaced evenly.

    Binning rather than raw curvature is deliberate. On the stratified training subset
    the per-bin median curvature runs 0.73, 2.00, 6.27, 21.08, 53.39: a seventy-fold
    range with a long tail. Any weight linear in curvature is then effectively a step
    function that hands most of the batch gradient to whichever frame happens to be
    sharpest, and one proportional to it is worse. The bins are also the partition the
    stratified metric reports, so the objective and the measurement now refer to the
    same thing.

    Args:
        kappa: Per-sample width-normalized curvature, shape ``(B,)``.
        bin_edges: The manifest's bin edges, shape ``(K + 1,)``.
        strength: 0 disables weighting entirely and returns ones.

    Returns:
        Weights of shape ``(B,)`` averaging one, so that enabling weighting does not
        also change the effective learning rate.
    """
    if strength <= 0.0:
        return torch.ones_like(kappa)
    interior = bin_edges[1:-1].to(kappa.device)
    index = torch.bucketize(kappa.abs(), interior).to(kappa.dtype)
    n_bins = interior.numel() + 1
    raw = 1.0 + strength * index / max(n_bins - 1, 1)
    return raw / raw.mean().clamp(min=1e-9)


def far_field_weights(height: int, strength: float, device=None, dtype=None):
    """Per-row pixel weights rising towards the top of the frame.

    A separate idea from curvature weighting, and a more targeted one. The failure this
    project measures is that the ego lane cannot be recovered on curves, and the reason
    IoU misses it is that pixels are dominated by the wide, unambiguous markings near the
    bumper while the controller depends on geometry further ahead. Weighting rows towards
    the vanishing point puts the loss where the recoverable geometry is.

    Unlike curvature weighting this says nothing about the dataset's mix of curves, so it
    should not specialize the model to one curvature distribution. The first
    curvature-weighted run gained 9.7 points of in-domain detection and lost 7.1 points
    on an unseen camera, which is what motivated trying a weighting that is geometric
    rather than distributional.

    Returns a ``(1, 1, H, 1)`` tensor averaging one, broadcastable over a mask.
    """
    if strength <= 0.0:
        return None
    rows = torch.arange(height, device=device, dtype=dtype or torch.float32)
    # 1 at the bottom row, 1 + strength at the top.
    w = 1.0 + strength * (1.0 - rows / max(height - 1, 1))
    w = w / w.mean()
    return w.view(1, 1, height, 1)


def per_sample_dice(logits: torch.Tensor, mask: torch.Tensor,
                    pixel_weights: torch.Tensor | None = None, eps: float = 1e-6):
    """Soft Dice loss per sample rather than pooled over the batch.

    Pooling over the batch, as the stock loss does, makes a per-sample weight impossible
    to apply: the frames are already summed together before the loss exists.
    """
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    if pixel_weights is None:
        intersection = (probs * mask).sum(dims)
        cardinality = probs.sum(dims) + mask.sum(dims)
    else:
        # Weighted soft Dice: the weight enters both the overlap and the cardinality, so
        # a uniform weight leaves the value unchanged.
        intersection = (pixel_weights * probs * mask).sum(dims)
        cardinality = (pixel_weights * probs).sum(dims) + (pixel_weights * mask).sum(dims)
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


def per_sample_bce(logits: torch.Tensor, mask: torch.Tensor,
                   pixel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Binary cross-entropy averaged within each sample, optionally row-weighted."""
    elementwise = nn.functional.binary_cross_entropy_with_logits(
        logits, mask, reduction="none"
    )
    if pixel_weights is not None:
        elementwise = elementwise * pixel_weights
    return elementwise.flatten(1).mean(1)


class CurvatureHead(nn.Module):
    """Regresses a frame's curvature from pooled encoder features.

    Predicts ``log1p(|kappa|)`` rather than ``|kappa|``: curvature after stratification
    still spans orders of magnitude, and regressing it directly makes the loss almost
    entirely about the few sharpest frames.
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(features)).squeeze(1)

    @staticmethod
    def target(kappa: torch.Tensor) -> torch.Tensor:
        return torch.log1p(kappa.abs())

    @staticmethod
    def loss(predicted: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(predicted, CurvatureHead.target(kappa))
