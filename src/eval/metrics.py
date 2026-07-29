"""Segmentation metrics, stratified by curvature bin.

Global IoU is the wrong headline number for this project: the near-straight
frames that dominate the data also dominate a global average, so a model can
score well while failing exactly on the high-curvature frames the system exists
for. Every metric here is therefore reported **per curvature bin** (the same bins
:mod:`src.data.subset` stratified on) as well as globally, so the tail is visible.

The accumulator sums intersection/union counts across an epoch rather than
averaging per-batch IoU, which keeps the result independent of batch composition
and correct when a bin is split across batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def assign_bins(kappas: np.ndarray, edges: list[float]) -> np.ndarray:
    """Assign curvatures to bin indices in ``[0, K-1]``.

    Mirrors :func:`src.data.subset.assign_bins` so evaluation buckets match the
    bins the subset was stratified on.
    """
    idx = np.digitize(kappas, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


@dataclass
class SegScores:
    """IoU and Dice for one group of frames (a bin, or all frames)."""

    iou: float
    dice: float
    count: int


@dataclass
class StratifiedSegMetric:
    """Accumulate intersection/union per curvature bin over an epoch.

    Foreground (lane) class only — the background dominates and its IoU is
    uninformative. Call :meth:`update` per validation batch, :meth:`compute` at
    epoch end, then :meth:`reset`.

    Args:
        bin_edges: ``K + 1`` edges (the ``bin_edges`` stored in the manifest).
        threshold: Probability cutoff for the positive class.
    """

    bin_edges: list[float]
    threshold: float = 0.5
    _inter: np.ndarray = field(init=False)
    _union: np.ndarray = field(init=False)
    _dice_num: np.ndarray = field(init=False)
    _dice_den: np.ndarray = field(init=False)
    _count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._n_bins = len(self.bin_edges) - 1
        self.reset()

    def reset(self) -> None:
        z = lambda: np.zeros(self._n_bins, dtype=np.float64)  # noqa: E731
        self._inter, self._union = z(), z()
        self._dice_num, self._dice_den = z(), z()
        self._count = np.zeros(self._n_bins, dtype=np.int64)

    @torch.no_grad()
    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        kappas: torch.Tensor,
    ) -> None:
        """Fold one batch into the running totals.

        Args:
            probs: Predicted foreground probabilities, ``(B, 1, H, W)``.
            targets: Binary ground-truth masks, ``(B, 1, H, W)``.
            kappas: Per-frame curvature, ``(B,)``.
        """
        preds = (probs >= self.threshold).float()
        tgt = (targets > 0.5).float()
        dims = (1, 2, 3)  # reduce over channel + spatial, keep batch
        inter = (preds * tgt).sum(dims)
        union = ((preds + tgt) >= 1).float().sum(dims)
        pred_sum = preds.sum(dims)
        tgt_sum = tgt.sum(dims)

        bins = assign_bins(kappas.detach().cpu().numpy(), self.bin_edges)
        inter_np = inter.detach().cpu().numpy()
        union_np = union.detach().cpu().numpy()
        dnum_np = (2.0 * inter).detach().cpu().numpy()
        dden_np = (pred_sum + tgt_sum).detach().cpu().numpy()

        for b, i, u, dn, dd in zip(bins, inter_np, union_np, dnum_np, dden_np):
            self._inter[b] += i
            self._union[b] += u
            self._dice_num[b] += dn
            self._dice_den[b] += dd
            self._count[b] += 1

    def compute(self, eps: float = 1e-7) -> tuple[SegScores, list[SegScores]]:
        """Return ``(overall, per_bin)`` scores from the accumulated totals.

        Empty bins yield ``iou = dice = nan`` so they are not silently counted as
        perfect. The overall score pools counts across all bins.
        """
        per_bin: list[SegScores] = []
        for b in range(self._n_bins):
            if self._count[b] == 0:
                per_bin.append(SegScores(float("nan"), float("nan"), 0))
                continue
            iou = self._inter[b] / (self._union[b] + eps)
            dice = self._dice_num[b] / (self._dice_den[b] + eps)
            per_bin.append(SegScores(float(iou), float(dice), int(self._count[b])))

        total_inter = self._inter.sum()
        total_union = self._union.sum()
        overall = SegScores(
            iou=float(total_inter / (total_union + eps)),
            dice=float(self._dice_num.sum() / (self._dice_den.sum() + eps)),
            count=int(self._count.sum()),
        )
        return overall, per_bin
