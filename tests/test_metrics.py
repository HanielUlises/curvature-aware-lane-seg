"""Tests for curvature-stratified segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch

from src.data.subset import assign_bins as subset_assign_bins
from src.eval.metrics import StratifiedSegMetric, assign_bins


def test_assign_bins_matches_subset() -> None:
    # The eval bins must be identical to the ones the subset stratified on.
    edges = [0.0, 1.0, 3.0, 10.0, float("inf")]
    kappas = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 9.9, 50.0])
    np.testing.assert_array_equal(
        assign_bins(kappas, edges), subset_assign_bins(kappas, edges)
    )


def _mask(rows: int, cols: int, on: bool) -> torch.Tensor:
    v = 1.0 if on else 0.0
    return torch.full((1, 1, rows, cols), v)


def test_perfect_prediction_scores_one() -> None:
    edges = [0.0, 1.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    target = (torch.arange(16).reshape(1, 1, 4, 4) % 2).float()
    metric.update(target, target, torch.tensor([0.5]))
    overall, per_bin = metric.compute()
    assert abs(overall.iou - 1.0) < 1e-5
    assert abs(overall.dice - 1.0) < 1e-5
    assert per_bin[0].count == 1
    assert per_bin[1].count == 0
    assert np.isnan(per_bin[1].iou)


def test_disjoint_prediction_scores_zero() -> None:
    edges = [0.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    target = torch.zeros(1, 1, 2, 2)
    target[..., 0, :] = 1.0  # top row foreground
    pred = torch.zeros(1, 1, 2, 2)
    pred[..., 1, :] = 1.0  # bottom row foreground, no overlap
    metric.update(pred, target, torch.tensor([0.1]))
    overall, _ = metric.compute()
    assert overall.iou == 0.0
    assert overall.dice == 0.0


def test_half_overlap_iou() -> None:
    # pred = 2 px, target = 2 px, intersection = 1 px -> IoU = 1/3, Dice = 2/4.
    edges = [0.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
    pred = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
    metric.update(pred, target, torch.tensor([0.2]))
    overall, _ = metric.compute()
    assert abs(overall.iou - 1.0 / 3.0) < 1e-5
    assert abs(overall.dice - 0.5) < 1e-5


def test_stratification_routes_to_correct_bins() -> None:
    edges = [0.0, 1.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    target = torch.ones(2, 1, 2, 2)
    # First frame low-kappa (bin 0), perfect; second high-kappa (bin 1), all wrong.
    pred = torch.stack([torch.ones(1, 2, 2), torch.zeros(1, 2, 2)])
    metric.update(pred, target, torch.tensor([0.2, 5.0]))
    _, per_bin = metric.compute()
    assert abs(per_bin[0].iou - 1.0) < 1e-5
    assert per_bin[1].iou == 0.0
    assert per_bin[0].count == 1
    assert per_bin[1].count == 1


def test_accumulates_across_batches() -> None:
    edges = [0.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    target = torch.ones(1, 1, 2, 2)
    metric.update(torch.ones(1, 1, 2, 2), target, torch.tensor([0.1]))  # perfect
    metric.update(torch.zeros(1, 1, 2, 2), target, torch.tensor([0.1]))  # empty
    overall, _ = metric.compute()
    # Pooled: intersection 4, union 8 -> IoU 0.5; dice 2*4 / (4+8) = 2/3.
    assert abs(overall.iou - 0.5) < 1e-5
    assert abs(overall.dice - 2.0 / 3.0) < 1e-5
    assert overall.count == 2


def test_reset_clears_state() -> None:
    edges = [0.0, float("inf")]
    metric = StratifiedSegMetric(edges)
    metric.update(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), torch.tensor([0.1]))
    metric.reset()
    overall, _ = metric.compute()
    assert overall.count == 0
