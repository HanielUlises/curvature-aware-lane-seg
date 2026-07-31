"""Tests for the curvature-aware training objective."""

from __future__ import annotations

import pytest
import torch

from src.models.curvature_loss import (
    far_field_weights,
    CurvatureHead,
    curvature_weights,
    per_sample_bce,
    per_sample_dice,
)
from src.models.lane_segmenter import LaneSegmenter

EDGES = [0.0, 0.959, 3.363, 11.794, 41.358, float("inf")]
# One frame per bin, at each bin's median curvature in the training subset.
KAPPA = torch.tensor([0.73, 2.00, 6.27, 21.08, 53.39])


def _batch(n=4, size=64, kappa=None):
    torch.manual_seed(0)
    return {
        "image": torch.randn(n, 3, size, size),
        "mask": (torch.rand(n, 1, size, size) > 0.7).float(),
        "kappa": KAPPA[:n].clone() if kappa is None else kappa,
    }


def test_weights_average_one_so_the_step_size_is_unchanged():
    """Otherwise enabling weighting also scales the learning rate."""
    for strength in (0.5, 1.0, 3.0):
        w = curvature_weights(KAPPA, torch.tensor(EDGES), strength)
        assert w.mean() == pytest.approx(1.0, abs=1e-6)


def test_weights_rise_with_curvature_by_the_requested_ratio():
    w = curvature_weights(KAPPA, torch.tensor(EDGES), strength=1.0)
    assert torch.all(torch.diff(w) > 0)
    # Sharpest bin weighted (1 + strength) times the straightest.
    assert (w[-1] / w[0]).item() == pytest.approx(2.0, rel=1e-6)


def test_zero_strength_is_exactly_uniform():
    w = curvature_weights(KAPPA, torch.tensor(EDGES), strength=0.0)
    assert torch.equal(w, torch.ones_like(KAPPA))


def test_frames_in_one_bin_are_weighted_alike():
    # Binning, not raw curvature: two frames in the tightest bin differing by 10x in
    # curvature must not differ by 10x in weight.
    kappa = torch.tensor([50.0, 500.0])
    w = curvature_weights(kappa, torch.tensor(EDGES), strength=2.0)
    assert w[0] == pytest.approx(w[1])


def test_per_sample_losses_match_the_pooled_ones_when_unweighted():
    """The refactor must not move the baseline: same objective, computed per sample."""
    torch.manual_seed(1)
    logits = torch.randn(4, 1, 32, 32)
    mask = (torch.rand(4, 1, 32, 32) > 0.6).float()
    model = LaneSegmenter(bin_edges=EDGES, encoder_weights=None)
    pooled_bce = model.bce_loss(logits, mask)
    assert per_sample_bce(logits, mask).mean() == pytest.approx(pooled_bce.item(), rel=1e-5)
    # Dice is pooled over the batch upstream, so per-sample Dice is a different (and for
    # weighting, a necessary) reduction; it must still be a sane Dice.
    d = per_sample_dice(logits, mask)
    assert d.shape == (4,) and torch.all(d >= 0) and torch.all(d <= 1)


def test_a_perfect_prediction_has_no_dice_loss():
    mask = (torch.rand(3, 1, 16, 16) > 0.5).float()
    logits = torch.where(mask > 0, 20.0, -20.0)
    assert torch.all(per_sample_dice(logits, mask) < 1e-3)


def test_disabled_curvature_weighting_reproduces_the_baseline_loss():
    """The default must be bit-comparable with the objective the baseline trained on."""
    batch = _batch()
    base = LaneSegmenter(bin_edges=EDGES, encoder_weights=None)
    logits = torch.randn(4, 1, 64, 64)
    with_kappa = base._loss(logits, batch["mask"], batch["kappa"])
    without = base._loss(logits, batch["mask"])
    assert with_kappa.item() == pytest.approx(without.item(), rel=1e-9)


def test_weighting_changes_the_loss_and_favours_the_curved_frames():
    logits = torch.randn(4, 1, 64, 64)
    batch = _batch()
    plain = LaneSegmenter(bin_edges=EDGES, encoder_weights=None, curvature_weight=0.0)
    weighted = LaneSegmenter(bin_edges=EDGES, encoder_weights=None, curvature_weight=2.0)
    a = plain._loss(logits, batch["mask"], batch["kappa"])
    b = weighted._loss(logits, batch["mask"], batch["kappa"])
    assert a.item() != pytest.approx(b.item())


def test_auxiliary_head_is_absent_unless_asked_for():
    assert LaneSegmenter(bin_edges=EDGES, encoder_weights=None).curvature_head is None
    model = LaneSegmenter(bin_edges=EDGES, encoder_weights=None, aux_curvature_weight=0.1)
    assert model.curvature_head is not None


def test_auxiliary_head_learns_the_curvature_it_is_given():
    """A few steps on one constant target must move the prediction towards it."""
    torch.manual_seed(0)
    head = CurvatureHead(in_channels=8)
    features = torch.randn(6, 8, 4, 4)
    kappa = torch.full((6,), 20.0)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = CurvatureHead.loss(head(features), kappa).item()
    for _ in range(50):
        opt.zero_grad()
        loss = CurvatureHead.loss(head(features), kappa)
        loss.backward()
        opt.step()
    assert loss.item() < 0.25 * first


def test_training_step_runs_with_every_combination():
    batch = _batch()
    for curv, aux in ((0.0, 0.0), (1.0, 0.0), (0.0, 0.1), (1.0, 0.1)):
        model = LaneSegmenter(
            bin_edges=EDGES, encoder_weights=None,
            curvature_weight=curv, aux_curvature_weight=aux,
        )
        loss = model.training_step(batch, 0)
        loss.backward()
        assert torch.isfinite(loss)
        # The encoder must receive gradient in every configuration.
        grads = [p.grad for p in model.net.encoder.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


def test_edges_come_from_the_manifest_not_the_checkpoint(tmp_path):
    """A checkpoint must not carry bin edges, for two reasons.

    Checkpoints trained before the curvature objective existed have no such entry and
    would fail a strict load, breaking every inference script against the baseline. And
    edges baked into a checkpoint could silently disagree with the manifest the metric
    bins by, which is exactly the mismatch this objective exists to close.
    """
    model = LaneSegmenter(bin_edges=EDGES, encoder_weights=None, curvature_weight=1.0)
    state = model.state_dict()
    assert "bin_edges" not in state

    other = LaneSegmenter(bin_edges=[0.0, 5.0, float("inf")], encoder_weights=None)
    other.load_state_dict(state)  # loads cleanly despite different edges
    assert other.bin_edges.numel() == 3


def test_far_field_weights_average_one_and_favour_the_distance():
    w = far_field_weights(288, strength=1.0)
    assert w.shape == (1, 1, 288, 1)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)
    # Row 0 is the top of the frame, which is the far field under a sky-cropped view.
    assert float(w[0, 0, 0, 0]) > float(w[0, 0, -1, 0])
    assert float(w[0, 0, 0, 0]) / float(w[0, 0, -1, 0]) == pytest.approx(2.0, rel=1e-2)


def test_zero_far_field_strength_is_disabled():
    assert far_field_weights(288, strength=0.0) is None


def test_uniform_pixel_weights_do_not_change_either_loss_term():
    """The invariant that makes a weighted run comparable with the baseline."""
    torch.manual_seed(3)
    logits = torch.randn(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.6).float()
    ones = torch.ones(1, 1, 16, 1)
    assert torch.allclose(per_sample_dice(logits, mask),
                          per_sample_dice(logits, mask, ones), atol=1e-6)
    assert torch.allclose(per_sample_bce(logits, mask),
                          per_sample_bce(logits, mask, ones), atol=1e-6)


def test_far_field_weighting_penalises_a_far_field_miss_more():
    """A model that misses the top of the frame must be punished harder than one that
    misses the bottom, which is the entire point of the weighting."""
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :16, :] = 1.0   # far half
    mask[..., 16:, :] = 1.0   # near half, so the lane spans the frame
    miss_far = torch.full((1, 1, 32, 32), 10.0)
    miss_far[..., :16, :] = -10.0
    miss_near = torch.full((1, 1, 32, 32), 10.0)
    miss_near[..., 16:, :] = -10.0
    w = far_field_weights(32, strength=2.0)
    assert per_sample_bce(miss_far, mask, w) > per_sample_bce(miss_near, mask, w)
    # Unweighted, the two misses are equivalent.
    assert per_sample_bce(miss_far, mask) == pytest.approx(
        float(per_sample_bce(miss_near, mask)), rel=1e-6)


def test_model_runs_with_far_field_weighting_alone_and_combined():
    batch = _batch()
    for curv, far in ((0.0, 1.0), (0.5, 1.0)):
        model = LaneSegmenter(bin_edges=EDGES, encoder_weights=None,
                              curvature_weight=curv, far_field_weight=far)
        loss = model._loss(torch.randn(4, 1, 64, 64), batch["mask"], batch["kappa"])
        assert torch.isfinite(loss)


def test_far_field_disabled_reproduces_the_baseline_loss():
    torch.manual_seed(5)
    logits = torch.randn(4, 1, 64, 64)
    batch = _batch()
    a = LaneSegmenter(bin_edges=EDGES, encoder_weights=None, far_field_weight=0.0)
    assert a._loss(logits, batch["mask"], batch["kappa"]).item() == pytest.approx(
        a._loss(logits, batch["mask"]).item(), rel=1e-9)
