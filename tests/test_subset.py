"""Tests for curvature-stratified subset construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.subset import (
    FrameRecord,
    allocate_flattened,
    assign_bins,
    build_subset,
    compute_bin_edges,
)


def _record(kappa: float, name: str) -> FrameRecord:
    return FrameRecord(
        image_path=Path(f"{name}.jpg"),
        label_path=Path(f"{name}.lines.json"),
        width=2560,
        height=1440,
        num_lanes=3,
        kappa=kappa,
    )


def test_bin_edges_are_monotonic_with_open_top() -> None:
    kappas = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 70.0])
    edges = compute_bin_edges(kappas, n_bins=5)
    assert len(edges) == 6
    assert edges[0] == 0.0
    assert edges[-1] == float("inf")
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))


def test_assign_bins_covers_full_range() -> None:
    edges = [0.0, 1.0, 4.0, 16.0, float("inf")]
    kappas = np.array([0.1, 0.9, 1.0, 3.9, 4.0, 100.0])
    bins = assign_bins(kappas, edges)
    assert bins.tolist() == [0, 0, 1, 1, 2, 3]
    assert bins.max() <= len(edges) - 2


def test_allocate_equal_when_supply_ample() -> None:
    take = allocate_flattened([1000, 1000, 1000, 1000], n_target=800)
    assert take == [200, 200, 200, 200]
    assert sum(take) == 800


def test_allocate_caps_and_redistributes_shortfall() -> None:
    # Tight bin (3) only has 50; its share should spill to bins with capacity.
    take = allocate_flattened([40000, 18000, 4000, 50], n_target=8000)
    assert sum(take) == 8000
    assert take[3] == 50  # capped at availability
    assert take[0] > 2000  # surplus redistributed to the abundant low-kappa bin


def test_allocate_limited_by_total_supply() -> None:
    take = allocate_flattened([100, 100, 30], n_target=8000)
    assert take == [100, 100, 30]
    assert sum(take) == 230


def test_flattening_boosts_rare_tail_relative_to_natural() -> None:
    # Natural distribution: heavy low-kappa, sparse high-kappa.
    records = (
        [_record(0.6, f"lo{i}") for i in range(4000)]
        + [_record(6.0, f"md{i}") for i in range(800)]
        + [_record(40.0, f"hi{i}") for i in range(120)]
    )
    subset = build_subset(records, n_target=900, n_bins=3, seed=0)
    # In the natural set the tail is 120/4920 ~ 2.4%; after flattening it should
    # be a far larger share of the selected set.
    tail_selected = subset.selected_counts[-1]
    tail_frac = tail_selected / sum(subset.selected_counts)
    assert tail_frac > 0.10
    assert len(subset.entries) == sum(subset.selected_counts)


def test_build_subset_is_deterministic() -> None:
    records = [_record(float(i % 50) + 0.5, f"f{i}") for i in range(2000)]
    a = build_subset(records, n_target=300, n_bins=4, seed=7)
    b = build_subset(records, n_target=300, n_bins=4, seed=7)
    names_a = [e.image_path.name for e in a.entries]
    names_b = [e.image_path.name for e in b.entries]
    assert names_a == names_b


def test_different_seed_changes_selection() -> None:
    records = [_record(float(i % 50) + 0.5, f"f{i}") for i in range(2000)]
    a = build_subset(records, n_target=300, n_bins=4, seed=1)
    b = build_subset(records, n_target=300, n_bins=4, seed=2)
    assert {e.image_path.name for e in a.entries} != {e.image_path.name for e in b.entries}


def test_empty_records_yields_empty_subset() -> None:
    subset = build_subset([], n_target=100, n_bins=5, seed=0)
    assert subset.entries == []
    assert sum(subset.selected_counts) == 0
