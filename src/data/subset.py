"""Build a curvature-stratified training subset from CurveLanes.

The natural CurveLanes distribution is dominated by near-straight roads. Training
on it as-is starves the high-curvature regime the project targets, so this module
computes a resolution-invariant curvature per frame (:mod:`src.geometry.curvature`),
bins frames over ``|kappa|`` with **fixed log-spaced edges**, and samples a
**flattened** subset — roughly equal counts per bin, capped by availability with
the shortfall redistributed. Fixed edges (rather than quantile edges) are what
make flattening meaningful: they leave the tight-curve bins genuinely rare, so
equal allocation oversamples them relative to their natural frequency.

Odd-aspect frames (1570x660) are excluded here — the same exclusion enforced by
:func:`src.data.rasterize.assert_target_aspect` — because they cannot be resized
to the 16:9 target without distorting curvature.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from src.data.curvelanes import build_frame
from src.data.rasterize import DEFAULT_TARGET_SIZE, is_target_aspect
from src.geometry.curvature import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PERCENTILE,
    DEFAULT_SMOOTHING,
    frame_curvature,
)


@dataclass(frozen=True)
class FrameRecord:
    """Per-frame curvature record (a subset candidate)."""

    image_path: Path
    label_path: Path
    width: int
    height: int
    num_lanes: int
    kappa: float


@dataclass(frozen=True)
class ManifestEntry:
    """A selected frame, tagged with its curvature bin."""

    image_path: Path
    label_path: Path
    width: int
    height: int
    num_lanes: int
    kappa: float
    bin_index: int


@dataclass(frozen=True)
class Subset:
    """Result of stratified subset construction.

    Attributes:
        entries: Selected frames with bin tags.
        bin_edges: ``K + 1`` edges (last is ``inf``) defining the ``K`` bins.
        natural_counts: Available frame count per bin (pre-sampling).
        selected_counts: Selected frame count per bin.
        seed: RNG seed used for sampling.
        excluded_aspect: Number of frames dropped for odd aspect ratio.
    """

    entries: list[ManifestEntry]
    bin_edges: list[float]
    natural_counts: list[int]
    selected_counts: list[int]
    seed: int
    excluded_aspect: int


# --- curvature computation (parallel) --------------------------------------

# Worker configuration is passed per-item to keep the worker picklable.
_WorkerArgs = tuple[str, str, float, int, float, tuple[int, int], float]


def _curvature_worker(args: _WorkerArgs) -> FrameRecord | None:
    image_path, label_path, percentile, num_samples, smoothing, target_size, tol = args
    frame = build_frame(Path(image_path), Path(label_path))
    if not is_target_aspect(frame.width, frame.height, target_size, tol):
        return None
    kappa = frame_curvature(frame, percentile, num_samples, smoothing)
    return FrameRecord(
        image_path=Path(image_path),
        label_path=Path(label_path),
        width=frame.width,
        height=frame.height,
        num_lanes=frame.num_lanes,
        kappa=kappa,
    )


def compute_frame_curvatures(
    pairs: list[tuple[Path, Path]],
    percentile: float = DEFAULT_PERCENTILE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    smoothing: float = DEFAULT_SMOOTHING,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    aspect_tol: float = 0.02,
    num_workers: int = 8,
) -> tuple[list[FrameRecord], int]:
    """Compute per-frame curvature for image/label pairs, excluding odd aspect.

    Args:
        pairs: ``(image_path, label_path)`` tuples (e.g. from ``index_split``).
        percentile: Per-lane curvature percentile (see :mod:`src.geometry.curvature`).
        num_samples: Curvature samples per lane.
        smoothing: Spline smoothing.
        target_size: Target ``(w, h)`` whose aspect defines the exclusion filter.
        aspect_tol: Relative aspect tolerance for inclusion.
        num_workers: Process-pool size; ``<= 1`` runs serially.

    Returns:
        ``(records, num_excluded)`` where ``records`` are the included frames and
        ``num_excluded`` counts frames dropped for odd aspect ratio.
    """
    work: list[_WorkerArgs] = [
        (str(img), str(lbl), percentile, num_samples, smoothing, target_size, aspect_tol)
        for img, lbl in pairs
    ]

    if num_workers <= 1:
        results = [_curvature_worker(a) for a in work]
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_curvature_worker, work, chunksize=64)

    records = [r for r in results if r is not None]
    excluded = len(results) - len(records)
    return records, excluded


# --- binning + flattened allocation ----------------------------------------


def compute_bin_edges(kappas: np.ndarray, n_bins: int) -> list[float]:
    """Fixed log-spaced curvature bin edges with an open top bin.

    Interior edges are log-spaced between the 5th and 95th percentiles of the
    curvature distribution (curvature spans ~an order of magnitude). The first
    bin starts at ``0`` and the last bin is open to ``inf`` so outliers land in
    the tightest-curvature bin rather than stretching the scale.

    Args:
        kappas: Curvature values of all included frames.
        n_bins: Number of bins ``K`` (``>= 2``).

    Returns:
        ``K + 1`` monotonically increasing edges; the last is ``inf``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    positive = kappas[kappas > 0]
    if positive.size == 0:
        # Degenerate: no curvature anywhere. Uniform edges over [0, 1].
        return [*np.linspace(0.0, 1.0, n_bins).tolist(), float("inf")]

    lo = max(float(np.percentile(positive, 5)), 1e-6)
    hi = float(np.percentile(positive, 95))
    if hi <= lo:
        hi = lo * 10.0
    interior = np.logspace(np.log10(lo), np.log10(hi), n_bins - 1)
    return [0.0, *interior.tolist(), float("inf")]


def assign_bins(kappas: np.ndarray, edges: list[float]) -> np.ndarray:
    """Assign each curvature to a bin index in ``[0, K-1]``."""
    # np.digitize with the inf top edge maps everything into [1, K]; shift to 0-based.
    idx = np.digitize(kappas, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def allocate_flattened(available: list[int], n_target: int) -> list[int]:
    """Allocate ``n_target`` samples roughly equally across bins, capped by supply.

    Each round distributes the remaining budget equally among bins that still
    have capacity; bins that fill up drop out and their surplus is redistributed.
    The result oversamples rare (tight-curvature) bins relative to their natural
    frequency, which is the point of flattening.

    Args:
        available: Available frame count per bin.
        n_target: Total desired sample count.

    Returns:
        Per-bin take counts, each ``<= available[b]`` and summing to
        ``min(n_target, sum(available))``.
    """
    take = [0] * len(available)
    remaining = min(n_target, sum(available))
    active = [b for b in range(len(available)) if available[b] > 0]

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Hand out the remainder one at a time to the emptiest-relative bins.
            for b in sorted(active, key=lambda b: available[b] - take[b], reverse=True):
                if remaining == 0:
                    break
                if take[b] < available[b]:
                    take[b] += 1
                    remaining -= 1
            break
        for b in list(active):
            add = min(share, available[b] - take[b])
            take[b] += add
            remaining -= add
            if take[b] >= available[b]:
                active.remove(b)
    return take


def build_subset(
    records: list[FrameRecord],
    n_target: int,
    n_bins: int,
    seed: int,
    excluded_aspect: int = 0,
    bin_edges: list[float] | None = None,
) -> Subset:
    """Bin frames by curvature and draw a flattened, seeded subset.

    Args:
        records: Per-frame curvature records (already aspect-filtered).
        n_target: Desired subset size.
        n_bins: Number of curvature bins.
        seed: RNG seed for reproducible sampling.
        excluded_aspect: Odd-aspect frames excluded upstream (recorded only).
        bin_edges: Precomputed edges to reuse (e.g. the training edges for a
            validation subset, so eval bins align with train stratification). If
            ``None``, edges are computed from ``records``.

    Returns:
        A :class:`Subset` with selected entries, bin edges, and both histograms.
    """
    if not records:
        return Subset([], bin_edges or [], [], [], seed, excluded_aspect)

    kappas = np.array([r.kappa for r in records], dtype=np.float64)
    edges = bin_edges if bin_edges is not None else compute_bin_edges(kappas, n_bins)
    n_bins = len(edges) - 1
    bins = assign_bins(kappas, edges)

    by_bin: list[list[int]] = [[] for _ in range(n_bins)]
    for i, b in enumerate(bins):
        by_bin[int(b)].append(i)
    natural_counts = [len(idxs) for idxs in by_bin]

    take = allocate_flattened(natural_counts, n_target)

    rng = random.Random(seed)
    entries: list[ManifestEntry] = []
    selected_counts = [0] * n_bins
    for b in range(n_bins):
        chosen = rng.sample(by_bin[b], take[b]) if take[b] else []
        selected_counts[b] = len(chosen)
        for i in chosen:
            r = records[i]
            entries.append(
                ManifestEntry(
                    image_path=r.image_path,
                    label_path=r.label_path,
                    width=r.width,
                    height=r.height,
                    num_lanes=r.num_lanes,
                    kappa=r.kappa,
                    bin_index=b,
                )
            )

    entries.sort(key=lambda e: (e.bin_index, e.image_path.name))
    return Subset(
        entries=entries,
        bin_edges=edges,
        natural_counts=natural_counts,
        selected_counts=selected_counts,
        seed=seed,
        excluded_aspect=excluded_aspect,
    )


def subset_to_dict(subset: Subset, extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a :class:`Subset` to a JSON-ready manifest dict.

    Paths are stored absolute so the manifest resolves regardless of CWD. Extra
    metadata (e.g. mask dir, target size, sky fraction) is merged into ``meta``.
    """
    meta: dict[str, Any] = {
        "seed": subset.seed,
        "n_bins": len(subset.natural_counts),
        "bin_edges": subset.bin_edges,
        "natural_counts": subset.natural_counts,
        "selected_counts": subset.selected_counts,
        "excluded_aspect": subset.excluded_aspect,
    }
    if extra_meta:
        meta.update(extra_meta)
    entries = [
        {
            "image": str(e.image_path),
            "label": str(e.label_path),
            "width": e.width,
            "height": e.height,
            "num_lanes": e.num_lanes,
            "kappa": e.kappa,
            "bin": e.bin_index,
        }
        for e in subset.entries
    ]
    return {"meta": meta, "entries": entries}


def write_manifest(subset: Subset, path: Path, extra_meta: dict[str, Any] | None = None) -> None:
    """Write a subset manifest to ``path`` as indented JSON."""
    Path(path).write_text(json.dumps(subset_to_dict(subset, extra_meta), indent=2))


def read_manifest(path: Path) -> tuple[list[ManifestEntry], dict[str, Any]]:
    """Read a manifest, returning ``(entries, meta)``."""
    doc = json.loads(Path(path).read_text())
    entries = [
        ManifestEntry(
            image_path=Path(e["image"]),
            label_path=Path(e["label"]),
            width=e["width"],
            height=e["height"],
            num_lanes=e["num_lanes"],
            kappa=e["kappa"],
            bin_index=e["bin"],
        )
        for e in doc["entries"]
    ]
    return entries, doc["meta"]


def format_histogram(subset: Subset) -> str:
    """Render a text histogram of natural vs selected counts per bin."""
    lines = ["bin  kappa_range              natural   selected"]
    edges = subset.bin_edges
    for b in range(len(subset.natural_counts)):
        lo, hi = edges[b], edges[b + 1]
        hi_str = "inf" if hi == float("inf") else f"{hi:7.3f}"
        lines.append(
            f"{b:>3}  [{lo:7.3f}, {hi_str})   {subset.natural_counts[b]:>8}   "
            f"{subset.selected_counts[b]:>8}"
        )
    lines.append(
        f"total{'':<24}{sum(subset.natural_counts):>8}   {sum(subset.selected_counts):>8}"
    )
    return "\n".join(lines)
