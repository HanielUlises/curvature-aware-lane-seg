"""Plot detection rate against curvature for a set of control-metric runs.

The project's central finding was that detection rate falls with curvature while IoU does
not, so the figure that settles whether the curvature-aware objective worked is this one:
per-bin detection for each weighting strength, on one pair of axes.

Reads the ``control_metrics.json`` files written by :mod:`scripts.eval_control`, so the
figure is generated from the evaluation artefacts rather than from numbers transcribed by
hand.

Run:
    python -m scripts.plot_curvature_ablation \\
        "baseline=outputs/<run>/control_metrics.json" \\
        "curvature w=0.5=outputs/<run>/control_metrics.json"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/assets/fig_curvature_objective.png")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    series = []
    for arg in argv:
        label, _, path = arg.rpartition("=")
        data = json.loads(Path(path).read_text())
        series.append((label or Path(path).parent.name, data))

    names = [b["name"] for b in series[0][1]["per_bin"]]
    x = range(len(names))

    # gridspec_kw rather than width_ratios: the latter needs a newer matplotlib than
    # this environment carries.
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    for label, data in series:
        y = [b["detection_rate"] * 100 for b in data["per_bin"]]
        ax.plot(x, y, marker="o", linewidth=2, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("frames yielding a usable ego lane (%)")
    ax.set_xlabel("curvature bin")
    ax.set_title("Detection rate across curvature")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # The gap between the straightest and tightest bins is the finding in one number.
    gaps = [
        (label, data["per_bin"][0]["detection_rate"] * 100
         - data["per_bin"][-1]["detection_rate"] * 100)
        for label, data in series
    ]
    ax2.bar([g[0] for g in gaps], [g[1] for g in gaps],
            color=["#c44", "#c84", "#4a8"][: len(gaps)])
    ax2.set_ylabel("straight minus tightest (points)")
    ax2.set_title("Curvature-dependent failure")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(alpha=0.3, axis="y")
    for i, (_, g) in enumerate(gaps):
        ax2.text(i, g + 0.4, f"{g:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
