"""Export golden test vectors for the curvature algorithm.

These vectors are the **language-agnostic numerical contract** for
``src/geometry/curvature.py``. The C++ deployment port (``deploy/``) cannot call
scipy/FITPACK, so it is validated against these vectors rather than by reading
code side by side.

Each case stores the exact points curvature is computed on (already normalized
where relevant), the algorithm parameters, the analytic curvature where it is
known in closed form, and the Python reference result. Two formats are written
from the same source:

- ``tests/golden/curvature/*.json`` — canonical, human-readable, used by the
  Python regression test.
- ``deploy/test/golden/*.txt`` — a flat whitespace-delimited form the C++ test
  parses without a JSON dependency.

Run:
    python -m scripts.export_golden_vectors
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.curvelanes import build_frame, index_split
from src.data.rasterize import is_target_aspect
from src.geometry.curvature_portable import lane_curvature_natural

_JSON_DIR = Path("tests/golden/curvature")
_TXT_DIR = Path("deploy/test/golden")

_PERCENTILE = 90.0
_NUM_SAMPLES = 100


@dataclass
class GoldenCase:
    name: str
    description: str
    points: list[list[float]]
    percentile: float
    num_samples: int
    # Analytic |kappa| if known in closed form, else None.
    expected_analytic: float | None
    # Portable natural-cubic reference result (the algorithm the C++ port mirrors).
    python_p90: float
    # Which target the C++ port is asserted against, and the allowed rel. error.
    compare: str  # "analytic" | "python"
    tolerance: float


def _p90(points: np.ndarray) -> float:
    return lane_curvature_natural(points, _PERCENTILE, _NUM_SAMPLES)


def _analytic_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # Straight line: curvature is exactly 0.
    t = np.linspace(0.0, 100.0, 25)
    line = np.stack([t, 2.0 * t + 5.0], axis=1)
    cases.append(
        GoldenCase(
            "line", "Straight line, kappa == 0", line.tolist(), _PERCENTILE,
            _NUM_SAMPLES, 0.0, _p90(line), "analytic", 1e-3,
        )
    )

    # Circles: curvature is exactly 1/R (compared on the interior via p90).
    for radius in (50.0, 100.0, 250.0):
        theta = np.linspace(-math.pi / 3, math.pi / 3, 45)
        arc = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        cases.append(
            GoldenCase(
                f"circle_R{int(radius)}", f"Circular arc, kappa == 1/{int(radius)}",
                arc.tolist(), _PERCENTILE, _NUM_SAMPLES, 1.0 / radius, _p90(arc),
                "analytic", 0.05,
            )
        )

    # Clothoid (Euler spiral): no single scalar; pin to the Python reference.
    from scipy.special import fresnel

    tt = np.linspace(0.05, 2.0, 200)
    sin_s, cos_s = fresnel(tt)
    clo = np.stack([cos_s, sin_s], axis=1) * 50.0
    cases.append(
        GoldenCase(
            "clothoid", "Euler spiral, curvature linear in arclength",
            clo.tolist(), _PERCENTILE, _NUM_SAMPLES, None, _p90(clo), "python", 0.02,
        )
    )
    return cases


def _real_cases(n: int = 3) -> list[GoldenCase]:
    """A few real CurveLanes lanes (width-normalized), pinned to Python output."""
    root = Path("data/raw/Curvelanes/train")
    idx = index_split(root / "images", root / "labels")
    if not idx.pairs:
        return []

    cases: list[GoldenCase] = []
    for image_path, label_path in idx.pairs:
        if len(cases) >= n:
            break
        frame = build_frame(image_path, label_path)
        if not is_target_aspect(frame.width, frame.height):
            continue
        # Pick the frame's most-curved lane, normalized by native width.
        best = None
        best_k = -1.0
        for lane in frame.lanes:
            pts = lane.points / float(frame.width)
            k = _p90(pts)
            if k > best_k:
                best_k, best = k, pts
        if best is None or best.shape[0] < 4:
            continue
        cases.append(
            GoldenCase(
                f"real_{image_path.stem[:8]}",
                f"Real CurveLanes lane (width-normalized), {best.shape[0]} points",
                best.tolist(), _PERCENTILE, _NUM_SAMPLES, None, best_k, "python", 0.02,
            )
        )
    return cases


def _write_txt(case: GoldenCase, path: Path) -> None:
    analytic = "nan" if case.expected_analytic is None else repr(case.expected_analytic)
    lines = [
        f"name {case.name}",
        f"percentile {case.percentile!r}",
        f"num_samples {case.num_samples}",
        f"expected_analytic {analytic}",
        f"python_p90 {case.python_p90!r}",
        f"tolerance {case.tolerance!r}",
        f"compare {case.compare}",
        f"points {len(case.points)}",
    ]
    lines.extend(f"{x!r} {y!r}" for x, y in case.points)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    _TXT_DIR.mkdir(parents=True, exist_ok=True)

    cases = _analytic_cases() + _real_cases()
    index = []
    for case in cases:
        (_JSON_DIR / f"{case.name}.json").write_text(json.dumps(asdict(case), indent=2))
        _write_txt(case, _TXT_DIR / f"{case.name}.txt")
        index.append(case.name)
        tag = (
            f"analytic={case.expected_analytic:.4g}"
            if case.expected_analytic is not None
            else f"python={case.python_p90:.4g}"
        )
        print(f"[golden] {case.name:<18} p90={case.python_p90:.4g}  {tag}  tol={case.tolerance}")

    (_TXT_DIR / "index.txt").write_text("\n".join(index) + "\n")
    print(f"[done] {len(cases)} cases -> {_JSON_DIR} and {_TXT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
