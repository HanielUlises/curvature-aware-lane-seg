"""Regression guard for the curvature golden vectors.

The golden vectors are the acceptance contract for the C++ deployment port
(``deploy/``). ``python_p90`` is the **portable natural-cubic reference**
(:mod:`src.geometry.curvature_portable`) that the C++ mirrors. Separately, the
FITPACK training implementation (:mod:`src.geometry.curvature`) must still recover
the analytic curvature on closed-form cases — the two agree on real geometry even
though they differ on noisy polylines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.geometry.curvature import lane_curvature
from src.geometry.curvature_portable import lane_curvature_natural

_GOLDEN_DIR = Path("tests/golden/curvature")


def _cases() -> list[dict]:
    if not _GOLDEN_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(_GOLDEN_DIR.glob("*.json"))]


_CASES = _cases()
_skip = pytest.mark.skipif(
    not _CASES, reason="golden vectors not exported; run scripts.export_golden_vectors"
)


@_skip
@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_portable_reference_matches_frozen_golden(case: dict) -> None:
    """The portable natural-cubic reference reproduces its own frozen values."""
    points = np.array(case["points"], dtype=np.float64)
    got = lane_curvature_natural(points, case["percentile"], case["num_samples"])
    assert got == pytest.approx(case["python_p90"], rel=1e-9, abs=1e-12)

    if case["expected_analytic"] is not None:
        expected = case["expected_analytic"]
        if expected == 0.0:
            assert got < 1e-3
        else:
            assert got == pytest.approx(expected, rel=max(case["tolerance"], 0.05))


@_skip
@pytest.mark.parametrize(
    "case",
    [c for c in _CASES if c["expected_analytic"] is not None],
    ids=[c["name"] for c in _CASES if c["expected_analytic"] is not None],
)
def test_fitpack_reference_also_recovers_analytic(case: dict) -> None:
    """FITPACK (training path) recovers the same closed-form curvature."""
    points = np.array(case["points"], dtype=np.float64)
    got = lane_curvature(points, case["percentile"], case["num_samples"], smoothing=0.0)
    expected = case["expected_analytic"]
    if expected == 0.0:
        assert got < 1e-3
    else:
        assert got == pytest.approx(expected, rel=0.05)
