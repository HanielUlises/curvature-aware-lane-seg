"""Regression guard for the projection and road-geometry golden vectors.

These fixtures are the acceptance contract the C++ deployment port is held to
(``deploy/test/test_geometry.cpp`` reads the same cases in flat form). This module
guards the Python side of the contract: the reference implementations must keep
reproducing the frozen values, and on the cases with a closed-form answer they must
reproduce the true geometry rather than merely themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.geometry.ipm import GroundPlane, apply_homography, homography_from_points
from src.geometry.road_geometry_portable import portable_road_geometry

_GOLDEN_DIR = Path("tests/golden/geometry")


def _cases(prefix: str) -> list[dict]:
    if not _GOLDEN_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(_GOLDEN_DIR.glob(f"{prefix}*.json"))]


_PROJECTIONS = _cases("projection")
_GEOMETRIES = _cases("geometry")
_skip = pytest.mark.skipif(
    not _PROJECTIONS or not _GEOMETRIES,
    reason="golden vectors not exported; run scripts.export_geometry_vectors",
)


@_skip
@pytest.mark.parametrize("case", _PROJECTIONS, ids=[c["name"] for c in _PROJECTIONS])
def test_dlt_reproduces_frozen_homography(case: dict) -> None:
    h = homography_from_points(np.array(case["src"]), np.array(case["dst"]))
    assert h == pytest.approx(np.array(case["expected_h"]), rel=1e-9, abs=1e-12)

    probe = np.array(case["probe_image"], dtype=np.float64)
    got = apply_homography(h, probe)
    assert got == pytest.approx(np.array(case["expected_ground"]), rel=1e-9, abs=1e-12)

    # The correspondences themselves must land on their targets: this is what makes
    # the frozen values meaningful rather than self-consistent.
    mapped_src = apply_homography(h, np.array(case["src"], dtype=np.float64))
    assert mapped_src == pytest.approx(np.array(case["dst"]), rel=1e-8, abs=1e-8)

    plane = GroundPlane(h=h, h_inv=np.linalg.inv(h))
    assert plane.to_image(plane.to_ground(probe)) == pytest.approx(probe, abs=1e-6)


@_skip
@pytest.mark.parametrize("case", _GEOMETRIES, ids=[c["name"] for c in _GEOMETRIES])
def test_portable_geometry_reproduces_frozen_values(case: dict) -> None:
    points = np.array(case["points"], dtype=np.float64)
    ground = points
    if case["homography"]:
        ground = apply_homography(np.array(case["homography"]), points)

    rg = portable_road_geometry(
        ground,
        tuple(case["preview_distances_m"]),
        case["num_samples"],
        case["offset_distance_m"],
    )
    assert rg is not None
    assert rg.lateral_offset_m == pytest.approx(case["expected_offset_m"], abs=1e-12)
    assert rg.heading_error_rad == pytest.approx(case["expected_heading_rad"], abs=1e-12)
    assert rg.curvature_1pm == pytest.approx(case["expected_curvature_1pm"], abs=1e-12)
    assert rg.preview_curvature_1pm == pytest.approx(
        np.array(case["expected_preview_curvature_1pm"]), abs=1e-12, nan_ok=True
    )


@_skip
def test_arc_previews_hold_the_true_curvature_at_every_lookahead() -> None:
    """Curvature must not sag at the near end, where the controller reads it.

    Under the natural end condition the spline was pinned to zero second derivative
    at the polyline ends, so the 5 m preview on a 50 m arc came back at 0.0032 1/m
    instead of 0.02. The port and its reference now use not-a-knot ends; this is the
    test that would catch a regression to the old behaviour.
    """
    theta = np.linspace(0.0, 0.8, 40)
    radius = 50.0
    arc = np.column_stack(
        [radius - radius * np.cos(theta), radius * np.sin(theta)]
    )
    rg = portable_road_geometry(arc, (5.0, 10.0, 20.0))
    assert rg is not None
    assert rg.preview_curvature_1pm == pytest.approx(1.0 / radius, rel=1e-3)
    assert rg.curvature_1pm == pytest.approx(1.0 / radius, rel=1e-3)
