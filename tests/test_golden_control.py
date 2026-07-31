"""Regression guard for the filter and controller golden vectors.

These fixtures are the acceptance contract the C++ port is held to
(``deploy/test/test_control.cpp`` reads the same cases in flat form). This module guards
the Python side: the reference must keep reproducing the frozen values, and the
controller must keep satisfying the closed-form anchor that holds independently of any
fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.control.mpc import KinematicLateralMPC, MPCWeights, VehicleParams
from src.geometry.temporal import RoadGeometryFilter

_DIR = Path("tests/golden/control")
_skip = pytest.mark.skipif(
    not (_DIR / "mpc.json").exists(),
    reason="control vectors not exported; run scripts.export_control_vectors",
)


class _Geometry:
    """The three fields the filter reads off a RoadGeometry."""

    def __init__(self, offset, heading, curvature):
        self.lateral_offset_m = offset
        self.heading_error_rad = heading
        self.curvature_1pm = curvature


@_skip
def test_filter_reproduces_the_frozen_sequence():
    case = json.loads((_DIR / "filter.json").read_text())
    filt = RoadGeometryFilter(dt=case["dt"])
    for step, (m, want) in enumerate(zip(case["measurements"], case["outputs"])):
        got = filt.update(None if m is None else _Geometry(*m))
        assert got.lateral_offset_m == pytest.approx(want["lateral_offset_m"], abs=1e-12)
        assert got.heading_error_rad == pytest.approx(want["heading_error_rad"], abs=1e-12)
        assert got.curvature_1pm == pytest.approx(want["curvature_1pm"], abs=1e-12)
        assert got.measured is want["measured"], f"step {step}"
        assert got.accepted is want["accepted"], f"step {step}"
        assert got.coasting_frames == want["coasting_frames"], f"step {step}"


@_skip
def test_the_sequence_actually_exercises_coasting_and_gating():
    """A fixture that never drops a frame or rejects a measurement proves little."""
    case = json.loads((_DIR / "filter.json").read_text())
    coasted = sum(1 for o in case["outputs"] if not o["measured"])
    gated = sum(1 for o in case["outputs"] if o["measured"] and not o["accepted"])
    assert coasted >= 3 and gated >= 1


@_skip
@pytest.mark.parametrize(
    "case", json.loads((_DIR / "mpc.json").read_text()) if (_DIR / "mpc.json").exists() else [],
    ids=lambda c: c["name"],
)
def test_controller_reproduces_the_frozen_command(case):
    mpc = KinematicLateralMPC(VehicleParams(), MPCWeights(), horizon=20)
    s = mpc.steer_for_geometry(
        case["lateral_offset_m"], case["heading_error_rad"],
        case["curvature_1pm"], case["speed_mps"],
    )
    assert s.steer_rad == pytest.approx(case["steer_rad"], abs=1e-12)
    assert s.steer_unsaturated_rad == pytest.approx(case["steer_unsaturated_rad"], abs=1e-12)
    assert bool(s.saturated) is case["saturated"]


def test_steady_state_steer_is_the_ackermann_value():
    """Holds independently of any fixture, so it catches a port that regenerated them."""
    params = VehicleParams()
    mpc = KinematicLateralMPC(params, MPCWeights(), horizon=20)
    for kappa in (0.005, 0.01, 0.02, -0.015):
        s = mpc.steer_for_geometry(0.0, 0.0, kappa, 15.0)
        assert s.steer_unsaturated_rad == pytest.approx(
            params.wheelbase_m * kappa, rel=1e-6
        )
