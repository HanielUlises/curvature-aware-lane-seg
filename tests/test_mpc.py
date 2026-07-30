"""Tests for the kinematic lateral MPC."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.control.mpc import (
    KinematicLateralMPC,
    MPCWeights,
    VehicleParams,
)

SPEED = 20.0  # m/s


def _mpc(horizon=20, max_steer_deg=35.0, weights=None):
    return KinematicLateralMPC(
        params=VehicleParams(wheelbase_m=2.7, dt=0.05,
                             max_steer_rad=math.radians(max_steer_deg)),
        weights=weights or MPCWeights(),
        horizon=horizon,
    )


def test_zero_error_on_a_straight_path_commands_no_steer():
    s = _mpc().solve(0.0, 0.0, 0.0, SPEED)
    assert abs(s.steer_rad) < 1e-12
    assert not s.saturated


def test_offset_alone_steers_toward_the_path():
    # Vehicle left of the path must steer right, which is negative internally.
    left = _mpc().solve(cross_track_m=1.0, heading_rad=0.0, curvature_1pm=0.0,
                        speed_mps=SPEED)
    assert left.steer_rad < 0
    # Mirrored error gives a mirrored command.
    right = _mpc().solve(-1.0, 0.0, 0.0, SPEED)
    assert right.steer_rad > 0
    assert abs(left.steer_rad + right.steer_rad) < 1e-12


def test_heading_error_alone_steers_to_correct_it():
    # Pointing left of the tangent must steer right.
    s = _mpc().solve(0.0, math.radians(5.0), 0.0, SPEED)
    assert s.steer_rad < 0


def test_curvature_feedforward_matches_ackermann():
    # On a constant-curvature path with no tracking error the optimal steer is the
    # kinematic relation delta = L * kappa, reached without any error building up.
    wheelbase, kappa = 2.7, 0.02  # radius 50 m, left turn
    mpc = _mpc(horizon=40)
    s = mpc.solve(0.0, 0.0, kappa, SPEED)
    assert abs(s.steer_rad - wheelbase * kappa) < 2e-3
    # And the predicted trajectory stays on the path rather than drifting off it.
    assert np.max(np.abs(s.predicted_states[:, 0])) < 0.05


def test_saturation_is_applied_and_reported():
    tight = _mpc(max_steer_deg=2.0)
    s = tight.solve(cross_track_m=8.0, heading_rad=0.0, curvature_1pm=0.0,
                    speed_mps=SPEED)
    assert s.saturated
    assert abs(s.steer_rad) <= math.radians(2.0) + 1e-12
    assert abs(s.steer_unsaturated_rad) > abs(s.steer_rad)


def test_heavier_steer_penalty_gives_gentler_command():
    gentle = _mpc(weights=MPCWeights(cross_track=1.0, heading=0.5, steer=5.0))
    sharp = _mpc(weights=MPCWeights(cross_track=1.0, heading=0.5, steer=0.001))
    a = gentle.solve(1.0, 0.0, 0.0, SPEED).steer_rad
    b = sharp.solve(1.0, 0.0, 0.0, SPEED).steer_rad
    assert abs(a) < abs(b)


def test_measured_convention_flips_the_sign():
    mpc = _mpc()
    # Lane centre to the right means steer right, which is positive when reported in
    # the measured convention.
    s = mpc.steer_for_geometry(lateral_offset_m=1.0, heading_error_rad=0.0,
                               curvature_1pm=0.0, speed_mps=SPEED)
    assert s.steer_rad > 0
    # A right-hand curve needs right steer of the Ackermann magnitude.
    curve = mpc.steer_for_geometry(0.0, 0.0, 0.02, SPEED)
    assert curve.steer_rad > 0
    assert abs(curve.steer_rad - 2.7 * 0.02) < 3e-3


def test_closed_loop_converges_to_the_path():
    # Simulate the plant the controller was linearized from and confirm the error is
    # driven out rather than oscillating or diverging.
    mpc = _mpc(horizon=25)
    dt, wheelbase = mpc.params.dt, mpc.params.wheelbase_m
    e, psi = 1.5, math.radians(-2.0)
    history = []
    for _ in range(200):
        u = mpc.solve(e, psi, 0.0, SPEED).steer_rad
        e += dt * SPEED * math.sin(psi)
        psi += dt * (SPEED / wheelbase) * math.tan(u)
        history.append(abs(e))
    assert history[-1] < 0.02
    assert history[-1] < history[0]
    assert max(history[120:]) < 0.05  # settled, not oscillating


def test_closed_loop_tracks_a_constant_curve():
    mpc = _mpc(horizon=25)
    dt, wheelbase, kappa = mpc.params.dt, mpc.params.wheelbase_m, 0.01
    e, psi = 0.0, 0.0
    for _ in range(300):
        u = mpc.solve(e, psi, kappa, SPEED).steer_rad
        e += dt * SPEED * math.sin(psi)
        psi += dt * ((SPEED / wheelbase) * math.tan(u) - SPEED * kappa)
    assert abs(e) < 0.05  # curvature feedforward keeps it on the curve


def test_rejects_nonpositive_speed_and_bad_horizon():
    with pytest.raises(ValueError):
        _mpc().solve(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        KinematicLateralMPC(horizon=0)
