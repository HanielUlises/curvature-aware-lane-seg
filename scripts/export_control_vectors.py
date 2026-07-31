"""Export golden vectors for the temporal filter and the controller.

Completes the contract set. The filter is stateful and the controller is not, so they are
pinned differently:

- **Filter**: a sequence of measurements including gaps and a deliberate outlier, with the
  filtered value after every step. A port that gets the gating or the coasting wrong
  diverges within a few steps, which a single-step fixture would never show.
- **Controller**: independent solves across speeds, curvatures and error states, each
  pinned to the reference's steering command. Includes the closed-form anchor, since on a
  constant-curvature path in steady state the optimal steer is the Ackermann value
  ``L * kappa`` and any port that disagrees there is wrong regardless of the fixtures.

Run:
    python -m scripts.export_control_vectors
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.control.mpc import KinematicLateralMPC, MPCWeights, VehicleParams
from src.geometry.temporal import RoadGeometryFilter

_JSON_DIR = Path("tests/golden/control")
_TXT_DIR = Path("deploy/test/golden/control")


def _filter_case() -> dict:
    """A 40-step run with a dropout, an outlier, and a manoeuvre."""
    rng = np.random.default_rng(3)
    steps = []
    offset, heading, curvature = 0.0, 0.0, 0.0
    for k in range(40):
        # A lane change between steps 12 and 24, so the rate estimate has to work.
        if 12 <= k < 24:
            offset += 0.12
            heading = 0.05
        curvature = 0.004 if k >= 28 else 0.0

        if k in (17, 18, 31):
            steps.append(None)                       # no ego lane recovered
        elif k == 22:
            steps.append((offset + 6.0, heading, curvature))   # gross outlier
        else:
            steps.append((
                offset + float(rng.normal(0.0, 0.03)),
                heading + float(rng.normal(0.0, 0.01)),
                curvature + float(rng.normal(0.0, 2e-4)),
            ))

    filt = RoadGeometryFilter(dt=0.05)
    outputs = []
    for s in steps:
        if s is None:
            f = filt.update(None)
        else:
            class _G:  # a RoadGeometry only needs these three fields here
                lateral_offset_m, heading_error_rad, curvature_1pm = s
            f = filt.update(_G())
        outputs.append({
            "lateral_offset_m": f.lateral_offset_m,
            "heading_error_rad": f.heading_error_rad,
            "curvature_1pm": f.curvature_1pm,
            "measured": f.measured,
            "accepted": f.accepted,
            "coasting_frames": f.coasting_frames,
        })
    return {"dt": 0.05, "measurements": steps, "outputs": outputs, "tolerance": 1e-9}


def _mpc_cases() -> list[dict]:
    params = VehicleParams()
    mpc = KinematicLateralMPC(params, MPCWeights(), horizon=20)
    cases = []
    grid = [
        ("centred_straight", 0.0, 0.0, 0.0, 15.0),
        ("offset_right", 0.6, 0.0, 0.0, 15.0),
        ("offset_left", -0.6, 0.0, 0.0, 15.0),
        ("heading_only", 0.0, 0.08, 0.0, 15.0),
        ("right_curve", 0.0, 0.0, 0.02, 15.0),
        ("left_curve", 0.0, 0.0, -0.02, 15.0),
        ("combined", 0.45, -0.05, 0.012, 22.0),
        ("slow", 0.3, 0.02, 0.005, 5.0),
        ("saturating", 3.5, 0.4, 0.05, 30.0),
    ]
    for name, off, head, kappa, speed in grid:
        s = mpc.steer_for_geometry(off, head, kappa, speed)
        cases.append({
            "name": name,
            "lateral_offset_m": off,
            "heading_error_rad": head,
            "curvature_1pm": kappa,
            "speed_mps": speed,
            "steer_rad": s.steer_rad,
            "steer_unsaturated_rad": s.steer_unsaturated_rad,
            "saturated": bool(s.saturated),
            "tolerance": 1e-9,
        })
    return cases


def main() -> int:
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    _TXT_DIR.mkdir(parents=True, exist_ok=True)

    filt = _filter_case()
    mpc = _mpc_cases()
    (_JSON_DIR / "filter.json").write_text(json.dumps(filt, indent=2))
    (_JSON_DIR / "mpc.json").write_text(json.dumps(mpc, indent=2))

    lines = [f"dt {filt['dt']!r}", f"tolerance {filt['tolerance']!r}",
             f"steps {len(filt['measurements'])}"]
    for m, o in zip(filt["measurements"], filt["outputs"]):
        if m is None:
            lines.append("m none")
        else:
            lines.append("m " + " ".join(repr(float(v)) for v in m))
        lines.append("o " + " ".join([
            repr(o["lateral_offset_m"]), repr(o["heading_error_rad"]),
            repr(o["curvature_1pm"]), str(int(o["measured"])),
            str(int(o["accepted"])), str(o["coasting_frames"]),
        ]))
    (_TXT_DIR / "filter.txt").write_text("\n".join(lines) + "\n")

    p = VehicleParams()
    lines = [f"wheelbase {p.wheelbase_m!r}", f"dt {p.dt!r}",
             f"max_steer {p.max_steer_rad!r}", f"horizon 20", f"cases {len(mpc)}"]
    for c in mpc:
        lines.append(f"case {c['name']}")
        lines.append(" ".join(repr(float(c[k])) for k in
                              ("lateral_offset_m", "heading_error_rad",
                               "curvature_1pm", "speed_mps")))
        lines.append(" ".join([repr(c["steer_rad"]), repr(c["steer_unsaturated_rad"]),
                               str(int(c["saturated"])), repr(c["tolerance"])]))
    (_TXT_DIR / "mpc.txt").write_text("\n".join(lines) + "\n")

    coasted = sum(1 for o in filt["outputs"] if not o["measured"])
    gated = sum(1 for o in filt["outputs"] if o["measured"] and not o["accepted"])
    print(f"[filter] {len(filt['outputs'])} steps, {coasted} coasted, {gated} gated")
    for c in mpc:
        print(f"[mpc]    {c['name']:<18} steer {math.degrees(c['steer_rad']):+7.2f} deg"
              f"{'  (saturated)' if c['saturated'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
