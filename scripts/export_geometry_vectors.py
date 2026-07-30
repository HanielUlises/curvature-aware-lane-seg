"""Export golden test vectors for the projection and road-geometry stages.

Companion to :mod:`scripts.export_golden_vectors`, which covers curvature. Same
contract, extended to the two stages that follow it in the perception-to-control
chain:

- **projection** (:mod:`src.geometry.ipm`): the Direct Linear Transform and the
  image-to-ground mapping it produces. Both sides run identical mathematics here,
  so the port is held to floating-point agreement rather than a loose tolerance.
- **road geometry** (:mod:`src.geometry.road_geometry_portable`): lateral offset,
  heading error, and preview curvature read off a ground centreline. The portable
  reference is used rather than the FITPACK-backed training implementation, for
  the reason set out in ``docs/geometry_port_spec.md``.

Cases are written twice from the same source: canonical JSON under
``tests/golden/geometry`` for the Python regression test, and a flat
whitespace-delimited form under ``deploy/test/golden/geometry`` that the C++ test
parses without a JSON dependency.

Run:
    python -m scripts.export_geometry_vectors
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.geometry.ipm import build_ground_homography, homography_from_points
from src.geometry.road_geometry_portable import portable_road_geometry

_JSON_DIR = Path("tests/golden/geometry")
_TXT_DIR = Path("deploy/test/golden/geometry")

_NUM_SAMPLES = 100
_OFFSET_DISTANCE_M = 5.0
_PREVIEWS = (5.0, 10.0, 20.0)
# The projection is the same arithmetic in both languages; hold it tight.
_PROJECTION_TOL = 1e-9
# The road-geometry read-out accumulates a spline solve and an interpolation.
_GEOMETRY_TOL = 1e-6


@dataclass
class ProjectionCase:
    """A homography to be recovered from correspondences, plus points to map."""

    name: str
    description: str
    src: list[list[float]]
    dst: list[list[float]]
    expected_h: list[list[float]]
    probe_image: list[list[float]]
    expected_ground: list[list[float]]
    tolerance: float


@dataclass
class GeometryCase:
    """A centreline whose control read-out the port must reproduce.

    ``homography`` is empty when ``points`` are already ground metres; otherwise
    they are image pixels the port must project first, which exercises the two
    stages composed.
    """

    name: str
    description: str
    points: list[list[float]]
    homography: list[list[float]]
    num_samples: int
    offset_distance_m: float
    preview_distances_m: list[float]
    expected_offset_m: float
    expected_heading_rad: float
    expected_curvature_1pm: float
    expected_preview_curvature_1pm: list[float]
    tolerance: float = _GEOMETRY_TOL
    valid: bool = field(default=True)


def _demo_ground_plane():
    """The project's configured placeholder mapping at the training frame size."""
    return build_ground_homography(
        image_size=(512, 288),
        src_trapezoid=((0.43, 0.58), (0.57, 0.58), (0.98, 1.00), (0.02, 1.00)),
        lane_width_m=3.7,
        look_ahead_m=30.0,
    )


def _projection_cases() -> list[ProjectionCase]:
    cases: list[ProjectionCase] = []

    # 1. The configured trapezoid mapping, probed across the road region.
    plane = _demo_ground_plane()
    src = np.array(
        [[0.43 * 512, 0.58 * 288], [0.57 * 512, 0.58 * 288],
         [0.98 * 512, 288.0], [0.02 * 512, 288.0]]
    )
    dst = np.array([[-1.85, 30.0], [1.85, 30.0], [1.85, 0.0], [-1.85, 0.0]])
    probe = np.array(
        [[256.0, 288.0], [256.0, 240.0], [200.0, 200.0], [320.0, 180.0], [256.0, 170.0]]
    )
    cases.append(
        ProjectionCase(
            "projection_trapezoid",
            "Configured placeholder road trapezoid mapped to a ground rectangle",
            src.tolist(), dst.tolist(), plane.h.tolist(),
            probe.tolist(), plane.to_ground(probe).tolist(), _PROJECTION_TOL,
        )
    )

    # 2. A general (non-trapezoidal, six-point) correspondence set: the DLT must
    #    solve the overdetermined system, not just the minimal one.
    rng = np.random.default_rng(7)
    general_src = np.array(
        [[10.0, 20.0], [400.0, 35.0], [480.0, 260.0], [30.0, 275.0],
         [250.0, 150.0], [120.0, 90.0]]
    )
    truth = np.array([[1.2, 0.15, -30.0], [-0.1, 0.9, 12.0], [2e-4, 1.1e-3, 1.0]])
    homog = np.column_stack([general_src, np.ones(len(general_src))]) @ truth.T
    general_dst = homog[:, :2] / homog[:, 2:3]
    solved = homography_from_points(general_src, general_dst)
    probe2 = rng.uniform([0.0, 0.0], [512.0, 288.0], size=(6, 2))
    mapped = np.column_stack([probe2, np.ones(len(probe2))]) @ solved.T
    cases.append(
        ProjectionCase(
            "projection_general",
            "Six correspondences from a known homography, recovered by DLT",
            general_src.tolist(), general_dst.tolist(), solved.tolist(),
            probe2.tolist(), (mapped[:, :2] / mapped[:, 2:3]).tolist(), _PROJECTION_TOL,
        )
    )
    return cases


def _straight(x_of_z, z0=12.0, z1=40.0, n=30) -> np.ndarray:
    z = np.linspace(z0, z1, n)
    return np.column_stack([x_of_z(z), z])


def _arc(radius: float, right: bool, theta_max=0.8, n=40) -> np.ndarray:
    theta = np.linspace(0.0, theta_max, n)
    x = radius - radius * np.cos(theta)
    return np.column_stack([x if right else -x, radius * np.sin(theta)])


def _geometry_cases() -> list[GeometryCase]:
    cases: list[GeometryCase] = []

    def add(name: str, description: str, points: np.ndarray, homography=None) -> None:
        ground = points
        if homography is not None:
            homog = np.column_stack([points, np.ones(len(points))]) @ np.asarray(homography).T
            ground = homog[:, :2] / homog[:, 2:3]
        rg = portable_road_geometry(
            ground, _PREVIEWS, _NUM_SAMPLES, _OFFSET_DISTANCE_M
        )
        assert rg is not None, name
        cases.append(
            GeometryCase(
                name, description, points.tolist(),
                [] if homography is None else np.asarray(homography).tolist(),
                _NUM_SAMPLES, _OFFSET_DISTANCE_M, list(_PREVIEWS),
                rg.lateral_offset_m, rg.heading_error_rad, rg.curvature_1pm,
                rg.preview_curvature_1pm.tolist(),
            )
        )

    add("geometry_straight_centred", "Straight lane on the vehicle axis",
        _straight(lambda z: np.zeros_like(z)))
    add("geometry_straight_offset", "Straight lane 1.4 m to the right",
        _straight(lambda z: np.full_like(z, 1.4)))
    add("geometry_heading", "Lane drifting right at a constant 0.12 rad slope",
        _straight(lambda z: 0.12 * z))
    add("geometry_arc_right_R50", "Right-hand arc, kappa == +1/50",
        _arc(50.0, right=True))
    add("geometry_arc_left_R80", "Left-hand arc, kappa == -1/80",
        _arc(80.0, right=False))

    # A noisy centreline: the near-field fit and the percentile-free median summary
    # must both stay stable under the kind of jitter a segmenter actually produces.
    rng = np.random.default_rng(11)
    noisy = _arc(120.0, right=True, n=45)
    noisy[:, 0] += rng.normal(0.0, 0.05, size=noisy.shape[0])
    add("geometry_arc_noisy", "Gentle right arc with 5 cm lateral jitter", noisy)

    # Composed: image-space centreline, projected by the configured mapping first.
    plane = _demo_ground_plane()
    rows = np.linspace(175.0, 287.0, 40)
    cols = 256.0 + 0.0016 * (287.0 - rows) ** 2  # bends right with distance
    add("geometry_from_image", "Image-space centreline through the trapezoid mapping",
        np.column_stack([cols, rows]), homography=plane.h)
    return cases


def _fmt(x: float) -> str:
    return "nan" if x != x else repr(float(x))


def _write_projection_txt(case: ProjectionCase, path: Path) -> None:
    lines = [f"name {case.name}", f"tolerance {case.tolerance!r}"]
    for key, pts in (("src", case.src), ("dst", case.dst)):
        lines.append(f"{key} {len(pts)}")
        lines.extend(f"{_fmt(a)} {_fmt(b)}" for a, b in pts)
    lines.append("expected_h 9")
    lines.extend(" ".join(_fmt(v) for v in row) for row in case.expected_h)
    lines.append(f"probe {len(case.probe_image)}")
    lines.extend(
        f"{_fmt(u)} {_fmt(v)} {_fmt(x)} {_fmt(z)}"
        for (u, v), (x, z) in zip(case.probe_image, case.expected_ground)
    )
    path.write_text("\n".join(lines) + "\n")


def _write_geometry_txt(case: GeometryCase, path: Path) -> None:
    lines = [
        f"name {case.name}",
        f"tolerance {case.tolerance!r}",
        f"num_samples {case.num_samples}",
        f"offset_distance {case.offset_distance_m!r}",
        f"expected_offset {_fmt(case.expected_offset_m)}",
        f"expected_heading {_fmt(case.expected_heading_rad)}",
        f"expected_curvature {_fmt(case.expected_curvature_1pm)}",
        f"homography {len(case.homography)}",
    ]
    lines.extend(" ".join(_fmt(v) for v in row) for row in case.homography)
    lines.append(f"previews {len(case.preview_distances_m)}")
    lines.extend(
        f"{_fmt(z)} {_fmt(k)}"
        for z, k in zip(case.preview_distances_m, case.expected_preview_curvature_1pm)
    )
    lines.append(f"points {len(case.points)}")
    lines.extend(f"{_fmt(a)} {_fmt(b)}" for a, b in case.points)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    _TXT_DIR.mkdir(parents=True, exist_ok=True)

    projections = _projection_cases()
    geometries = _geometry_cases()

    for case in projections:
        (_JSON_DIR / f"{case.name}.json").write_text(json.dumps(asdict(case), indent=2))
        _write_projection_txt(case, _TXT_DIR / f"{case.name}.txt")
        print(f"[projection] {case.name:<22} {len(case.probe_image)} probes")

    for case in geometries:
        (_JSON_DIR / f"{case.name}.json").write_text(json.dumps(asdict(case), indent=2))
        _write_geometry_txt(case, _TXT_DIR / f"{case.name}.txt")
        print(
            f"[geometry]   {case.name:<22} offset={case.expected_offset_m:+.3f} m  "
            f"heading={np.degrees(case.expected_heading_rad):+.2f} deg  "
            f"kappa={case.expected_curvature_1pm:+.5f} 1/m"
        )

    (_TXT_DIR / "index_projection.txt").write_text(
        "\n".join(c.name for c in projections) + "\n"
    )
    (_TXT_DIR / "index_geometry.txt").write_text(
        "\n".join(c.name for c in geometries) + "\n"
    )
    total = len(projections) + len(geometries)
    print(f"[done] {total} cases -> {_JSON_DIR} and {_TXT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
