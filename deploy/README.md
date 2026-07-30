# Deployment port (C++)

The on-vehicle consumer of this system is a kinematic MPC controller running in C++,
so the geometry between the segmenter and the controller is ported rather than called
through Python. This directory holds that port and the tests that hold it to the
numerical contract in [docs/geometry_port_spec.md](../docs/geometry_port_spec.md).

## What is ported

| Stage | Header | Python counterpart |
|---|---|---|
| Curvature | [curvature.hpp](include/curvature_port/curvature.hpp) | [curvature_portable.py](../src/geometry/curvature_portable.py) |
| Projection | [ipm.hpp](include/curvature_port/ipm.hpp) | [ipm.py](../src/geometry/ipm.py) |
| Road geometry | [road_geometry.hpp](include/curvature_port/road_geometry.hpp) | [road_geometry_portable.py](../src/geometry/road_geometry_portable.py) |

The shared cubic-spline kernel is internal (`src/spline_internal.hpp`); deployment code
includes the three public headers only.

Not ported: the lane-mask decomposition that produces the centreline, the temporal
filter, and the controller itself.

## Build and test

Requires CMake 3.16+, a C++17 compiler, and Eigen 3 (`apt install libeigen3-dev`).

```
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Two suites run: `curvature_golden` and `geometry_golden`. Both read fixtures from
`test/golden/`, which are generated on the Python side:

```
python -m scripts.export_golden_vectors      # curvature
python -m scripts.export_geometry_vectors    # projection and road geometry
```

Regenerate them after any change to the geometry mathematics, and re-run both this
suite and the Python guards (`tests/test_golden_curvature.py`,
`tests/test_golden_geometry.py`) — the fixtures are the only thing keeping the two
implementations honest with each other.

## Why the port does not target FITPACK

The training path fits annotation polylines with a FITPACK smoothing spline through
SciPy. It has no portable C++ equivalent, so the port targets a fully specified
interpolating cubic spline instead, and correctness is established numerically against
shared vectors rather than by reading the two implementations side by side. The
rationale, the algorithm, and the tolerances are set out in the port specification.
