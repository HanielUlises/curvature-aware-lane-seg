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
| Mask decomposition | [centerline.hpp](include/curvature_port/centerline.hpp) | [centerline.py](../src/geometry/centerline.py) |
| Boundary tracking | [lane_tracker.hpp](include/curvature_port/lane_tracker.hpp) | [lane_tracker.py](../src/geometry/lane_tracker.py) |

The shared cubic-spline kernel is internal (`src/spline_internal.hpp`); deployment code
includes the three public headers only.

Not ported: the temporal filter and the controller itself.

## Performance

The whole mask-to-centreline path, which is what runs per frame on the vehicle, takes
**59 microseconds** per 512x288 frame on a desktop x86 core: 38 for the mask, 21 for the
tracker. The Python reference does the same work in 5,327 microseconds, so the port is
about 90 times faster, and a 20 Hz control loop spends roughly a tenth of a per cent of
its budget here.

Two changes account for most of that. Components are labelled over horizontal runs
rather than pixels, which took the mask stage from 335 to 108 microseconds, and the
background between runs is skipped a machine word at a time, which took it to 38. The
tracker's buffers are allocated once at construction, so the per-frame path allocates
nothing.

The release build is `-O3 -DNDEBUG`. `-march=native` is available behind
`-DCURVATURE_PORT_NATIVE=ON` but off by default, since a binary tuned for the building
machine will not run on an older CPU, which is the wrong default for something meant to
be deployed.

## Build and test

Requires CMake 3.16+, a C++17 compiler, and Eigen 3 (`apt install libeigen3-dev`).

```
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Three suites run: `curvature_golden`, `geometry_golden` and `pipeline_golden`. All read
fixtures from `test/golden/`, generated on the Python side:

```
python -m scripts.export_golden_vectors      # curvature
python -m scripts.export_geometry_vectors    # projection and road geometry
python -m scripts.export_pipeline_vectors infer.source=<clip-dir>   # mask to centreline
```

The pipeline fixture is a run of real predicted masks, run-length encoded, with the
centreline the reference produced from each. The stage is stateful, so a sequence is the
only meaningful test of it. `test_pipeline` takes an optional path argument, which is how
a longer video is checked without carrying its masks in the repository:

```
./build/test_pipeline /path/to/sequence.txt
```

Regenerate them after any change to the geometry mathematics, and re-run both this
suite and the Python guards (`tests/test_golden_curvature.py`,
`tests/test_golden_geometry.py`) — the fixtures are the only thing keeping the two
implementations honest with each other.

## NOTE: this port does not target FITPACK

The training path fits annotation polylines with a FITPACK smoothing spline through
SciPy. It has no portable C++ equivalent, so the port targets a fully specified
interpolating cubic spline instead, and correctness is established numerically against
shared vectors rather than by reading the two implementations side by side. The
rationale, the algorithm, and the tolerances are set out in the port specification.
