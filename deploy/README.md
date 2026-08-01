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
| Temporal filter | [temporal.hpp](include/curvature_port/temporal.hpp) | [temporal.py](../src/geometry/temporal.py) |
| Controller | [mpc.hpp](include/curvature_port/mpc.hpp) | [mpc.py](../src/control/mpc.py) |

The shared cubic-spline kernel is internal (`src/spline_internal.hpp`); deployment code
includes the public headers only.

The chain is complete: a mask goes in, a steering command comes out, with no Python on
the path.

## Calling it from Python

`libcurvature_port_c.so` exposes a C ABI (`include/curvature_port/c_api.h`) that
[`src/native.py`](../src/native.py) drives through `ctypes`, so Python runs the same code
the vehicle does rather than a second implementation of it:

```python
from src import native
chain = native.NativeChain(512, 288, calibration)   # fx fy cx cy height pitch yaw
result = chain.process(mask, speed_mps=15.0)        # -> offset, heading, curvature, steer
```

`ctypes` rather than pybind11 on purpose: no build dependency beyond the compiler the port
already needs, and nothing to install. The cost is that everything crossing the boundary
is plain data, which is why the result is a flat struct and polylines are copied into
caller-owned buffers.

The pure-Python implementations stay, but as the **reference the fixtures are generated
from and the port is checked against**, not as a runtime path. That distinction is load
bearing: if Python became a thin wrapper with nothing behind it, the golden vectors would
be comparing the port against itself. `tests/test_native.py` exercises the reference
directly and requires the native output to match it over a sequence.

Measured over the committed 40-frame fixture: the pure-Python chain takes 16,442 us per
frame and the native chain 178 us, a **51x** speedup. `process_into`, which reuses the
result struct instead of building a Python object, runs at 162 us — the same as the
standalone C++ binary, so the `ctypes` boundary itself costs nothing measurable (a
trivial exported call is 0.22 us).

## Performance

The whole per-frame path takes about **157 microseconds** per 512x288 frame on a desktop
x86 core: 38 for the mask decomposition, 22 for the tracker, and 97 for the projection,
road-geometry read-out, filter and MPC together. A 20 Hz control loop spends about three
tenths of a per cent of its budget here. The Python reference needs 5,327 microseconds for
the mask and tracker alone, so that part of the port is roughly 90 times faster.

Three changes account for most of that. Components are labelled over horizontal runs
rather than pixels, which took the mask stage from 335 to 108 microseconds, and the
background between runs is skipped a machine word at a time, which took it to 38. The
tracker's buffers are allocated once at construction, so its per-frame path allocates
nothing. And the road-geometry read-out builds its spline once for both curvature and
positions rather than once for each, which halved the downstream cost from 165 to 97
microseconds; the moment solve dominates everything after the mask.

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

Four suites run: `curvature_golden`, `geometry_golden`, `pipeline_golden` and
`control_golden`. All read
fixtures from `test/golden/`, generated on the Python side:

```
python -m scripts.export_golden_vectors      # curvature
python -m scripts.export_geometry_vectors    # projection and road geometry
python -m scripts.export_pipeline_vectors infer.source=<clip-dir>   # mask to centreline
python -m scripts.export_control_vectors                           # filter and MPC
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
