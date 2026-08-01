"""Python access to the deployed C++ chain.

The package used to carry a working implementation of every stage *and* a C++ port of
each. Keeping both was defensible while the port was being written, but it left the fast
path and the shipped path as different code, and the one that ran was the slow one.

This module makes the split explicit. The C++ is the implementation that runs; the pure
Python stays as the reference the golden fixtures are generated from and the port is
checked against. That is a test role, not a runtime one, and the distinction matters:
if Python became a thin wrapper with nothing behind it, the golden-vector contract would
be comparing the port against itself and would stop meaning anything.

Loaded through ``ctypes`` rather than a binding framework, so there is no build-time
dependency beyond the compiler the port already needs, and nothing to install.

    from src.native import NativeChain, available
    if available():
        chain = NativeChain(512, 288, calibration)
        result = chain.process(mask, speed_mps=15.0)
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bump in lockstep with cp_abi_version in the shim, so a stale .so is refused rather
# than called with a mismatched struct layout.
REQUIRED_ABI = 1

_SEARCH = (
    "deploy/build/libcurvature_port_c.so",
    "deploy/build/Release/curvature_port_c.dll",
    "deploy/build/libcurvature_port_c.dylib",
)


class _FrameResult(ctypes.Structure):
    """Mirrors cp_frame_result; field order and types must match exactly."""

    _fields_ = [
        ("has_centerline", ctypes.c_int32),
        ("has_geometry", ctypes.c_int32),
        ("n_centerline", ctypes.c_int32),
        ("lateral_offset_m", ctypes.c_double),
        ("heading_error_rad", ctypes.c_double),
        ("curvature_1pm", ctypes.c_double),
        ("preview_curvature_1pm", ctypes.c_double * 3),
        ("steer_rad", ctypes.c_double),
        ("steer_unsaturated_rad", ctypes.c_double),
        ("saturated", ctypes.c_int32),
        ("coasting_frames", ctypes.c_int32),
        ("resets", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class FrameResult:
    """One frame of the chain, as plain Python."""

    has_centerline: bool
    has_geometry: bool
    lateral_offset_m: float
    heading_error_rad: float
    curvature_1pm: float
    preview_curvature_1pm: tuple[float, float, float]
    steer_rad: float
    steer_unsaturated_rad: float
    saturated: bool
    coasting_frames: int
    resets: int


_lib = None
_load_error: str | None = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    root = Path(__file__).resolve().parents[1]
    candidates = [Path(os.environ["CURVATURE_PORT_LIB"])] if "CURVATURE_PORT_LIB" in os.environ else []
    candidates += [root / p for p in _SEARCH]
    for path in candidates:
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:  # wrong architecture, missing Eigen symbols, ...
            _load_error = f"{path}: {exc}"
            continue
        lib.cp_abi_version.restype = ctypes.c_int32
        if lib.cp_abi_version() != REQUIRED_ABI:
            _load_error = (
                f"{path}: ABI {lib.cp_abi_version()}, this module needs {REQUIRED_ABI}; "
                "rebuild the shared library"
            )
            continue
        _bind(lib)
        _lib = lib
        return _lib
    if _load_error is None:
        _load_error = (
            "libcurvature_port_c not found; build it with "
            "`cmake -S deploy -B deploy/build && cmake --build deploy/build`"
        )
    return None


def _bind(lib) -> None:
    d, i32, u8p = ctypes.c_double, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8)
    dp, i32p = ctypes.POINTER(d), ctypes.POINTER(i32)
    lib.cp_chain_create.argtypes = [i32, i32, dp]
    lib.cp_chain_create.restype = ctypes.c_void_p
    lib.cp_chain_destroy.argtypes = [ctypes.c_void_p]
    lib.cp_chain_reset.argtypes = [ctypes.c_void_p]
    lib.cp_chain_process.argtypes = [ctypes.c_void_p, u8p, d,
                                     ctypes.POINTER(_FrameResult)]
    lib.cp_chain_centerline.argtypes = [ctypes.c_void_p, dp, i32]
    lib.cp_chain_centerline.restype = i32
    lib.cp_chain_boundary.argtypes = [ctypes.c_void_p, i32, dp, i32]
    lib.cp_chain_boundary.restype = i32
    lib.cp_extract_polylines.argtypes = [u8p, i32, i32, dp, i32p, i32, i32]
    lib.cp_extract_polylines.restype = i32


def available() -> bool:
    """Whether the shared library could be loaded."""
    return _load() is not None


def why_unavailable() -> str | None:
    """The reason :func:`available` returned False, for an actionable error message."""
    _load()
    return _load_error


class NativeChain:
    """The whole deployed chain: mask in, steering command out.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        calibration: ``(fx, fy, cx, cy, height_m, pitch_rad, yaw_rad)``, or ``None`` to
            run without a ground plane, in which case only the centreline is produced.

    Raises:
        RuntimeError: If the shared library is unavailable.
    """

    _MAX_POINTS = 512

    def __init__(self, width: int, height: int, calibration=None) -> None:
        lib = _load()
        if lib is None:
            raise RuntimeError(why_unavailable())
        self._lib = lib
        self.width, self.height = int(width), int(height)
        calib_ptr = None
        if calibration is not None:
            arr = (ctypes.c_double * 7)(*[float(v) for v in calibration])
            calib_ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
            self._calib_keepalive = arr
        self._handle = lib.cp_chain_create(self.width, self.height, calib_ptr)
        if not self._handle:
            raise RuntimeError("cp_chain_create failed")
        self._buf = (ctypes.c_double * (2 * self._MAX_POINTS))()
        self._result = _FrameResult()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.cp_chain_destroy(handle)
            self._handle = None

    def reset(self) -> None:
        """Clear the tracker and filter state."""
        self._lib.cp_chain_reset(self._handle)

    def process(self, mask: np.ndarray, speed_mps: float = 15.0) -> FrameResult:
        """Run one frame.

        Args:
            mask: ``(H, W)`` array, non-zero meaning lane. Converted to contiguous
                ``uint8`` if it is not already, which is the only copy on this path.
            speed_mps: Forward speed handed to the controller.
        """
        if mask.shape != (self.height, self.width):
            raise ValueError(
                f"expected a {self.height}x{self.width} mask, got {mask.shape}"
            )
        # Checking the flags is cheaper than calling ascontiguousarray unconditionally,
        # and a mask straight from the segmenter is already contiguous uint8.
        buf = mask if (mask.dtype == np.uint8 and mask.flags["C_CONTIGUOUS"]) \
            else np.ascontiguousarray(mask, dtype=np.uint8)
        self._lib.cp_chain_process(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_double(float(speed_mps)),
            ctypes.byref(self._result),
        )
        r = self._result
        return FrameResult(
            has_centerline=bool(r.has_centerline),
            has_geometry=bool(r.has_geometry),
            lateral_offset_m=r.lateral_offset_m,
            heading_error_rad=r.heading_error_rad,
            curvature_1pm=r.curvature_1pm,
            preview_curvature_1pm=tuple(r.preview_curvature_1pm),
            steer_rad=r.steer_rad,
            steer_unsaturated_rad=r.steer_unsaturated_rad,
            saturated=bool(r.saturated),
            coasting_frames=int(r.coasting_frames),
            resets=int(r.resets),
        )

    def process_into(self, mask: np.ndarray, speed_mps: float = 15.0):
        """Run one frame and return the raw result struct, avoiding a Python object.

        The struct is reused between calls, so its fields are only valid until the next
        one. Worth about ten per cent of the per-frame cost in a tight loop; use
        :meth:`process` unless the loop is the bottleneck.
        """
        buf = mask if (mask.dtype == np.uint8 and mask.flags["C_CONTIGUOUS"]) \
            else np.ascontiguousarray(mask, dtype=np.uint8)
        self._lib.cp_chain_process(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_double(float(speed_mps)),
            ctypes.byref(self._result),
        )
        return self._result

    def centerline(self) -> np.ndarray:
        """The current ego centreline as ``(N, 2)`` image points."""
        n = self._lib.cp_chain_centerline(self._handle, self._buf, self._MAX_POINTS)
        return np.ctypeslib.as_array(self._buf)[: 2 * n].reshape(-1, 2).copy()

    def boundary(self, side: str) -> np.ndarray:
        """A tracked ego boundary as ``(N, 2)`` image points; side is 'left' or 'right'."""
        idx = 0 if side == "left" else 1
        n = self._lib.cp_chain_boundary(self._handle, idx, self._buf, self._MAX_POINTS)
        return np.ctypeslib.as_array(self._buf)[: 2 * n].reshape(-1, 2).copy()


def extract_lane_polylines(mask: np.ndarray, max_polylines: int = 32,
                           max_points: int = 8192) -> list[np.ndarray]:
    """Stateless mask decomposition, matching src.geometry.centerline's function."""
    lib = _load()
    if lib is None:
        raise RuntimeError(why_unavailable())
    buf = np.ascontiguousarray(mask, dtype=np.uint8)
    xy = (ctypes.c_double * (2 * max_points))()
    counts = (ctypes.c_int32 * max_polylines)()
    n = lib.cp_extract_polylines(
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        buf.shape[1], buf.shape[0], xy, counts, max_polylines, max_points,
    )
    flat = np.ctypeslib.as_array(xy)
    out, off = [], 0
    for i in range(n):
        c = int(counts[i])
        out.append(flat[2 * off : 2 * (off + c)].reshape(-1, 2).copy())
        off += c
    return out
