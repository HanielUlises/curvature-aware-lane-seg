"""Python access to the deployed C++ pipeline.

The package used to carry a working implementation of every stage *and* a C++ port of
each. Keeping both was defensible while the port was being written, but it left the fast
path and the shipped path as different code, and the one that ran was the slow one.

This module makes the split explicit. The C++ is the implementation that runs; the pure
Python stays as the reference the golden fixtures are generated from and the port is
checked against. That is a test role, not a runtime one, and the distinction matters:
if Python became a thin wrapper with nothing behind it, the golden-vector contract would
be comparing the port against itself and would stop meaning anything.

That split was for a while only half true. The geometry ran in C++ while the network ran
in PyTorch, so the deployed pipeline re-entered Python for the stage that costs most of
the frame, and the preprocessing in front of it was not being counted at all.
:class:`NativePipeline` closes it: a frame goes in at its native resolution and a
steering command comes out, with the crop, the resize, the normalization, the network
and the whole geometry chain on the other side of one call.

Three levels, depending on how much of the path a caller owns:

    NativePipeline   frame -> steering command; what a deployment wants
    NativeSegmenter  frame -> mask; for checking the mask against PyTorch
    NativeChain      mask -> steering command; for a caller with its own network

Loaded through ``ctypes`` rather than a binding framework, so there is no build-time
dependency beyond the compiler the port already needs, and nothing to install.

    from src.native import NativePipeline, segmenter_available
    if segmenter_available():
        pipeline = NativePipeline("lane_segmenter.onnx", 512, 288, calibration)
        result = pipeline.process(frame_bgr, speed_mps=15.0)
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bump in lockstep with cp_abi_version in the shim, so a stale .so is refused rather
# than called with a mismatched struct layout.
REQUIRED_ABI = 3

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
        ("raw_lateral_offset_m", ctypes.c_double),
        ("raw_heading_error_rad", ctypes.c_double),
        ("raw_curvature_1pm", ctypes.c_double),
        ("preview_curvature_1pm", ctypes.c_double * 3),
        ("steer_rad", ctypes.c_double),
        ("steer_unsaturated_rad", ctypes.c_double),
        ("saturated", ctypes.c_int32),
        ("coasting_frames", ctypes.c_int32),
        ("resets", ctypes.c_int32),
    ]


class _SegmenterConfig(ctypes.Structure):
    """Mirrors cp_segmenter_config; field order and types must match exactly."""

    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("backend", ctypes.c_char_p),
        ("fp16", ctypes.c_int32),
        ("engine_cache_dir", ctypes.c_char_p),
        ("threshold", ctypes.c_double),
        ("sky_frac", ctypes.c_double),
    ]


@dataclass(frozen=True)
class FrameResult:
    """One frame of the chain, as plain Python."""

    has_centerline: bool
    has_geometry: bool
    lateral_offset_m: float
    heading_error_rad: float
    curvature_1pm: float
    raw_lateral_offset_m: float
    raw_heading_error_rad: float
    raw_curvature_1pm: float
    preview_curvature_1pm: tuple[float, float, float]
    steer_rad: float
    steer_unsaturated_rad: float
    saturated: bool
    coasting_frames: int
    resets: int


def _as_frame_result(r: _FrameResult) -> FrameResult:
    """Copy the C struct into a frozen Python one, since the C struct is reused."""
    return FrameResult(
        has_centerline=bool(r.has_centerline),
        has_geometry=bool(r.has_geometry),
        lateral_offset_m=r.lateral_offset_m,
        heading_error_rad=r.heading_error_rad,
        curvature_1pm=r.curvature_1pm,
        raw_lateral_offset_m=r.raw_lateral_offset_m,
        raw_heading_error_rad=r.raw_heading_error_rad,
        raw_curvature_1pm=r.raw_curvature_1pm,
        preview_curvature_1pm=tuple(r.preview_curvature_1pm),
        steer_rad=r.steer_rad,
        steer_unsaturated_rad=r.steer_unsaturated_rad,
        saturated=bool(r.saturated),
        coasting_frames=int(r.coasting_frames),
        resets=int(r.resets),
    )


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
    cfgp, charp = ctypes.POINTER(_SegmenterConfig), ctypes.c_char_p
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

    lib.cp_segmenter_available.restype = i32
    lib.cp_segmenter_create.argtypes = [cfgp, charp, i32]
    lib.cp_segmenter_create.restype = ctypes.c_void_p
    lib.cp_segmenter_destroy.argtypes = [ctypes.c_void_p]
    lib.cp_segmenter_run.argtypes = [ctypes.c_void_p, u8p, i32, i32, i32, i32, u8p]
    lib.cp_segmenter_run.restype = i32
    lib.cp_segmenter_backend.argtypes = [ctypes.c_void_p]
    lib.cp_segmenter_backend.restype = charp
    for name in ("cp_segmenter_last_preprocess_us", "cp_segmenter_last_network_us"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = d

    lib.cp_pipeline_create.argtypes = [cfgp, dp, charp, i32]
    lib.cp_pipeline_create.restype = ctypes.c_void_p
    lib.cp_pipeline_destroy.argtypes = [ctypes.c_void_p]
    lib.cp_pipeline_reset.argtypes = [ctypes.c_void_p]
    lib.cp_pipeline_process.argtypes = [ctypes.c_void_p, u8p, i32, i32, i32, i32, d,
                                        ctypes.POINTER(_FrameResult)]
    lib.cp_pipeline_process.restype = i32
    lib.cp_pipeline_centerline.argtypes = [ctypes.c_void_p, dp, i32]
    lib.cp_pipeline_centerline.restype = i32
    lib.cp_pipeline_boundary.argtypes = [ctypes.c_void_p, i32, dp, i32]
    lib.cp_pipeline_boundary.restype = i32
    lib.cp_pipeline_mask.argtypes = [ctypes.c_void_p]
    lib.cp_pipeline_mask.restype = u8p
    lib.cp_pipeline_source_roi.argtypes = [ctypes.c_void_p, i32p]
    for name in ("cp_pipeline_last_preprocess_us", "cp_pipeline_last_network_us",
                 "cp_pipeline_last_chain_us"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = d


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
        return _as_frame_result(self._result)

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


def segmenter_available() -> bool:
    """Whether this build of the library can run a network, not just the geometry."""
    lib = _load()
    return lib is not None and bool(lib.cp_segmenter_available())


def _build_config(model_path, width, height, backend, fp16, engine_cache_dir,
                  threshold, sky_frac) -> _SegmenterConfig:
    return _SegmenterConfig(
        model_path=str(model_path).encode(),
        width=int(width),
        height=int(height),
        backend=str(backend).encode(),
        fp16=1 if fp16 else 0,
        engine_cache_dir=(str(engine_cache_dir).encode()
                          if engine_cache_dir else None),
        threshold=float(threshold),
        sky_frac=float(sky_frac),
    )


class NativePipeline:
    """The whole deployed path: a camera frame in, a steering command out.

    The network runs in C++ too, through ONNX Runtime, so a frame never returns to
    Python between the pixels and the command. Python's remaining job is to hand over
    the frame and read the result, which is what a harness does.

    Args:
        model_path: The ``.onnx`` written by ``scripts/export_onnx.py``.
        width: Network input width; must match what the model was exported at.
        height: Network input height.
        calibration: ``(fx, fy, cx, cy, height_m, pitch_rad, yaw_rad)``, or ``None`` to
            run without a ground plane.
        backend: ``auto``, ``tensorrt``, ``cuda`` or ``cpu``. ``auto`` takes the first
            of those three that can build a session.
        engine_cache_dir: Where TensorRT keeps its built engine. Without it the engine
            is rebuilt on every construction, which takes minutes.

    Raises:
        RuntimeError: If the library is unavailable, has no segmenter, or the model
            cannot be loaded on any backend.
    """

    _MAX_POINTS = 512

    def __init__(self, model_path, width: int, height: int, calibration=None,
                 backend: str = "auto", fp16: bool = True, engine_cache_dir=None,
                 threshold: float = 0.5, sky_frac: float = 0.30) -> None:
        lib = _load()
        if lib is None:
            raise RuntimeError(why_unavailable())
        if not lib.cp_segmenter_available():
            raise RuntimeError(
                "this build of libcurvature_port_c has no segmenter; rebuild with "
                "-DONNXRUNTIME_ROOT=<dir>"
            )
        self._lib = lib
        self.width, self.height = int(width), int(height)
        config = _build_config(model_path, width, height, backend, fp16,
                               engine_cache_dir, threshold, sky_frac)
        calib_ptr = None
        if calibration is not None:
            arr = (ctypes.c_double * 7)(*[float(v) for v in calibration])
            calib_ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
            self._calib_keepalive = arr
        err = ctypes.create_string_buffer(512)
        self._handle = lib.cp_pipeline_create(ctypes.byref(config), calib_ptr, err, 512)
        if not self._handle:
            raise RuntimeError(
                f"cannot load {model_path}: {err.value.decode(errors='replace')}"
            )
        if err.value:
            # Non-fatal, but the caller asked for metric output and is not getting it.
            print(f"native pipeline: {err.value.decode(errors='replace')}")
        self._buf = (ctypes.c_double * (2 * self._MAX_POINTS))()
        self._roi = (ctypes.c_int32 * 4)()
        self._result = _FrameResult()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.cp_pipeline_destroy(handle)
            self._handle = None

    def reset(self) -> None:
        """Clear the tracker and filter state."""
        self._lib.cp_pipeline_reset(self._handle)

    def process(self, frame: np.ndarray, speed_mps: float = 15.0,
                bgr: bool = True) -> FrameResult:
        """Run one frame, from native resolution to steering command.

        Args:
            frame: ``(H, W, 3)`` 8-bit image at its native resolution. The crop, resize
                and normalization happen in C++; do not preprocess it here.
            speed_mps: Forward speed handed to the controller.
            bgr: Channel order. ``cv2.imread`` and ``cv2.VideoCapture`` give BGR, so
                this defaults to what a caller most likely has.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) image, got {frame.shape}")
        buf = frame if (frame.dtype == np.uint8 and frame.flags["C_CONTIGUOUS"]) \
            else np.ascontiguousarray(frame, dtype=np.uint8)
        ok = self._lib.cp_pipeline_process(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            buf.shape[1], buf.shape[0], buf.strides[0], 1 if bgr else 0,
            ctypes.c_double(float(speed_mps)),
            ctypes.byref(self._result),
        )
        if not ok:
            raise RuntimeError("the native pipeline failed on this frame")
        return _as_frame_result(self._result)

    def mask(self) -> np.ndarray:
        """The last frame's binary mask, as a ``(height, width)`` array.

        A copy: the pipeline reuses its buffer on the next call.
        """
        ptr = self._lib.cp_pipeline_mask(self._handle)
        size = self.height * self.width
        return np.ctypeslib.as_array(ptr, shape=(size,)).reshape(
            self.height, self.width).copy()

    def source_roi(self) -> tuple[int, int, int, int]:
        """The region of the source frame the network input came from: x, y, w, h."""
        self._lib.cp_pipeline_source_roi(self._handle, self._roi)
        return tuple(int(v) for v in self._roi)

    def centerline(self) -> np.ndarray:
        """The current ego centreline as ``(N, 2)`` image points."""
        n = self._lib.cp_pipeline_centerline(self._handle, self._buf, self._MAX_POINTS)
        return np.ctypeslib.as_array(self._buf)[: 2 * n].reshape(-1, 2).copy()

    def boundary(self, side: str) -> np.ndarray:
        """A tracked ego boundary as ``(N, 2)`` image points; side is 'left' or 'right'."""
        idx = 0 if side == "left" else 1
        n = self._lib.cp_pipeline_boundary(self._handle, idx, self._buf,
                                           self._MAX_POINTS)
        return np.ctypeslib.as_array(self._buf)[: 2 * n].reshape(-1, 2).copy()

    @property
    def timings_us(self) -> dict[str, float]:
        """Microseconds in the last frame: preprocess, network, chain."""
        return {
            "preprocess": self._lib.cp_pipeline_last_preprocess_us(self._handle),
            "network": self._lib.cp_pipeline_last_network_us(self._handle),
            "chain": self._lib.cp_pipeline_last_chain_us(self._handle),
        }


class NativeSegmenter:
    """Just the network stage: a frame in, a binary mask out.

    :class:`NativePipeline` is what a deployment wants. This exists for the case where
    only the mask is of interest, most usefully checking it against the PyTorch model
    the ONNX file was exported from.
    """

    def __init__(self, model_path, width: int, height: int, backend: str = "auto",
                 fp16: bool = True, engine_cache_dir=None, threshold: float = 0.5,
                 sky_frac: float = 0.30) -> None:
        lib = _load()
        if lib is None:
            raise RuntimeError(why_unavailable())
        if not lib.cp_segmenter_available():
            raise RuntimeError("this build of libcurvature_port_c has no segmenter")
        self._lib = lib
        self.width, self.height = int(width), int(height)
        config = _build_config(model_path, width, height, backend, fp16,
                               engine_cache_dir, threshold, sky_frac)
        err = ctypes.create_string_buffer(512)
        self._handle = lib.cp_segmenter_create(ctypes.byref(config), err, 512)
        if not self._handle:
            raise RuntimeError(
                f"cannot load {model_path}: {err.value.decode(errors='replace')}"
            )
        self._mask = np.zeros((self.height, self.width), dtype=np.uint8)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.cp_segmenter_destroy(handle)
            self._handle = None

    @property
    def backend(self) -> str:
        """The backend that actually built the session."""
        return self._lib.cp_segmenter_backend(self._handle).decode()

    def run(self, frame: np.ndarray, bgr: bool = True) -> np.ndarray:
        """Preprocess and segment one native-resolution frame.

        Returns a ``(height, width)`` mask. The buffer is reused between calls.
        """
        buf = frame if (frame.dtype == np.uint8 and frame.flags["C_CONTIGUOUS"]) \
            else np.ascontiguousarray(frame, dtype=np.uint8)
        ok = self._lib.cp_segmenter_run(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            buf.shape[1], buf.shape[0], buf.strides[0], 1 if bgr else 0,
            self._mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if not ok:
            raise RuntimeError("the native segmenter failed on this frame")
        return self._mask

    @property
    def timings_us(self) -> dict[str, float]:
        """Microseconds in the last run: preprocess, network."""
        return {
            "preprocess": self._lib.cp_segmenter_last_preprocess_us(self._handle),
            "network": self._lib.cp_segmenter_last_network_us(self._handle),
        }


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
