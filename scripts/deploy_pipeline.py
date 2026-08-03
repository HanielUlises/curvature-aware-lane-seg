"""Inference and control on real footage, with the whole pipeline running in C++.

The deployment shape of this project: a camera frame goes in and a steering command
comes out, and every stage between them — preprocessing, the network, decomposition,
tracking, projection, road geometry, filtering, the controller — runs in the C++ library
that ships to the vehicle, reached through ``src.native``. Python opens the file and
prints the summary.

That was not true until recently. The geometry ran in C++ while the network ran in
PyTorch, so the "deployed" pipeline re-entered Python for the stage that costs most of
the frame, and the reported latency left out the preprocessing entirely because the
preprocessing happened before the clock started. Both are fixed here: the split below
covers the whole frame, and it adds up.

    python -m scripts.deploy_pipeline infer.source=<clip-dir-or-video> \\
        [+onnx=<model.onnx>] [+backend=tensorrt] [ipm.calibration_file=<calibration.json>]

The C++ tool next to it does the same thing with no Python at all:

    deploy/build/run_infer --model <onnx> --source <clip> --calibration <json>
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

from src import native
from scripts.infer_sequence import _frames_from_source


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(p / 100.0 * (len(ordered) - 1)), len(ordered) - 1)]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not native.segmenter_available():
        raise RuntimeError(
            "this script needs the native pipeline: "
            f"{native.why_unavailable() or 'rebuild with -DONNXRUNTIME_ROOT=<dir>'}"
        )
    if not cfg.infer.source:
        raise ValueError("set infer.source=<image-dir-or-video-file>")

    target = tuple(cfg.data.target_size)
    out_root = Path(cfg.paths.output_root) / "export"
    model_path = Path(cfg.get("onnx", out_root / "lane_segmenter.onnx"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist; run scripts.export_onnx first"
        )

    calibration = None
    calib_path = cfg.get("ipm", {}).get("calibration_file", None) or (
        Path(cfg.paths.output_root) / "calibration.json"
    )
    if Path(calib_path).exists():
        d = json.loads(Path(calib_path).read_text())
        calibration = tuple(
            d[k] for k in ("fx", "fy", "cx", "cy", "height_m", "pitch_rad", "yaw_rad")
        )
        print(f"calibration: {calib_path}")
    else:
        print("no calibration: the metric outputs and the steering command are skipped")

    pipeline = native.NativePipeline(
        model_path, target[0], target[1], calibration,
        backend=str(cfg.get("backend", "auto")),
        engine_cache_dir=out_root / "trt_cache",
        threshold=float(cfg.infer.get("threshold", 0.5)),
        sky_frac=float(cfg.data.sky_frac),
    )
    speed = float(cfg.infer.get("speed_mps", 15.0))

    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else Path("results/deploy")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "chain.csv"

    pre_us: list[float] = []
    net_us: list[float] = []
    chain_us: list[float] = []
    total_us: list[float] = []
    detected = 0
    rows = []

    for index, rgb in enumerate(
        _frames_from_source(Path(cfg.infer.source), cfg.infer.get("max_frames", None))
    ):
        # Handed over at native resolution: the crop, resize and normalization are part
        # of the deployed path and are timed as part of it.
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        r = pipeline.process(frame, speed)
        whole = (time.perf_counter() - t0) * 1e6

        t = pipeline.timings_us
        pre_us.append(t["preprocess"])
        net_us.append(t["network"])
        chain_us.append(t["chain"])
        total_us.append(whole)
        detected += 1 if r.has_centerline else 0
        rows.append({
            "frame": index,
            "detected": int(r.has_centerline),
            "offset_m": round(r.lateral_offset_m, 4),
            "heading_deg": round(math.degrees(r.heading_error_rad), 3),
            "curvature_1pm": round(r.curvature_1pm, 6),
            "steer_deg": round(math.degrees(r.steer_rad), 3),
            "saturated": int(r.saturated),
            "coasting": int(r.coasting_frames),
            "preprocess_us": round(t["preprocess"], 1),
            "network_us": round(t["network"], 1),
            "chain_us": round(t["chain"], 1),
        })

    if not rows:
        raise RuntimeError(f"no frames read from {cfg.infer.source}")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    mean = lambda v: sum(v) / n  # noqa: E731
    total = mean(total_us)
    print(f"\n{n} frames at {target[0]}x{target[1]}, everything in C++")
    print(f"  ego lane on            {detected} frames ({100.0 * detected / n:.1f}%)")
    print(f"  preprocess             {mean(pre_us):8.1f} us/frame  "
          f"(p95 {_percentile(pre_us, 95):.0f})")
    print(f"  network                {mean(net_us):8.1f} us/frame  "
          f"(p95 {_percentile(net_us, 95):.0f})")
    print(f"  geometry chain         {mean(chain_us):8.1f} us/frame  "
          f"(p95 {_percentile(chain_us, 95):.0f})")
    print(f"  whole frame            {total:8.1f} us/frame  "
          f"(p95 {_percentile(total_us, 95):.0f}) = {1e6 / total:.0f} fps")
    print(f"  chain share of budget  {100.0 * mean(chain_us) / total:.1f}%")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
