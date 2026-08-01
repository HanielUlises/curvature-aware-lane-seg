"""Inference and control on real footage, with the geometry running in C++.

The deployment shape of this project: a segmenter produces a mask, and everything after
it — decomposition, tracking, projection, road geometry, filtering, the controller — runs
in the C++ library that ships to the vehicle, reached through ``src.native``. Python is
the harness, not the implementation.

Reports the split between the two costs, which is the number that matters when deciding
what to optimize next. On this machine the network dominates the chain by roughly two
orders of magnitude, so the geometry is not the bottleneck and further micro-optimizing it
would be wasted effort; that only became obvious once both were measured together.

    python -m scripts.deploy_pipeline infer.source=<clip-dir-or-video> \\
        infer.ckpt=<checkpoint> ipm.calibration_file=<calibration.json>
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src import native
from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform, preprocess_geometry
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_sequence import _center_crop_aspect, _frames_from_source
from scripts.infer_video import _resolve_ckpt


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(p / 100.0 * (len(ordered) - 1)), len(ordered) - 1)]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not native.available():
        raise RuntimeError(
            f"the native chain is required by this script: {native.why_unavailable()}"
        )
    if not cfg.infer.source:
        raise ValueError("set infer.source=<image-dir-or-video-file>")

    target = tuple(cfg.data.target_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    model = LaneSegmenter.load_from_checkpoint(
        str(_resolve_ckpt(cfg)), bin_edges=meta["bin_edges"], map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

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

    chain = native.NativeChain(target[0], target[1], calibration)
    speed = float(cfg.infer.get("speed_mps", 15.0))
    threshold = float(cfg.infer.get("threshold", 0.5))

    out_dir = Path(cfg.infer.out_dir) if cfg.infer.out_dir else Path("results/deploy")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "chain.csv"

    net_us: list[float] = []
    chain_us: list[float] = []
    detected = 0
    rows = []

    for index, rgb in enumerate(
        _frames_from_source(Path(cfg.infer.source), cfg.infer.get("max_frames", None))
    ):
        rgb = _center_crop_aspect(rgb, target)
        image = preprocess_geometry(rgb, target, cfg.data.sky_frac)
        tensor = transform(image=image)["image"].unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        mask = np.ascontiguousarray(
            (prob[0, 0].cpu().numpy() >= threshold).astype(np.uint8)
        )

        t2 = time.perf_counter()
        r = chain.process_into(mask, speed)
        t3 = time.perf_counter()

        net_us.append((t1 - t0) * 1e6)
        chain_us.append((t3 - t2) * 1e6)
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
            "net_us": round(net_us[-1], 1),
            "chain_us": round(chain_us[-1], 1),
        })

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    net_mean = sum(net_us) / n
    chain_mean = sum(chain_us) / n
    total = net_mean + chain_mean
    print(f"\n{n} frames at {target[0]}x{target[1]} on {device}")
    print(f"  ego lane on            {detected} frames ({100.0 * detected / n:.1f}%)")
    print(f"  network                {net_mean:8.1f} us/frame  "
          f"(p95 {_percentile(net_us, 95):.0f})")
    print(f"  geometry chain (C++)   {chain_mean:8.1f} us/frame  "
          f"(p95 {_percentile(chain_us, 95):.0f})")
    print(f"  total                  {total:8.1f} us/frame  "
          f"= {1e6 / total:.0f} fps")
    print(f"  chain share of budget  {100.0 * chain_mean / total:.1f}%")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
