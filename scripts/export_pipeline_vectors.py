"""Export golden vectors for the mask-to-centreline pipeline.

The curvature, projection and road-geometry stages are pinned by synthetic and
single-frame fixtures. The two stages added last, mask decomposition and boundary
tracking, are stateful and only meaningful over a sequence, so they are pinned by a run
of **real predicted masks** together with the centreline the Python reference produces
from each.

Masks are stored run-length encoded, which on a lane mask is about a hundredth of the
raw bitmap and keeps the fixture small enough to live in the repository. Each frame's
line is ``row_start count ...`` pairs over the flattened mask.

Run, after producing masks with a segmenter:
    python -m scripts.export_pipeline_vectors infer.source=<clip-dir>
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.subset import read_manifest
from src.data.transforms import build_eval_transform
from src.geometry.centerline import extract_lane_polylines
from src.geometry.lane_tracker import EgoBoundaryTracker
from src.models.lane_segmenter import LaneSegmenter
from scripts.infer_sequence import _center_crop_aspect, _frames_from_source, _predict
from scripts.infer_video import _resolve_ckpt

_TXT_DIR = Path(os.environ.get("PIPELINE_OUT_DIR", "deploy/test/golden/pipeline"))
_JSON_DIR = Path(os.environ.get("PIPELINE_JSON_DIR", "tests/golden/pipeline"))


def _rle(mask: np.ndarray) -> list[int]:
    """Run-length encode a binary mask as (start, length) pairs of set runs."""
    flat = (mask.reshape(-1) > 0).astype(np.uint8)
    if flat.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    starts, ends = edges[0::2], edges[1::2]
    out: list[int] = []
    for s, e in zip(starts, ends):
        out.extend((int(s), int(e - s)))
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _TXT_DIR.mkdir(parents=True, exist_ok=True)
    _JSON_DIR.mkdir(parents=True, exist_ok=True)

    target = (512, 288)
    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LaneSegmenter.load_from_checkpoint(
        str(_resolve_ckpt(cfg)), bin_edges=meta["bin_edges"], map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

    tracker = EgoBoundaryTracker(*target)
    frames: list[dict] = []
    limit = int(cfg.infer.get("max_frames") or 40)
    for i, rgb in enumerate(_frames_from_source(Path(cfg.infer.source), limit)):
        rgb = _center_crop_aspect(rgb, target)
        _, pred = _predict(
            model, rgb, transform, target, cfg.data.sky_frac, cfg.infer.tta, device
        )
        polys = extract_lane_polylines(pred)
        centre = tracker.centerline(tracker.update(polys))
        frames.append(
            {
                "index": i,
                "rle": _rle(pred),
                "num_polylines": len(polys),
                "centerline": [] if centre is None else centre.tolist(),
            }
        )

    payload = {
        "width": target[0],
        "height": target[1],
        "frames": frames,
        "tolerance_px": 0.5,
    }
    (_JSON_DIR / "sequence.json").write_text(json.dumps(payload))

    lines = [f"width {target[0]}", f"height {target[1]}",
             f"tolerance {payload['tolerance_px']}", f"frames {len(frames)}"]
    for f in frames:
        lines.append(f"frame {f['index']} polylines {f['num_polylines']}")
        lines.append(f"rle {len(f['rle'])} " + " ".join(str(v) for v in f["rle"]))
        pts = f["centerline"]
        lines.append(f"centerline {len(pts)}")
        lines.extend(f"{x!r} {y!r}" for x, y in pts)
    (_TXT_DIR / "sequence.txt").write_text("\n".join(lines) + "\n")

    detected = sum(1 for f in frames if f["centerline"])
    size_kb = (_TXT_DIR / "sequence.txt").stat().st_size / 1024
    print(f"[pipeline] {len(frames)} frames, {detected} with a centreline, "
          f"{size_kb:.0f} kB -> {_TXT_DIR/'sequence.txt'}")


if __name__ == "__main__":
    main()
