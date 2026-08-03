"""Does the deployed path see the same road the reference does?

The geometry port has golden vectors, and the crop has its own (test_preprocess). This
covers the join between them: the C++ segmenter preprocesses a frame and runs the ONNX
graph, the Python reference preprocesses the same frame and runs the PyTorch model, and
the two masks are compared. Everything the chain does afterwards is already under
contract, so if the masks agree the deployed pipeline and the reference are the same
pipeline.

Agreement is measured on the mask rather than on the logits deliberately. The chain
consumes a thresholded mask; fp16 moves logits far more than any of the preprocessing
differences do, and a logit tolerance would either be so loose it proves nothing or so
tight it fails on a difference the mask cannot see. What matters is whether a pixel
changed sides.

The two remaining sources of disagreement are worth naming, because neither is a port
bug. The TensorRT engine is fp16 and the reference is fp32, and the normalization is
fused differently by a bit or two in the mantissa. Both move pixels only where the logit
was already at the threshold, which is the lane's own boundary.

    python -m scripts.check_native_parity infer.source=<clip-dir-or-video> \\
        infer.ckpt=<checkpoint> [infer.max_frames=50] [backend=tensorrt]
"""

from __future__ import annotations

from pathlib import Path

import cv2
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not native.segmenter_available():
        raise RuntimeError(
            "this build has no segmenter: "
            f"{native.why_unavailable() or 'rebuild with -DONNXRUNTIME_ROOT=<dir>'}"
        )
    if not cfg.infer.source:
        raise ValueError("set infer.source=<image-dir-or-video-file>")

    target = tuple(cfg.data.target_size)
    threshold = float(cfg.infer.get("threshold", 0.5))
    sky = float(cfg.data.sky_frac)
    out_dir = Path(cfg.paths.output_root) / "export"
    model_path = Path(cfg.get("onnx", out_dir / "lane_segmenter.onnx"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist; run scripts.export_onnx first"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, meta = read_manifest(Path(cfg.paths.output_root) / "manifests" / "val.json")
    reference = LaneSegmenter.load_from_checkpoint(
        str(_resolve_ckpt(cfg)), bin_edges=meta["bin_edges"], map_location=device
    ).eval().to(device)
    transform = build_eval_transform()

    backend = str(cfg.get("backend", "auto"))
    segmenter = native.NativeSegmenter(
        model_path, target[0], target[1], backend=backend,
        engine_cache_dir=out_dir / "trt_cache", threshold=threshold, sky_frac=sky,
    )
    print(f"reference: PyTorch on {device}")
    print(f"deployed:  {model_path.name} on {segmenter.backend}\n")

    agreements: list[float] = []
    disagreements: list[int] = []
    lane_iou: list[float] = []
    max_frames = cfg.infer.get("max_frames", None) or 50

    for index, rgb in enumerate(_frames_from_source(Path(cfg.infer.source), max_frames)):
        rgb = _center_crop_aspect(rgb, target)

        # Reference: preprocess in Python, run PyTorch.
        image = preprocess_geometry(rgb, target, sky)
        tensor = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(reference(tensor))
        want = (prob[0, 0].cpu().numpy() >= threshold).astype(np.uint8)

        # Deployed: hand the native-resolution frame over and let C++ do all of it.
        got = segmenter.run(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bgr=True).copy()

        same = int(np.sum(want == got))
        agreements.append(same / want.size)
        disagreements.append(want.size - same)
        # Agreement over the whole mask is flattered by the background, which is most of
        # it. The lane class is what the chain reads, so it is scored on its own.
        union = int(np.sum((want | got) > 0))
        lane_iou.append(1.0 if union == 0 else int(np.sum((want & got) > 0)) / union)

    if not agreements:
        raise RuntimeError(f"no frames read from {cfg.infer.source}")

    n = len(agreements)
    print(f"{n} frames at {target[0]}x{target[1]}")
    print(f"  mask agreement      mean {100 * float(np.mean(agreements)):.4f}%  "
          f"worst {100 * float(np.min(agreements)):.4f}%")
    print(f"  lane-class IoU      mean {float(np.mean(lane_iou)):.5f}  "
          f"worst {float(np.min(lane_iou)):.5f}")
    print(f"  differing pixels    mean {float(np.mean(disagreements)):.1f}  "
          f"worst {int(np.max(disagreements))}  of {target[0] * target[1]}")
    timings = segmenter.timings_us
    print(f"\n  last frame: preprocess {timings['preprocess']:.0f} us, "
          f"network {timings['network']:.0f} us")


if __name__ == "__main__":
    main()
