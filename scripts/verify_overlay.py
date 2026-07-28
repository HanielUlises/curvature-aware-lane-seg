"""Render lane-mask overlays for visual verification of the rasterization pipeline.

This is the verification gate for the data-preprocessing phase: it draws each
rasterized mask on top of its source image and writes the composites to disk for
human inspection. Misaligned masks caught here look, downstream, like a model
that will not converge.

Only frames matching the target aspect ratio are sampled (odd-aspect CurveLanes
frames are excluded from the pipeline). Images and masks are both rendered at the
target resolution the model actually consumes.

Usage
-----
    python -m scripts.verify_overlay \
        --images-dir data/raw/Curvelanes/train/images \
        --labels-dir data/raw/Curvelanes/train/labels \
        --out-dir viz/overlays --n 12 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.data.curvelanes import build_frame, index_split
from src.data.rasterize import (
    DEFAULT_STROKE_PX,
    DEFAULT_TARGET_SIZE,
    is_target_aspect,
    rasterize_frame,
)

_OVERLAY_COLOR = (0, 0, 255)  # BGR red
_OVERLAY_ALPHA = 0.5


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Composite a binary mask over an image, both at target resolution."""
    target_w, target_h = target_size
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tint = np.zeros_like(resized)
    tint[mask > 0] = _OVERLAY_COLOR
    blended = cv2.addWeighted(resized, 1.0, tint, _OVERLAY_ALPHA, 0.0)
    # Keep a hard 1px core so thin lanes stay visible under the alpha blend.
    blended[mask > 0] = (
        (1 - _OVERLAY_ALPHA) * blended[mask > 0] + _OVERLAY_ALPHA * np.array(_OVERLAY_COLOR)
    ).astype(np.uint8)
    return blended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12, help="Number of samples to render.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stroke", type=int, default=DEFAULT_STROKE_PX)
    parser.add_argument(
        "--target",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=list(DEFAULT_TARGET_SIZE),
    )
    args = parser.parse_args(argv)

    target_size = (args.target[0], args.target[1])
    index = index_split(args.images_dir, args.labels_dir)
    if index.images_without_label or index.labels_without_image:
        print(
            f"[warn] {len(index.images_without_label)} images without labels, "
            f"{len(index.labels_without_image)} labels without images"
        )
    if not index.pairs:
        print("[fail] no image/label pairs found")
        return 1

    rng = random.Random(args.seed)
    pairs = index.pairs[:]
    rng.shuffle(pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_aspect = 0
    for image_path, label_path in pairs:
        if written >= args.n:
            break
        frame = build_frame(image_path, label_path)
        if not is_target_aspect(frame.width, frame.height, target_size):
            skipped_aspect += 1
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[warn] could not read image {image_path}")
            continue
        mask = rasterize_frame(frame, target_size=target_size, stroke_px=args.stroke)
        composite = _overlay(image_bgr, mask, target_size)
        out_path = args.out_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(out_path), composite)
        written += 1
        print(f"[ok  ] {out_path}  lanes={frame.num_lanes}  native={frame.width}x{frame.height}")

    print(
        f"[done] wrote {written} overlays to {args.out_dir} "
        f"(skipped {skipped_aspect} odd-aspect frames)"
    )
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
