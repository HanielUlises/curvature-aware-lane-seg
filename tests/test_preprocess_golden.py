"""Ties the preprocessing golden vectors to the code they are supposed to describe.

``deploy/test/golden/preprocess.txt`` says which region of a source frame the network
input is taken from, and the C++ test asserts the port computes the same regions. That
contract has a hole in it: the generator carries its own copy of the crop arithmetic, so
if ``preprocess_geometry`` changed and the vectors were not regenerated, both sides of
the C++ test would agree with each other and disagree with the model's actual
preprocessing. Nothing would fail, and the deployed pipeline would quietly be looking at
a different piece of road than the one it was trained on.

So this closes it from the other end, without duplicating the arithmetic a third time:
cropping a frame to the golden region and resizing must produce exactly what the real
preprocessing produces. If either drifts, this fails.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.transforms import DEFAULT_TARGET_SIZE, preprocess_geometry
from scripts.infer_sequence import _center_crop_aspect

GOLDEN = Path("deploy/test/golden/preprocess.txt")


def _regions():
    """The (src_w, src_h, sky_frac, x, y, w, h) rows of the golden file."""
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN} not generated; run scripts.export_preprocess_vectors")
    rows, target = [], None
    for line in GOLDEN.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "target":
            target = (int(fields[1]), int(fields[2]))
        elif fields[0] == "region":
            rows.append((int(fields[1]), int(fields[2]), float(fields[3]),
                         int(fields[4]), int(fields[5]), int(fields[6]),
                         int(fields[7])))
    return target, rows


def test_golden_target_matches_the_configured_input_size():
    target, rows = _regions()
    assert target == DEFAULT_TARGET_SIZE, (
        "the golden vectors were generated for a different network input size; "
        "regenerate them with scripts.export_preprocess_vectors"
    )
    assert rows, "no regions in the golden file"


def test_golden_regions_reproduce_the_reference_preprocessing():
    """Crop to the golden region and resize == what the reference pipeline computes."""
    target, rows = _regions()
    rng = np.random.default_rng(0)

    for src_w, src_h, sky, x, y, w, h in rows:
        # Structured noise rather than flat noise: INTER_AREA averages, so a frame with
        # no low-frequency content would give nearly the same output from a region that
        # is off by a pixel, and the test would pass while the region was wrong.
        image = np.zeros((src_h, src_w, 3), np.uint8)
        image[:, :, 0] = np.linspace(0, 255, src_w, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.linspace(0, 255, src_h, dtype=np.uint8)[:, None]
        image[:, :, 2] = rng.integers(0, 256, (src_h, src_w), dtype=np.uint8)

        want = preprocess_geometry(_center_crop_aspect(image, target), target, sky)
        got = cv2.resize(image[y:y + h, x:x + w], target,
                         interpolation=cv2.INTER_AREA)

        assert got.shape == want.shape
        assert np.array_equal(got, want), (
            f"region ({x}, {y}, {w}, {h}) does not reproduce the reference "
            f"preprocessing for a {src_w}x{src_h} frame at sky_frac {sky}; "
            "regenerate with scripts.export_preprocess_vectors"
        )
