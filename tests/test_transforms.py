"""Tests for the training/eval transform pipelines and the curvature-safe guard."""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pytest

from src.data.transforms import (
    assert_curvature_safe,
    build_eval_transform,
    build_train_transform,
    preprocess_geometry,
)


def test_preprocess_geometry_output_shape_and_aspect() -> None:
    img = np.zeros((1440, 2560, 3), dtype=np.uint8)
    out = preprocess_geometry(img, target_size=(512, 288), sky_frac=0.3)
    assert out.shape == (288, 512, 3)


def test_preprocess_geometry_is_isotropic() -> None:
    # A drawn circle must remain a circle (equal x/y scale), not an ellipse.
    img = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cv2.circle(img, (1280, 1000), 200, (255, 255, 255), 3)
    out = preprocess_geometry(img, target_size=(512, 288), sky_frac=0.3)
    gray = out[..., 0]
    ys, xs = np.nonzero(gray)
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    # Isotropic resize keeps the circle's aspect ~1 (allow rounding slack).
    assert abs(span_x - span_y) <= 3


def test_preprocess_geometry_keeps_mask_and_image_aligned() -> None:
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    mask = np.zeros((900, 1600), dtype=np.uint8)
    cv2.line(img, (100, 800), (1500, 400), (255, 255, 255), 6)
    cv2.line(mask, (100, 800), (1500, 400), 255, 6)
    out_img = preprocess_geometry(img, (512, 288), 0.3, cv2.INTER_AREA)
    out_mask = preprocess_geometry(mask, (512, 288), 0.3, cv2.INTER_NEAREST)
    # Foreground of image and mask should overlap strongly (same geometry applied).
    img_fg = out_img[..., 0] > 40
    mask_fg = out_mask > 0
    inter = np.logical_and(img_fg, mask_fg).sum()
    assert inter > 0 and inter >= 0.4 * mask_fg.sum()


def test_train_transform_produces_tensors() -> None:
    import torch

    tf = build_train_transform()
    img = (np.random.rand(288, 512, 3) * 255).astype(np.uint8)
    mask = (np.random.rand(288, 512) > 0.5).astype(np.uint8)
    out = tf(image=img, mask=mask)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 288, 512)
    assert out["mask"].shape == (288, 512)


def test_eval_transform_is_deterministic() -> None:
    tf = build_eval_transform()
    img = (np.random.rand(288, 512, 3) * 255).astype(np.uint8)
    a = tf(image=img.copy())["image"]
    b = tf(image=img.copy())["image"]
    assert np.allclose(a.numpy(), b.numpy())


def test_train_pipeline_passes_curvature_guard() -> None:
    # Building the train transform runs the guard internally; explicit re-check.
    assert_curvature_safe(build_train_transform())


def test_guard_rejects_perspective() -> None:
    bad = A.Compose([A.Perspective(p=1.0)])
    with pytest.raises(ValueError, match="not allowed"):
        assert_curvature_safe(bad)


def test_guard_rejects_nonrigid_affine() -> None:
    scaled = A.Compose([A.Affine(scale=1.2, p=1.0)])
    with pytest.raises(ValueError, match="scale must be exactly"):
        assert_curvature_safe(scaled)
    sheared = A.Compose([A.Affine(scale=1.0, shear=10, p=1.0)])
    with pytest.raises(ValueError, match="shear must be exactly"):
        assert_curvature_safe(sheared)
