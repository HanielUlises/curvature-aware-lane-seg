"""Image/mask transforms for training and evaluation.

Two concerns drive this module, both from the project's curvature focus:

1. **Sky crop before resize, aspect-preserving.** The sky above the horizon is
   uninformative; it is cropped at native resolution *before* the resize to the
   model input size. Critically, the crop restores the target aspect ratio (by a
   symmetric width crop) so the subsequent resize is **isotropic** — a uniform
   scale that preserves curvature. A naive "crop the top, then resize to 512x288"
   is *anisotropic* and silently distorts the curvature the stratification bins
   on; it is rejected for that reason. :func:`preprocess_geometry` is applied
   identically to the image at load time and to the mask at cache time, keeping
   them pixel-aligned.

2. **Curvature-safe augmentation only.** Random geometric augmentations must
   preserve curvature or the stratification label no longer describes the sample.
   Permitted: horizontal flip (mirrors |kappa|), and rigid affine — rotation and
   translation with **scale fixed at 1 and zero shear**. Forbidden: perspective,
   elastic/grid/optical distortion, shear, and anisotropic scale — all change
   curvature. Photometric augmentations do not touch geometry and are unrestricted.
   :func:`assert_curvature_safe` enforces the whitelist.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
from albumentations.pytorch import ToTensorV2

DEFAULT_TARGET_SIZE: tuple[int, int] = (512, 288)
# Fraction of image height cropped from the top as sky. Kept conservative: the
# far field near the horizon is where lanes converge and where curvature is most
# informative, so an aggressive crop would remove the signal the project needs.
DEFAULT_SKY_FRAC: float = 0.30

# ImageNet statistics (encoder is an ImageNet-pretrained timm ResNet-18).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Rigid-only augmentation limits.
_ROTATE_DEG = 5.0
_TRANSLATE_FRAC = 0.04

# Geometric transforms that change curvature and must never appear in a pipeline.
_FORBIDDEN_TRANSFORMS: tuple[type, ...] = (
    A.Perspective,
    A.ElasticTransform,
    A.GridDistortion,
    A.OpticalDistortion,
    A.PiecewiseAffine,
    A.RandomResizedCrop,
    A.RandomScale,
    A.ShiftScaleRotate,
)

Array = npt.NDArray[np.uint8]


def preprocess_geometry(
    array: Array,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    sky_frac: float = DEFAULT_SKY_FRAC,
    interpolation: int = cv2.INTER_AREA,
) -> Array:
    """Sky-crop (aspect-preserving) then isotropically resize to the target.

    Applied to both image (at load) and mask (at cache) so they stay aligned.
    The crop removes the top ``sky_frac`` of height, then trims the longer axis
    so the remaining region matches the target aspect ratio; the final resize is
    therefore a uniform scale that preserves curvature.

    Args:
        array: ``(H, W)`` mask or ``(H, W, C)`` image, native resolution.
        target_size: Output ``(width, height)``.
        sky_frac: Fraction of height to remove from the top.
        interpolation: OpenCV interpolation (use ``INTER_NEAREST`` for masks).

    Returns:
        The cropped-and-resized array at ``(target_height, target_width[, C])``.
    """
    if not 0.0 <= sky_frac < 1.0:
        raise ValueError(f"sky_frac must be in [0, 1), got {sky_frac}")
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    height, width = array.shape[:2]
    top = min(int(round(height * sky_frac)), height - 1)
    cropped = array[top:, :]
    crop_h = cropped.shape[0]

    cur_ratio = width / crop_h
    if cur_ratio > target_ratio:
        # Too wide: symmetric width crop.
        new_w = int(round(crop_h * target_ratio))
        x0 = (width - new_w) // 2
        cropped = cropped[:, x0 : x0 + new_w]
    elif cur_ratio < target_ratio:
        # Too tall: crop from the top (keep the road at the bottom).
        new_h = int(round(width / target_ratio))
        cropped = cropped[crop_h - new_h :, :]

    return cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)


def build_train_transform(
    rotate_deg: float = _ROTATE_DEG,
    translate_frac: float = _TRANSLATE_FRAC,
) -> A.Compose:
    """Training augmentation pipeline (curvature-safe geometry + photometric).

    Operates on already-preprocessed target-resolution image+mask pairs. Geometry
    is limited to horizontal flip and rigid affine (scale=1, shear=0); photometric
    augmentations are unrestricted.
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=1.0,
                shear=0.0,
                rotate=(-rotate_deg, rotate_deg),
                translate_percent={"x": (-translate_frac, translate_frac),
                                   "y": (-translate_frac, translate_frac)},
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    assert_curvature_safe(transform)
    return transform


def build_eval_transform() -> A.Compose:
    """Deterministic evaluation pipeline: normalize + to-tensor, no augmentation."""
    return A.Compose(
        [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def assert_curvature_safe(transform: A.Compose) -> None:
    """Raise if a pipeline contains a curvature-distorting geometric transform.

    Checks for forbidden transform classes and for any ``Affine`` configured with
    non-unit scale or non-zero shear (which would break the rigid-only contract).

    Raises:
        ValueError: If a forbidden or misconfigured transform is present.
    """
    for t in transform.transforms:
        if isinstance(t, _FORBIDDEN_TRANSFORMS):
            raise ValueError(
                f"Curvature-distorting transform {type(t).__name__} is not allowed "
                f"in this pipeline (it changes kappa relative to the stratification label)."
            )
        if isinstance(t, A.Affine):
            _assert_rigid_affine(t)


def _assert_rigid_affine(affine: A.Affine) -> None:
    """Ensure an Affine is rigid: scale == 1 everywhere and shear == 0."""

    def _bounds(spec: object) -> list[float]:
        # albumentations normalizes scale/shear to {"x": (lo, hi), "y": (lo, hi)}.
        if isinstance(spec, dict):
            vals: list[float] = []
            for pair in spec.values():
                vals.extend(pair if isinstance(pair, (tuple, list)) else [pair])
            return [float(v) for v in vals]
        if isinstance(spec, (tuple, list)):
            return [float(v) for v in spec]
        return [float(spec)]

    if any(abs(v - 1.0) > 1e-9 for v in _bounds(affine.scale)):
        raise ValueError("Affine scale must be exactly 1.0 (uniform scale changes curvature).")
    if any(abs(v) > 1e-9 for v in _bounds(affine.shear)):
        raise ValueError("Affine shear must be exactly 0.0 (shear changes curvature).")
