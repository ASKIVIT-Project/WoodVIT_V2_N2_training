"""Augmentation (training) and resizing (evaluation) with albumentations.

Images are float32 (H, W, C) arrays in [0, 1]; the instance-id map is passed as a single uint16
mask and transformed with nearest-neighbour interpolation and fill value 0, so every instance is
transformed exactly like a separate binary mask would be. Images are resized bilinearly.

Training pipeline (all sensors): random resized crop to the target size, the eight symmetries of
the square, coarse dropout, Gaussian noise, and affine or perspective distortion. RGB additionally
gets colour jitter and random grey conversion; RGB and Thermo get brightness/contrast changes and
Gaussian or motion blur.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np


def train_transform(sensor: str, target_size: tuple[int, int], seed: int | None = None) -> A.Compose:
    """Training augmentation. With ``seed`` the sequence of random augmentations is reproducible
    (albumentations draws from its own generator, not from the global random state)."""
    h, w = target_size
    augs: list = [
        A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.SquareSymmetry(p=1.0),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.05, 0.15), fill=0, p=0.3),
    ]
    if sensor == "RGB":
        augs.append(A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5))
        augs.append(A.ToGray(p=0.1))
    if sensor in ("RGB", "Thermo"):
        augs.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5))
        augs.append(A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.2))
    augs.append(A.GaussNoise(std_range=(0.01, 0.05), p=0.2))
    augs.append(A.OneOf([
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=1.0),
        A.Perspective(scale=(0.05, 0.1), p=1.0),
    ], p=0.7))
    return A.Compose(augs, seed=seed)


def eval_transform(target_size: tuple[int, int]) -> A.Compose:
    h, w = target_size
    return A.Compose([A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR,
                               mask_interpolation=cv2.INTER_NEAREST)])


def resize_instance_map(instance_map: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    """Instance-id map at the target size (nearest neighbour), as in ``eval_transform``."""
    if target_size is None or tuple(instance_map.shape[:2]) == tuple(target_size):
        return instance_map
    h, w = target_size
    return cv2.resize(instance_map, (w, h), interpolation=cv2.INTER_NEAREST)
