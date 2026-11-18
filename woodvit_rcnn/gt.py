"""Reading ground-truth PNGs and extracting instances.

A ground-truth PNG is uint16 with three channels. Read with OpenCV (B, G, R order), channel 2
holds the class value and channel 1 the instance id (0 = no instance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from woodvit_rcnn.classes import BACKGROUND_VALUE, BY_VALUE, Target

CLASS_CHANNEL = 2
INSTANCE_CHANNEL = 1


def read_gt(path: str | Path) -> np.ndarray:
    """Ground truth as (H, W, 3) uint16 array in OpenCV channel order."""
    gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"cannot read ground truth {path}")
    if gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError(f"{path}: expected three channels, got shape {gt.shape}")
    if gt.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16, got {gt.dtype}")
    return gt


def class_and_instance_maps(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return gt[..., CLASS_CHANNEL], gt[..., INSTANCE_CHANNEL]


@dataclass(frozen=True)
class Instance:
    instance_id: int
    value: int          # ground-truth class value (majority of the instance's pixels)
    category_id: int    # category of the training target


@dataclass
class InstanceStats:
    """Counts of instances that need attention; filled by ``extract_instances``."""

    background_class: int = 0           # instance pixels with class value 0: skipped
    ignored_class: int = 0              # class value ignored by the target (Undefined in ``sub``): skipped
    mixed_class: int = 0                # instances whose pixels carry more than one class value
    mixed_examples: list = field(default_factory=list)

    def add(self, other: "InstanceStats") -> None:
        self.background_class += other.background_class
        self.ignored_class += other.ignored_class
        self.mixed_class += other.mixed_class
        self.mixed_examples.extend(other.mixed_examples[: max(0, 20 - len(self.mixed_examples))])


def extract_instances(class_map: np.ndarray, instance_map: np.ndarray, target: Target,
                      stats: InstanceStats | None = None) -> list[Instance]:
    """Instances of a ground truth with their class and category.

    The class of an instance is the most frequent class value of its pixels. Instances whose
    class is Background or a class value the target ignores are skipped and counted. Class values
    outside the class table raise a ValueError.
    """
    stats = stats if stats is not None else InstanceStats()
    ids = np.unique(instance_map)
    ids = ids[ids != 0]
    out = []
    for uid in ids.tolist():
        values = class_map[instance_map == uid]
        uniq, counts = np.unique(values, return_counts=True)
        unknown = [int(v) for v in uniq if int(v) not in BY_VALUE]
        if unknown:
            raise ValueError(f"instance {uid}: class values {unknown} are not in the class table")
        if len(uniq) > 1:
            stats.mixed_class += 1
            if len(stats.mixed_examples) < 20:
                stats.mixed_examples.append({int(v): int(c) for v, c in zip(uniq, counts)})
        value = int(uniq[np.argmax(counts)])
        if value == BACKGROUND_VALUE:
            stats.background_class += 1
            continue
        if value in target.ignored_values:
            stats.ignored_class += 1
            continue
        out.append(Instance(instance_id=int(uid), value=value, category_id=target.category_id(value)))
    return out
