"""PyTorch dataset of one sensor, target and split.

Each item is (image, target): the image as float32 tensor (C, H, W) and the target as the dict
torchvision's Mask R-CNN expects (boxes, labels, masks) plus image_id, area (mask area),
iscrowd and class_values (ground-truth class value of every instance).

* The sensor data are read as stored (float32 in [0, 1], channels last). Sensors stored as
  B, G, R are flipped to R, G, B.
* Instances come from the instance channel of the ground truth; the class of an instance is the
  most frequent class value of its pixels, mapped to the category of the training target.
* Boxes are computed after the transform from the transformed masks, with inclusive extent
  (x_max + 1), so every non-empty mask has a box of at least 1 x 1 pixel. Instances that vanish
  in the transform are dropped. There is no minimum instance size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from woodvit_rcnn.classes import Target, get_target
from woodvit_rcnn.gt import InstanceStats, class_and_instance_maps, extract_instances, read_gt, Instance
from woodvit_rcnn.inventory import PatchRecord, check_patch_size, read_inventory, select_records
from woodvit_rcnn.sensors import get_sensor
from woodvit_rcnn.transforms import eval_transform, train_transform


def load_image(path: str | Path, channels: int, flip_bgr: bool) -> np.ndarray:
    """Sensor data as contiguous float32 (H, W, C)."""
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[2] != channels:
        raise ValueError(f"{path}: expected (H, W, {channels}), got {arr.shape}")
    if flip_bgr:
        arr = arr[..., ::-1]
    return np.ascontiguousarray(arr, dtype=np.float32)


def build_target(instance_map: np.ndarray, instances: list[Instance], image_id: int) -> dict:
    """Target dict from a (transformed) instance map and the instances of the patch."""
    boxes, labels, masks, areas, values = [], [], [], [], []
    for inst in instances:
        m = instance_map == inst.instance_id
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        labels.append(inst.category_id)
        masks.append(m)
        areas.append(float(m.sum()))
        values.append(inst.value)
    h, w = instance_map.shape[:2]
    n = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(n, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if n else torch.zeros((0, h, w), dtype=torch.uint8),
        "image_id": torch.tensor(int(image_id), dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros((n,), dtype=torch.int64),
        "class_values": torch.tensor(values, dtype=torch.int64),
    }


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


class WoodVitDataset(Dataset):
    """Patches of one sensor and split.

    ``train=True`` applies the training augmentation, otherwise the image is only resized to
    ``target_size``. ``include_covered=False`` drops the covered recordings 48, 49, 51, 54.
    ``keep_stored_order`` keeps B, G, R data unflipped (for models trained on the stored order).
    ``channel_window`` keeps only the channels [start, stop) of the loaded data, before any
    transform (the file is still read and checked with all channels of the sensor).
    ``seed`` makes the training augmentation reproducible (see ``seed_worker`` for data-loader workers).
    """

    def __init__(self, data_root: str | Path, sensor: str, target: str | Target, split: str, *,
                 train: bool = False, target_size: tuple[int, int] = (256, 256), include_covered: bool = True,
                 probes=None, keep_stored_order: bool = False, seed: int | None = None,
                 channel_window: tuple[int, int] | None = None):
        self.root = Path(data_root)
        self.sensor = get_sensor(sensor)
        self.target = get_target(target) if isinstance(target, str) else target
        self.split = split
        self.target_size = tuple(target_size)
        self.records: list[PatchRecord] = select_records(
            read_inventory(self.root), sensor=self.sensor.name, split=split, probes=probes,
            include_covered=include_covered)
        if not self.records:
            raise ValueError(f"no {self.sensor.name} patches for split {split!r} in {self.root} "
                             f"(probes={probes}, include_covered={include_covered})")
        check_patch_size(self.records, self.root)
        self.seed = seed
        self.transform = (train_transform(self.sensor.name, self.target_size, seed=seed) if train
                          else eval_transform(self.target_size))
        self.flip_bgr = self.sensor.stored_bgr and not keep_stored_order
        if channel_window is not None and not 0 <= channel_window[0] < channel_window[1] <= self.sensor.channels:
            raise ValueError(f"channel window {tuple(channel_window)} is not inside 0..{self.sensor.channels}")
        self.channel_window = None if channel_window is None else (int(channel_window[0]), int(channel_window[1]))
        self.stats = InstanceStats()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def image_ids(self) -> list[int]:
        return [r.image_id for r in self.records]

    def load(self, index: int) -> tuple[np.ndarray, np.ndarray, list[Instance]]:
        """Untransformed (image, instance map, instances) of an item."""
        record = self.records[index]
        image = load_image(record.data_path(self.root), self.sensor.channels, self.flip_bgr)
        if self.channel_window is not None:
            image = np.ascontiguousarray(image[..., self.channel_window[0]:self.channel_window[1]])
        gt = read_gt(record.gt_path(self.root))
        if gt.shape[:2] != image.shape[:2]:
            raise ValueError(f"{record.rel_path}: data {image.shape[:2]} and ground truth {gt.shape[:2]} differ in size")
        class_map, instance_map = class_and_instance_maps(gt)
        instances = extract_instances(class_map, instance_map, self.target, self.stats)
        return image, instance_map, instances

    def __getitem__(self, index: int):
        image, instance_map, instances = self.load(index)
        out = self.transform(image=image, mask=instance_map)
        target = build_target(out["mask"], instances, self.records[index].image_id)
        return to_tensor(out["image"]), target


def collate(batch):
    """Batches as (images, targets) tuples of variable-size items."""
    return tuple(zip(*batch))


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for data loaders: every worker augments with its own reproducible seed."""
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset if info is not None else None
    if dataset is not None and getattr(dataset, "seed", None) is not None:
        dataset.transform.set_random_seed(dataset.seed + worker_id)
