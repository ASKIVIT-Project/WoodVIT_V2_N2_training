"""COCO ground truth built directly from the ground-truth PNGs.

The ground truth is rebuilt on every call, so it always matches the current dataset and class
mapping. Only the ground-truth PNGs are read, not the sensor data; the RGB test split takes about
a minute. Masks are resized to the target size with
nearest neighbour exactly as in the evaluation transform of the dataset. Every annotation carries
the category of the training target and the original class value (``class_value``), so one file
serves both the COCO metrics and the per-subclass instance counts.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from woodvit_rcnn.classes import Target, get_target
from woodvit_rcnn.coco_utils import mask_bbox, mask_to_rle
from woodvit_rcnn.gt import InstanceStats, class_and_instance_maps, extract_instances, read_gt
from woodvit_rcnn.inventory import check_patch_size, find_inventory, read_inventory, select_records
from woodvit_rcnn.run_io import dataset_identity, with_sensor
from woodvit_rcnn.sensors import get_sensor
from woodvit_rcnn.transforms import resize_instance_map


def build_coco_gt(data_root: str | Path, sensor: str, target: str | Target, split: str, *,
                  target_size: tuple[int, int] | None = (256, 256), include_covered: bool = True, probes=None,
                  image_sensor_index: int | None = None, progress: bool = False) -> tuple[dict, InstanceStats]:
    """COCO ground-truth dict of one sensor, target and split, and the instance statistics.

    ``image_sensor_index`` replaces the sensor digit of the image ids (late fusion uses the RGB
    ground truth with index 6).
    """
    root = Path(data_root)
    target = get_target(target) if isinstance(target, str) else target
    sensor_name = get_sensor(sensor).name
    records = select_records(read_inventory(root), sensor=sensor_name, split=split, probes=probes,
                             include_covered=include_covered)
    check_patch_size(records, root)
    stats = InstanceStats()
    images, annotations = [], []
    ann_id = 1
    for record in tqdm(records, desc=f"Ground truth {sensor_name} {split}", disable=not progress):
        gt = read_gt(record.gt_path(root))
        class_map, instance_map = class_and_instance_maps(gt)
        instances = extract_instances(class_map, instance_map, target, stats)
        resized = resize_instance_map(instance_map, target_size)
        img_id = record.image_id if image_sensor_index is None else with_sensor(record.image_id, image_sensor_index)
        h, w = resized.shape[:2]
        images.append({"id": img_id, "file_name": record.rel_path, "width": int(w), "height": int(h)})
        for inst in instances:
            mask = resized == inst.instance_id
            area = int(mask.sum())
            if area == 0:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": inst.category_id, "class_value": inst.value,
                "bbox": mask_bbox(mask), "area": float(area), "iscrowd": 0, "segmentation": mask_to_rle(mask),
            })
            ann_id += 1
    gt_dict = {
        "info": {
            "description": "WoodVIT V2 ground truth", "sensor": sensor_name, "target": target.name, "split": split,
            "target_size": None if target_size is None else list(target_size),
            "include_covered": include_covered, "probes": None if probes is None else sorted(int(p) for p in probes),
            "dataset": dataset_identity(find_inventory(root)),
        },
        "images": images,
        "annotations": annotations,
        "categories": target.coco_categories(),
    }
    return gt_dict, stats
