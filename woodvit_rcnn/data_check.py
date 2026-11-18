"""Consistency check and instance statistics of a dataset folder.

``check_dataset`` compares the files on disk with the inventory, checks shape, channel count,
dtype and value range of the sensor data, and counts the ground-truth instances per sensor,
split and class. It also counts the instances that need attention at the training resolution:

* boxes with a side below ``small_side`` pixels after resizing to ``target_size``,
* instances that disappear when the instance map is resized with nearest neighbour,
* instances whose pixels carry more than one class value, and instances with class 0.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from woodvit_rcnn.classes import BY_VALUE, MAIN
from woodvit_rcnn.gt import InstanceStats, class_and_instance_maps, extract_instances, read_gt
from woodvit_rcnn.inventory import find_inventory, missing_files, read_inventory, select_records
from woodvit_rcnn.run_io import dataset_identity
from woodvit_rcnn.sensors import get_sensor


@dataclass
class DataCheckReport:
    root: str
    identity: dict
    rows: pd.DataFrame                       # inventory rows per sensor and split
    missing: list[str]
    incomplete_patches: list[tuple]          # patch keys without all sensors
    arrays: pd.DataFrame                     # one row per checked array
    instances: pd.DataFrame                  # instances per sensor, split and class
    attention: pd.DataFrame                  # small, lost, mixed and background-class instances per sensor and split
    problems: list[str] = field(default_factory=list)


def check_dataset(root: str | Path, sensors=None, probes=None, target_size=(256, 256), small_side: int = 11,
                  array_samples: int = 3, progress: bool = True) -> DataCheckReport:
    root = Path(root)
    inventory = find_inventory(root)
    records = select_records(read_inventory(inventory), probes=probes)
    if sensors is not None:
        records = [r for r in records if r.sensor in set(sensors)]
    problems: list[str] = []

    rows = (pd.DataFrame([{"sensor": r.sensor, "split": r.split} for r in records])
            .value_counts().rename("rows").reset_index().sort_values(["sensor", "split"]))
    missing = missing_files(root, records)
    if missing:
        problems.append(f"{len(missing)} files named in the inventory are missing")

    present = defaultdict(set)
    for r in records:
        present[r.patch_key].add(r.sensor)
    all_sensors = {r.sensor for r in records}
    incomplete = sorted(k for k, s in present.items() if s != all_sensors)
    if incomplete:
        problems.append(f"{len(incomplete)} patches do not have all sensors {sorted(all_sensors)}")

    # sensor data: shape of every file (memory-mapped header), value range of a few files per sensor and split
    array_rows = []
    sampled = Counter()
    for r in tqdm(records, desc="Arrays", disable=not progress):
        path = r.data_path(root)
        if not path.is_file():
            continue
        arr = np.load(path, mmap_mode="r")
        sensor = get_sensor(r.sensor)
        channels = 1 if arr.ndim == 2 else arr.shape[2]
        row = {"sensor": r.sensor, "split": r.split, "file": r.rel_path, "shape": tuple(arr.shape),
               "dtype": str(arr.dtype), "channels_ok": channels == sensor.channels, "min": None, "max": None}
        if sampled[(r.sensor, r.split)] < array_samples:
            sampled[(r.sensor, r.split)] += 1
            data = np.asarray(arr)
            row["min"], row["max"] = float(np.nanmin(data)), float(np.nanmax(data))
            if row["min"] < 0.0 or row["max"] > 1.0 or np.isnan(data).any():
                problems.append(f"{r.rel_path}: values outside [0, 1] or NaN")
        if not row["channels_ok"]:
            problems.append(f"{r.rel_path}: {channels} channels, expected {sensor.channels}")
        if arr.dtype != np.float32:
            problems.append(f"{r.rel_path}: dtype {arr.dtype}, expected float32")
        array_rows.append(row)
    arrays = pd.DataFrame(array_rows)

    # ground truth: instances per class, and those that need attention at the training resolution
    counts = Counter()
    attention = defaultdict(Counter)
    th, tw = target_size
    for r in tqdm(records, desc="Ground truth", disable=not progress):
        gt_path = r.gt_path(root)
        if not gt_path.is_file():
            continue
        gt = read_gt(gt_path)
        class_map, instance_map = class_and_instance_maps(gt)
        stats = InstanceStats()
        instances = extract_instances(class_map, instance_map, MAIN, stats)
        key = (r.sensor, r.split)
        attention[key]["background_class"] += stats.background_class
        attention[key]["mixed_class"] += stats.mixed_class
        resized = cv2.resize(instance_map, (tw, th), interpolation=cv2.INTER_NEAREST)
        kept = set(np.unique(resized).tolist())
        for inst in instances:
            counts[(r.sensor, r.split, inst.value)] += 1
            if inst.instance_id not in kept:
                attention[key]["lost_at_target_size"] += 1
                continue
            ys, xs = np.nonzero(resized == inst.instance_id)
            if (xs.max() - xs.min() + 1) < small_side or (ys.max() - ys.min() + 1) < small_side:
                attention[key]["small_at_target_size"] += 1
        attention[key]["instances"] += len(instances)
        data_path = r.data_path(root)
        if data_path.is_file() and gt.shape[:2] != tuple(np.load(data_path, mmap_mode="r").shape[:2]):
            problems.append(f"{r.rel_path}: ground truth {gt.shape[:2]} and data differ in size")

    instances_df = pd.DataFrame(
        [{"sensor": s, "split": sp, "class_value": v, "class": BY_VALUE[v].name, "instances": n}
         for (s, sp, v), n in sorted(counts.items())]
    )
    attention_df = pd.DataFrame(
        [{"sensor": s, "split": sp, **dict(c)} for (s, sp), c in sorted(attention.items())]
    ).fillna(0)
    return DataCheckReport(root=str(root), identity=dataset_identity(inventory), rows=rows, missing=missing,
                           incomplete_patches=incomplete, arrays=arrays, instances=instances_df,
                           attention=attention_df, problems=problems)


def write_report(report: DataCheckReport, out_dir: str | Path) -> Path:
    """CSV files and a short text summary; returns the summary path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.rows.to_csv(out_dir / "inventory_rows.csv", index=False)
    report.arrays.to_csv(out_dir / "arrays.csv", index=False)
    report.instances.to_csv(out_dir / "instances_per_class.csv", index=False)
    report.attention.to_csv(out_dir / "instances_attention.csv", index=False)
    lines = [
        f"Dataset: {report.root}",
        f"Inventory: {report.identity['inventory']} sha1 {report.identity['sha1']} rows {report.identity['rows']}",
        f"Missing files: {len(report.missing)}",
        f"Patches without all sensors: {len(report.incomplete_patches)}",
        "",
        "Inventory rows per sensor and split:",
        report.rows.to_string(index=False),
        "",
        "Instances that need attention:",
        report.attention.to_string(index=False) if len(report.attention) else "(none)",
        "",
        "Problems:" if report.problems else "Problems: none",
        *[f"- {p}" for p in report.problems[:200]],
    ]
    summary = out_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
