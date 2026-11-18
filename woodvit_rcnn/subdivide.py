"""Subdivision of N=1 patches (48 cm) into N=2 quarters (24 cm).

Each N=1 array is cut into four children of floor(H/2) x floor(W/2) pixels, counted from the
upper-left corner (x to the right, y downwards): x0_y0, x1_y0, x0_y1, x1_y1. No interpolation is
involved: the children are exact sub-arrays, and the ground truth keeps its 16 bits, all three
channels, class values and instance ids. Previews (8-bit PNG) are cut the same way if present; for
NIR, THz and Thermo they therefore keep the contrast stretch of the parent patch (previews are for
viewing only).

Children inherit the split of their parent. Their file-name label is computed from the RGB-raster
ground truth of the child (``patch_class.patch_label``), so all sensors of a child share one name.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from woodvit_rcnn.gt import CLASS_CHANNEL, read_gt
from woodvit_rcnn.inventory import PATCH_SIZE, PatchRecord, read_inventory, select_records, write_inventory
from woodvit_rcnn.patch_class import patch_label, split_label

CHILDREN: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))
N2_INVENTORY = "patch_inventory_N2.csv"
N2_SIZE = PATCH_SIZE


def quarter(array: np.ndarray, x: int, y: int) -> np.ndarray:
    """Child (x, y) of a 2-D or 3-D array (rows, columns[, channels])."""
    h2, w2 = array.shape[0] // 2, array.shape[1] // 2
    return array[y * h2:(y + 1) * h2, x * w2:(x + 1) * w2]


def child_stem(sensor: str, probe: int, patch: int, x: int, y: int, label: str) -> str:
    return f"{sensor}_probe{probe:02d}_patch{patch:02d}_{N2_SIZE}_x{x}_y{y}_{label}"


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.ascontiguousarray(array))
    os.replace(tmp, path)


def _save_png(path: Path, image: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), np.ascontiguousarray(image)):
        raise OSError(f"cannot write {path}")
    os.replace(tmp, path)


@dataclass
class SubdivideReport:
    parents: int = 0
    children_written: int = 0
    children_skipped: int = 0           # already present and overwrite=False
    previews_missing: int = 0
    records: list[PatchRecord] = field(default_factory=list)


def subdivide_dataset(n1_root: str | Path, n2_root: str | Path, probes=None, sensors=None,
                      previews: bool = True, overwrite: bool = False, progress: bool = True) -> SubdivideReport:
    """Write the N=2 dataset of ``n1_root`` (or of selected recordings and sensors) to ``n2_root``,
    including ``patch_inventory_N2.csv``.

    The inventory is merged: rows of patches written by earlier calls (other recordings or sensors)
    are kept, rows of the patches of this call are replaced. It is also written if the call is
    interrupted, with the patches finished so far.
    """
    n1_root, n2_root = Path(n1_root), Path(n2_root)
    report = SubdivideReport()
    try:
        _subdivide(n1_root, n2_root, probes, sensors, previews, overwrite, progress, report)
    finally:
        _write_merged_inventory(n2_root / N2_INVENTORY, report.records)
    return report


def _write_merged_inventory(path: Path, new_records: list[PatchRecord]) -> None:
    keep: list[PatchRecord] = []
    if path.is_file():
        new_keys = {(r.sensor, r.patch_key) for r in new_records}
        keep = [r for r in read_inventory(path) if (r.sensor, r.patch_key) not in new_keys]
    if keep or new_records:
        write_inventory(path, keep + new_records)


def _subdivide(n1_root: Path, n2_root: Path, probes, sensors, previews: bool, overwrite: bool, progress: bool,
               report: SubdivideReport) -> None:
    records = select_records(read_inventory(n1_root), probes=probes)
    if sensors is not None:
        wanted = set(sensors)
        records = [r for r in records if r.sensor in wanted or r.sensor == "RGB"]
    by_parent: dict[tuple[int, int], dict[str, PatchRecord]] = defaultdict(dict)
    for r in records:
        by_parent[(r.probe, r.patch)][r.sensor] = r

    for (probe, patch), sensor_records in tqdm(sorted(by_parent.items()), desc="Subdividing", disable=not progress):
        rgb = sensor_records.get("RGB")
        if rgb is None:
            raise ValueError(f"recording {probe}, patch {patch}: the RGB ground truth is needed for the child names")
        rgb_classes = read_gt(rgb.gt_path(n1_root))[..., CLASS_CHANNEL]
        labels = {(x, y): patch_label(quarter(rgb_classes, x, y)) for x, y in CHILDREN}
        report.parents += 1
        for sensor, record in sorted(sensor_records.items()):
            if sensors is not None and sensor not in set(sensors):
                continue
            out_dir = n2_root / sensor / f"patches_{record.split}"
            out_dir.mkdir(parents=True, exist_ok=True)
            stems = {xy: child_stem(sensor, probe, patch, xy[0], xy[1], labels[xy]) for xy in CHILDREN}
            todo = [xy for xy in CHILDREN
                    if overwrite or not ((out_dir / (stems[xy] + ".npy")).is_file()
                                         and (out_dir / (stems[xy] + "_GT.png")).is_file())]
            report.children_skipped += len(CHILDREN) - len(todo)
            if todo:
                data = np.load(record.data_path(n1_root))
                gt = read_gt(record.gt_path(n1_root))
                preview_path = record.preview_path(n1_root)
                preview = cv2.imread(str(preview_path), cv2.IMREAD_UNCHANGED) if previews and preview_path.is_file() else None
                if previews and preview is None:
                    report.previews_missing += 1
                for xy in todo:
                    _save_npy(out_dir / (stems[xy] + ".npy"), quarter(data, *xy))
                    _save_png(out_dir / (stems[xy] + "_GT.png"), quarter(gt, *xy))
                    if preview is not None:
                        _save_png(out_dir / (stems[xy] + ".png"), quarter(preview, *xy))
                    report.children_written += 1
            for x, y in CHILDREN:
                main, sub = split_label(labels[(x, y)])
                report.records.append(PatchRecord(
                    sensor=sensor, split=record.split, probe=probe, patch=patch, x=x, y=y,
                    rel_path=f"{sensor}/patches_{record.split}/{stems[(x, y)]}.npy", name=stems[(x, y)],
                    size=N2_SIZE, main_class=main, subclass=sub,
                ))
