"""Early-fusion input: the four sensors of a patch resized to one raster and stacked.

For every patch that has all four sensors, the sensor data are loaded as the dataset loads them
(``dataset.load_image``; RGB is flipped from the stored B, G, R to R, G, B), each sensor is
resized to the target size with bilinear interpolation (``cv2.INTER_LINEAR``), and the results
are stacked along the channel axis in the order of ``sensors.EF_ORDER``: R, G, B, the 224 NIR
bands, Thermo, the 200 THz bands (428 channels, float32). The ground truth is the RGB-raster
ground truth resized with nearest neighbour (all three channels, uint16). If the RGB patch has
a preview, it is resized as well (for viewing only).

Output in ``ef_root``, laid out like a single-sensor dataset:

    EF/patches_<split>/<stem>.npy, <stem>_GT.png, <stem>.png
    patch_inventory_EF.csv      inventory, sensor "EF"
    ef_source.json              source fingerprint

``<stem>`` is the RGB stem with ``EF`` instead of ``RGB``. Every file is written atomically
(temporary file, then rename). Patches already built are skipped unless ``overwrite``. The
source fingerprint (identity of the input inventory, target size, channel order, code version)
identifies a build: if it differs from the one in ``ef_root``, the build stops unless
``overwrite`` is set. Only patches listed in the EF inventory of the same build are skipped, so
files of another build are never reused; the inventory is also written when a build is
interrupted. Since the stored RGB part is already R, G, B, EF data are never flipped again when
loaded (``sensors.EF.stored_bgr`` is False).
"""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
from tqdm import tqdm

from woodvit_rcnn import __version__
from woodvit_rcnn.dataset import load_image
from woodvit_rcnn.gt import read_gt
from woodvit_rcnn.inventory import (SPLITS, PatchRecord, find_inventory, read_inventory, select_records,
                                    write_inventory)
from woodvit_rcnn.run_io import dataset_identity, read_json, write_json
from woodvit_rcnn.sensors import EF, EF_ORDER, SENSORS, ef_channel_ranges

EF_INVENTORY = "patch_inventory_EF.csv"
SOURCE_FILE = "ef_source.json"
SOURCE_FORMAT = "woodvit_rcnn.ef_source/1"
SOURCE_INVENTORY_KEY = "source_inventory"   # key of the source dataset identity in SOURCE_FILE
NPY_HEADER_BYTES = 128


def ef_stem(rgb_stem: str) -> str:
    """``RGB_probe04_patch01_24.0cm_x0_y0_Non_wood_Metal`` -> ``EF_probe04_...``."""
    return "EF" + rgb_stem[3:] if rgb_stem.startswith("RGB") else "EF_" + rgb_stem


def ef_bytes_per_patch(target_size: tuple[int, int]) -> int:
    """Estimated disk space of one EF patch: float32 data plus upper bounds for ground truth and preview."""
    h, w = target_size
    return 4 * h * w * EF.channels + NPY_HEADER_BYTES + 6 * h * w + 3 * h * w


def source_fingerprint(inventory_path: str | Path, target_size: tuple[int, int]) -> dict:
    """Everything an EF build depends on; the source data are identified by their inventory."""
    return {
        "format": SOURCE_FORMAT,
        SOURCE_INVENTORY_KEY: dataset_identity(inventory_path),
        "target_size": [int(target_size[0]), int(target_size[1])],
        "channel_order": list(EF_ORDER),
        "channel_ranges": {name: list(r) for name, r in ef_channel_ranges().items()},
        "rgb_channels": "R, G, B",
        "dtype": "float32",
        "data_interpolation": "cv2.INTER_LINEAR",
        "gt_interpolation": "cv2.INTER_NEAREST",
        "code_version": __version__,
    }


def resize_to(array: np.ndarray, target_size: tuple[int, int], interpolation: int) -> np.ndarray:
    """(H, W[, C]) array at the target size (height, width), channel axis kept."""
    h, w = target_size
    out = cv2.resize(array, (w, h), interpolation=interpolation)
    return out.reshape((h, w) + array.shape[2:])


def fuse_arrays(arrays: Mapping[str, np.ndarray], target_size: tuple[int, int]) -> np.ndarray:
    """Stack of the sensors in ``EF_ORDER``, each resized bilinearly: float32 (H, W, 428).

    ``arrays`` holds the data as returned by ``dataset.load_image`` (channels last, RGB already
    flipped to R, G, B).
    """
    parts = []
    for name in EF_ORDER:
        arr = arrays[name]
        if arr.ndim != 3 or arr.shape[2] != SENSORS[name].channels:
            raise ValueError(f"{name}: expected (H, W, {SENSORS[name].channels}), got {arr.shape}")
        parts.append(resize_to(np.ascontiguousarray(arr, dtype=np.float32), target_size, cv2.INTER_LINEAR))
    return np.ascontiguousarray(np.concatenate(parts, axis=2), dtype=np.float32)


def resize_gt(gt: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Ground truth (all three channels, uint16) at the target size with nearest neighbour."""
    return resize_to(gt, target_size, cv2.INTER_NEAREST)


# ----------------------------------------------------------------------------- atomic writes
def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _save_png(path: Path, image: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), np.ascontiguousarray(image)):
        raise OSError(f"cannot write {path}")
    os.replace(tmp, path)


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


# ----------------------------------------------------------------------------- build
@dataclass
class EarlyFusionReport:
    patches: int = 0                    # complete patch sets selected
    written: int = 0
    skipped: int = 0                    # already present and overwrite=False
    incomplete: list = field(default_factory=list)      # patch keys without all four sensors
    records: list[PatchRecord] = field(default_factory=list)
    inventory: Path | None = None


def _patch_sets(records: Iterable[PatchRecord]) -> tuple[dict, list]:
    by_key: dict[tuple, dict[str, PatchRecord]] = defaultdict(dict)
    for r in records:
        if r.sensor not in EF_ORDER:
            continue
        if r.sensor in by_key[r.patch_key]:
            raise ValueError(f"patch {r.patch_key}: sensor {r.sensor} listed twice in the inventory")
        by_key[r.patch_key][r.sensor] = r
    complete = {k: v for k, v in sorted(by_key.items()) if len(v) == len(EF_ORDER)}
    incomplete = sorted(k for k, v in by_key.items() if len(v) != len(EF_ORDER))
    for k, sensor_records in complete.items():
        splits = {r.split for r in sensor_records.values()}
        if len(splits) != 1:
            raise ValueError(f"patch {k}: the sensors are in different splits {sorted(splits)}")
    return complete, incomplete


def _ef_record(rgb: PatchRecord) -> PatchRecord:
    stem = ef_stem(rgb.stem)
    return PatchRecord(sensor=EF.name, split=rgb.split, probe=rgb.probe, patch=rgb.patch, x=rgb.x, y=rgb.y,
                       rel_path=f"{EF.name}/patches_{rgb.split}/{stem}.npy", name=stem, size=rgb.size,
                       main_class=rgb.main_class, subclass=rgb.subclass)


def build_early_fusion(data_root: str | Path, ef_root: str | Path, target_size: tuple[int, int] = (256, 256),
                       splits: Iterable[str] | None = None, probes: Iterable[int] | None = None,
                       overwrite: bool = False, progress: bool = True, *,
                       inventory: str | Path | None = None) -> EarlyFusionReport:
    """Write the early-fusion input of ``data_root`` (or of selected splits and recordings) to ``ef_root``.

    ``inventory`` is the inventory of the single-sensor data (CSV file); by default the one in
    ``data_root``. Its ``rel_path`` entries are relative to ``data_root``. The EF inventory lists
    the patches of this call plus those of earlier calls with the same fingerprint. Patches
    without all four sensors are skipped and listed in ``report.incomplete``. Raises ValueError
    for a different build in ``ef_root`` (unless ``overwrite``) and OSError if the estimated
    space (``ef_bytes_per_patch``) of the patches to write exceeds the free disk space.
    """
    data_root, ef_root = Path(data_root), Path(ef_root)
    target_size = (int(target_size[0]), int(target_size[1]))
    inventory_path = find_inventory(data_root) if inventory is None else Path(inventory)
    if inventory_path.is_dir():
        inventory_path = find_inventory(inventory_path)

    records = select_records(read_inventory(inventory_path), probes=probes)
    if splits is not None:
        wanted = set(splits)
        unknown = wanted - set(SPLITS)
        if unknown:
            raise ValueError(f"unknown splits {sorted(unknown)}, expected some of {SPLITS}")
        records = [r for r in records if r.split in wanted]
    complete, incomplete = _patch_sets(records)
    if not complete:
        raise ValueError(f"no patch with all four sensors {EF_ORDER} in {inventory_path} "
                         f"(splits={splits}, probes={probes})")

    # source fingerprint: an existing build must come from the same input and settings
    fingerprint = source_fingerprint(inventory_path, target_size)
    source_file = ef_root / SOURCE_FILE
    existing = read_json(source_file) if source_file.is_file() else None
    has_data = (ef_root / EF.name).is_dir() and any((ef_root / EF.name).rglob("*.npy"))
    same_build = existing == fingerprint
    if not same_build and (existing is not None or has_data) and not overwrite:
        found = "no source fingerprint" if existing is None else f"fingerprint {existing}"
        raise ValueError(
            f"{ef_root} holds early-fusion data of a different build ({found}); expected {fingerprint}. "
            f"Use another EF folder or set overwrite=True to rebuild.")

    # only patches listed in the inventory of the same build are reused, never files of another build
    inventory_out = ef_root / EF_INVENTORY
    previous = {}
    if same_build and inventory_out.is_file():
        previous = {r.patch_key: r for r in read_inventory(inventory_out)
                    if r.data_path(ef_root).is_file() and r.gt_path(ef_root).is_file()}
    done = dict(previous)       # inventory rows: earlier calls of this build plus the finished patches of this call
    report = EarlyFusionReport(patches=len(complete), incomplete=incomplete)
    todo = []
    for sensor_records in complete.values():
        ef = _ef_record(sensor_records["RGB"])
        report.records.append(ef)
        if overwrite or ef.patch_key not in previous:
            todo.append((ef, sensor_records))
            done.pop(ef.patch_key, None)
        else:
            report.skipped += 1

    need = len(todo) * ef_bytes_per_patch(target_size)
    free = _free_bytes(ef_root)
    if need > free:
        raise OSError(f"early fusion of {len(todo)} patches at {target_size[0]} x {target_size[1]} needs about "
                      f"{need / 1e9:.1f} GB, but only {free / 1e9:.1f} GB are free for {ef_root}. "
                      f"Free space, choose another EF_ROOT, or restrict SPLITS/PROBES.")

    ef_root.mkdir(parents=True, exist_ok=True)
    write_json(source_file, fingerprint)
    try:
        for ef, sensor_records in tqdm(todo, desc="Early fusion", disable=not progress):
            _write_patch(data_root, ef_root, ef, sensor_records, target_size)
            done[ef.patch_key] = ef
            report.written += 1
    finally:
        # also after an interruption, so a rerun skips the patches finished so far
        report.inventory = write_inventory(inventory_out, done.values())
    return report


def _write_patch(data_root: Path, ef_root: Path, ef: PatchRecord, sensor_records: Mapping[str, PatchRecord],
                 target_size: tuple[int, int]) -> None:
    arrays = {}
    for name in EF_ORDER:
        sensor = SENSORS[name]
        arrays[name] = load_image(sensor_records[name].data_path(data_root), sensor.channels, sensor.stored_bgr)
    rgb = sensor_records["RGB"]
    gt = read_gt(rgb.gt_path(data_root))
    if gt.shape[:2] != arrays["RGB"].shape[:2]:
        raise ValueError(f"{rgb.rel_path}: data {arrays['RGB'].shape[:2]} and ground truth {gt.shape[:2]} "
                         f"differ in size")
    ef.data_path(ef_root).parent.mkdir(parents=True, exist_ok=True)
    _save_npy(ef.data_path(ef_root), fuse_arrays(arrays, target_size))
    _save_png(ef.gt_path(ef_root), resize_gt(gt, target_size))
    preview_path = rgb.preview_path(data_root)
    preview = cv2.imread(str(preview_path), cv2.IMREAD_UNCHANGED) if preview_path.is_file() else None
    if preview is not None:
        _save_png(ef.preview_path(ef_root), resize_to(preview, target_size, cv2.INTER_AREA))
