"""The patch inventory of a WoodVIT V2 dataset folder.

Every dataset folder holds ``<Sensor>/patches_<split>/`` and one inventory CSV with a row per
sensor file. The inventory is the only source of split, recording, patch and child position;
file names are never parsed. Columns:

    global_id, name, sensor, split, probe, patch, size, x, y, main_class, subclass, class_id, rel_path

``main_class``/``subclass`` describe the patch as a whole (the dominant class, used in the
file name) and are not used for training. Each patch has three files next to each other:
``<stem>.npy`` (sensor data), ``<stem>_GT.png`` (ground truth) and optionally ``<stem>.png``
(an 8-bit preview).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from woodvit_rcnn.classes import BY_NAME
from woodvit_rcnn.run_io import image_id as _image_id
from woodvit_rcnn.sensors import get_sensor

# Upper layers of the covered configurations (base layer in brackets): 48 (47), 49 (47), 51 (50), 54 (53).
COVERED_TOP_RECORDINGS: frozenset[int] = frozenset({48, 49, 51, 54})

INVENTORY_NAMES: tuple[str, ...] = (
    "patch_inventory_N2.csv", "patch_inventory_N1.csv", "patch_inventory_EF.csv", "patch_inventory.csv",
)
FIELDS: tuple[str, ...] = (
    "global_id", "name", "sensor", "split", "probe", "patch", "size", "x", "y",
    "main_class", "subclass", "class_id", "rel_path",
)
SPLITS: tuple[str, ...] = ("train", "val", "test")
PATCH_SIZE = "24.0cm"       # training and evaluation use N=2 patches only
DATA_SUFFIX = ".npy"
GT_SUFFIX = "_GT.png"
PREVIEW_SUFFIX = ".png"


@dataclass(frozen=True)
class PatchRecord:
    sensor: str
    split: str
    probe: int
    patch: int
    x: int
    y: int
    rel_path: str           # relative to the dataset folder, '/'-separated, ends with .npy
    name: str = ""
    size: str = ""
    main_class: str = ""
    subclass: str = ""
    global_id: int = 0

    @property
    def stem(self) -> str:
        return PurePosixPath(self.rel_path).name[: -len(DATA_SUFFIX)]

    @property
    def patch_key(self) -> tuple[int, int, int, int]:
        """(probe, patch, x, y): identical for all sensors of one patch."""
        return self.probe, self.patch, self.x, self.y

    @property
    def image_id(self) -> int:
        return _image_id(get_sensor(self.sensor).index, self.probe, self.patch, self.x, self.y)

    def data_path(self, root: str | Path) -> Path:
        return Path(root) / self.rel_path

    def gt_path(self, root: str | Path) -> Path:
        p = self.data_path(root)
        return p.with_name(self.stem + GT_SUFFIX)

    def preview_path(self, root: str | Path) -> Path:
        p = self.data_path(root)
        return p.with_name(self.stem + PREVIEW_SUFFIX)


def find_inventory(root: str | Path) -> Path:
    """Inventory file of a dataset folder (first of INVENTORY_NAMES that exists)."""
    root = Path(root)
    for name in INVENTORY_NAMES:
        if (root / name).is_file():
            return root / name
    raise FileNotFoundError(f"no inventory in {root} (looked for {', '.join(INVENTORY_NAMES)})")


def _int(row: dict, key: str, default: int = 0) -> int:
    value = (row.get(key) or "").strip()
    return int(value) if value else default


def read_inventory(path: str | Path) -> list[PatchRecord]:
    """Rows of an inventory. ``path`` is the CSV file or the dataset folder that contains it."""
    path = Path(path)
    if path.is_dir():
        path = find_inventory(path)
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"sensor", "split", "probe", "patch", "rel_path"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: columns missing: {sorted(missing)}")
        for row in reader:
            split = row["split"].strip()
            if split not in SPLITS:
                raise ValueError(f"{path}: unknown split {split!r} in row {row}")
            rel_path = row["rel_path"].strip().replace("\\", "/")
            records.append(PatchRecord(
                sensor=row["sensor"].strip(), split=split,
                probe=_int(row, "probe"), patch=_int(row, "patch"), x=_int(row, "x"), y=_int(row, "y"),
                rel_path=rel_path, name=(row.get("name") or "").strip() or PurePosixPath(rel_path).stem,
                size=(row.get("size") or "").strip(), main_class=(row.get("main_class") or "").strip(),
                subclass=(row.get("subclass") or "").strip(), global_id=_int(row, "global_id"),
            ))
    return records


def select_records(records: Iterable[PatchRecord], sensor: str | None = None, split: str | None = None,
                   probes: Iterable[int] | None = None, include_covered: bool = True) -> list[PatchRecord]:
    """Filter by sensor, split and recordings; ``include_covered=False`` drops 48, 49, 51, 54."""
    probe_set = None if probes is None else {int(p) for p in probes}
    out = []
    for r in records:
        if sensor is not None and r.sensor != sensor:
            continue
        if split is not None and r.split != split:
            continue
        if probe_set is not None and r.probe not in probe_set:
            continue
        if not include_covered and r.probe in COVERED_TOP_RECORDINGS:
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r.probe, r.patch, r.x, r.y, r.sensor))


def check_patch_size(records: Iterable[PatchRecord], root: str | Path) -> None:
    """Raise a ValueError if the records are not N=2 patches (``size`` column 24.0cm). Rows of
    inventories without a size column pass."""
    wrong = sorted({r.size for r in records if r.size and r.size != PATCH_SIZE})
    if wrong:
        raise ValueError(f"{root}: patches of size {', '.join(wrong)}; training and evaluation use the N=2 patches "
                         f"({PATCH_SIZE}). Point DATA_ROOT in paths.py to the N=2 dataset "
                         "(preprocessing/subdivide_n1_to_n2.py creates it from N=1).")


def write_inventory(path: str | Path, records: Iterable[PatchRecord]) -> Path:
    """Write an inventory. ``global_id`` numbers the patches (all sensors of a patch share one)
    in the order recording, patch, x, y."""
    rows = sorted(records, key=lambda r: (r.probe, r.patch, r.x, r.y, r.sensor))
    ids: dict[tuple, int] = {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            gid = ids.setdefault(r.patch_key, len(ids) + 1)
            sub = BY_NAME.get(r.subclass)
            writer.writerow({
                "global_id": gid, "name": r.name or r.stem, "sensor": r.sensor, "split": r.split,
                "probe": r.probe, "patch": r.patch, "size": r.size, "x": r.x, "y": r.y,
                "main_class": r.main_class, "subclass": r.subclass,
                "class_id": "" if sub is None else sub.value, "rel_path": r.rel_path,
            })
    tmp.replace(path)
    return path


def missing_files(root: str | Path, records: Iterable[PatchRecord], with_gt: bool = True) -> list[str]:
    """Relative paths of data (and ground-truth) files named in the inventory but not on disk."""
    root = Path(root)
    missing = []
    for r in records:
        if not r.data_path(root).is_file():
            missing.append(r.rel_path)
        if with_gt and not r.gt_path(root).is_file():
            missing.append(str(PurePosixPath(r.rel_path).with_name(r.stem + GT_SUFFIX)))
    return missing
