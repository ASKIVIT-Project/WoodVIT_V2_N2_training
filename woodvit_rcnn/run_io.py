"""Image ids, dataset identity, run folders and the prediction file format.

Image ids are composed from the patch coordinates, so the same patch keeps its id across
dataset builds and across sensors only differs in the leading digit:

    image_id = sensor_index * 10**7 + probe * 10**5 + patch * 100 + x * 10 + y

Every run stores the identity of the dataset it was trained on (hash of the inventory file).
Prediction files carry the identity of the dataset they were computed on; metrics and late
fusion refuse to combine predictions and ground truth from different dataset builds.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from woodvit_rcnn import __version__
from woodvit_rcnn.classes import MAPPING_VERSION
from woodvit_rcnn.presets import TrainConfig

ID_SENSOR = 10**7
ID_PROBE = 10**5
ID_PATCH = 100
ID_X = 10

TRAINED_MODELS = "Trained_Models"
RUN_FORMAT = "woodvit_rcnn.run/1"
PREDICTION_FORMAT = "woodvit_rcnn.predictions/1"


# ----------------------------------------------------------------------------- image ids
def image_id(sensor_index: int, probe: int, patch: int, x: int = 0, y: int = 0) -> int:
    if not 0 < sensor_index < 10:
        raise ValueError(f"sensor index {sensor_index} out of range 1..9")
    if not 0 <= probe < ID_SENSOR // ID_PROBE:
        raise ValueError(f"probe {probe} out of range 0..99")
    if not 0 <= patch < ID_PROBE // ID_PATCH:
        raise ValueError(f"patch {patch} out of range 0..999")
    if not (0 <= x < 10 and 0 <= y < 10):
        raise ValueError(f"child position x={x}, y={y} out of range 0..9")
    return sensor_index * ID_SENSOR + probe * ID_PROBE + patch * ID_PATCH + x * ID_X + y


def split_image_id(value: int) -> tuple[int, int, int, int, int]:
    """(sensor_index, probe, patch, x, y) of an image id."""
    value = int(value)
    sensor_index, rest = divmod(value, ID_SENSOR)
    probe, rest = divmod(rest, ID_PROBE)
    patch, rest = divmod(rest, ID_PATCH)
    x, y = divmod(rest, ID_X)
    return sensor_index, probe, patch, x, y


def core_id(value: int) -> int:
    """Image id without the sensor digit: identical for all sensors of one patch."""
    return int(value) % ID_SENSOR


def with_sensor(value: int, sensor_index: int) -> int:
    """The same patch seen by another sensor (or by late fusion)."""
    return sensor_index * ID_SENSOR + core_id(value)


# ----------------------------------------------------------------------------- dataset identity
def dataset_identity(inventory_path: str | Path) -> dict:
    """Name, SHA-1 and number of data rows of an inventory file.

    The identity covers the inventory only: it changes with the list of patches, their names and
    splits, not when a data or ground-truth file is replaced in place.
    """
    path = Path(inventory_path)
    content = path.read_bytes()
    rows = max(content.count(b"\n") - 1, 0) if content.endswith(b"\n") else content.count(b"\n")
    return {"inventory": path.name, "sha1": hashlib.sha1(content).hexdigest(), "rows": rows}


def check_dataset_identity(expected: dict | None, found: dict | None, what: str) -> None:
    """ValueError if two dataset identities differ. A missing identity is an error as well."""
    if not expected or not found:
        raise ValueError(f"{what}: dataset identity missing ({expected!r} / {found!r})")
    if expected.get("sha1") != found.get("sha1"):
        raise ValueError(
            f"{what}: computed on a different dataset build "
            f"({found.get('inventory')} {found.get('sha1')}) than expected "
            f"({expected.get('inventory')} {expected.get('sha1')})"
        )


# ----------------------------------------------------------------------------- run folders
@dataclass(frozen=True)
class RunPaths:
    """Files of one trained model: <results>/Trained_Models/<sensor>/<model folder>/."""

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def inference(self) -> Path:
        return self.root / "inference"

    @property
    def config_file(self) -> Path:
        return self.training / "config.json"

    @property
    def checkpoint(self) -> Path:
        return self.training / "checkpoint.pth"

    @property
    def train_log(self) -> Path:
        return self.training / "train_log.csv"

    @property
    def log_file(self) -> Path:
        return self.training / "train.log"

    @property
    def tensorboard(self) -> Path:
        return self.training / "tensorboard"

    def predictions(self, split: str) -> Path:
        return self.inference / f"pred_{split}.json"

    def ground_truth(self, split: str) -> Path:
        return self.inference / f"gt_{split}.json"

    def metrics(self, split: str) -> Path:
        return self.root / "metrics" / split


def new_model_folder(config: TrainConfig, now: _dt.datetime | None = None) -> str:
    """Folder name of a new run, e.g. ``RGB_N2_261014_101500_MAIN_256_lr5e-05_pat25``; a channel
    window is named with its first and last channel (``..._MAIN_ch160-190_256_...``)."""
    now = now or _dt.datetime.now()
    covered = "withCovered_" if config.covered_in_training else ""
    window = "" if config.channel_window is None else f"ch{config.channel_window[0]}-{config.channel_window[1] - 1}_"
    return (f"{config.sensor}_N2_{now:%y%m%d_%H%M%S}_{config.target.upper()}_{window}{config.target_size[0]}_"
            f"{covered}lr{config.learning_rate}_pat{config.patience}")


def run_paths(results_root: str | Path, sensor: str, model_folder: str) -> RunPaths:
    return RunPaths(Path(results_root) / TRAINED_MODELS / sensor / model_folder)


def find_run(results_root: str | Path, model_folder: str) -> RunPaths:
    """Run folder of a model folder name, searched under all sensors."""
    hits = sorted(p for p in (Path(results_root) / TRAINED_MODELS).glob(f"*/{model_folder}") if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no run {model_folder!r} under {Path(results_root) / TRAINED_MODELS}")
    if len(hits) > 1:
        raise ValueError(f"run {model_folder!r} exists more than once: {hits}")
    return RunPaths(hits[0])


# ----------------------------------------------------------------------------- json files
def write_json(path: str | Path, data) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_run_config(run: RunPaths, config: TrainConfig, dataset: dict) -> None:
    write_json(run.config_file, {
        "format": RUN_FORMAT,
        "code_version": __version__,
        "class_mapping_version": MAPPING_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_folder": run.name,
        "dataset": dataset,
        "config": config.to_dict(),
    })


def read_run_config(run: RunPaths) -> tuple[TrainConfig, dict]:
    """(configuration, full file content) of a run."""
    data = read_json(run.config_file)
    if data.get("format") != RUN_FORMAT:
        raise ValueError(f"{run.config_file}: not a run configuration of this code ({data.get('format')!r})")
    return TrainConfig.from_dict(data["config"]), data


def write_predictions(path: str | Path, results: list[dict], *, image_ids: list[int], sensor: str,
                      target: str, split: str, target_size: tuple[int, int], dataset: dict,
                      source: str) -> None:
    """Prediction file: COCO results plus the list of all predicted images and their provenance."""
    write_json(path, {
        "format": PREDICTION_FORMAT,
        "code_version": __version__,
        "class_mapping_version": MAPPING_VERSION,
        "source": source,
        "sensor": sensor,
        "target": target,
        "split": split,
        "target_size": list(target_size),
        "dataset": dataset,
        "image_ids": sorted(int(i) for i in image_ids),
        "results": results,
    })


def read_predictions(path: str | Path, expected_dataset: dict | None = None) -> dict:
    data = read_json(path)
    if data.get("format") != PREDICTION_FORMAT:
        raise ValueError(f"{path}: not a prediction file of this code ({data.get('format')!r})")
    if expected_dataset is not None:
        check_dataset_identity(expected_dataset, data.get("dataset"), f"predictions {Path(path).name}")
    return data
