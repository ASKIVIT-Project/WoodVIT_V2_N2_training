"""THz metal expert, second version (Metal_Expert_2).

A Mask R-CNN for Metal versus Non_metal on the THz data. It starts from the ``thz_metal`` preset
and differs from the baseline in six points, all stored with the run in
``config.extra["metal_expert_2"]`` (``MetalExpert2Settings``):

* **Channel window.** Only the channels ``channel_start`` to ``channel_stop - 1`` (160 to 190
  inclusive, 31 of the 200 THz channels) are fed to the model.
* **Normalisation** with the per-channel mean and standard deviation of the window channels over
  all pixels of the training split (``channel_stats``) instead of the ImageNet values.
* **Augmentation** without CoarseDropout and GaussNoise; the rest of the baseline list stays.
* **Backbone.** The whole backbone is trainable (with COCO weights the baseline freezes ``conv1``,
  ``bn1`` and ``layer1``). The first convolution starts from the cyclically repeated RGB weights
  scaled by 3 / C, so that the sum over the C input channels keeps the magnitude of the RGB
  filters (only when the model is built without a checkpoint).
* **Sampling.** Training images whose ground truth contains covered metal (class values 110, 120,
  130) are drawn ``covered_metal_repeat`` times as often: weighted sampling with replacement, as
  many draws per epoch as the training split has images. The flags come from one pass over the
  ground truth; file names are not used.
* **Selection.** Checkpoint and early stopping follow ``selection_metric``, by default the segm AP
  at IoU 0.5 of the class Metal on the validation split. All baseline metrics are logged as well.

Everything else is the baseline: model, optimiser, schedule, validation, run files. A run is a
normal run folder (sensor THz, target metal); ``evaluate.evaluate_model_run`` and the metric scripts
read it unchanged. The baseline inference cannot load these checkpoints (31 input channels, own
normalisation): use ``predict_split_me2``. ``covered_metal_table`` is the analysis of the
covered-metal subclasses after training.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import io
import math
import time
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from woodvit_rcnn.classes import BY_VALUE, COVERED_METAL_VALUES, METAL
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import IOU_TYPES, METRIC_NAMES, load_results, rle_to_mask, rle_union, to_coco
from woodvit_rcnn.dataset import WoodVitDataset, collate, seed_worker
from woodvit_rcnn.evaluate import check_inputs
from woodvit_rcnn.gt import class_and_instance_maps, extract_instances, read_gt
from woodvit_rcnn.instance_counts import UNDETECTED, count_instances
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.matching import ImageData, match_images, prepare_images
from woodvit_rcnn.model import MaskRCNNWithEvalLoss, build_model, frozen_parameters, resolve_device, save_checkpoint
from woodvit_rcnn.predict import predict_dataset
from woodvit_rcnn.presets import TrainConfig, make_config
from woodvit_rcnn.run_io import (RunPaths, check_dataset_identity, dataset_identity, read_json, read_run_config,
                                 write_json, write_predictions, write_run_config)
from woodvit_rcnn.sensors import THZ
from woodvit_rcnn.train_loop import (METRIC_KEYS, EarlyStopping, _close_logger, _fmt, _logger, make_loader, one_cycle,
                                     set_seed, to_device, train_epoch, train_step, validation_step)
from woodvit_rcnn.transforms import train_transform

SENSOR = THZ.name                           # "THz"
TARGET = METAL                              # Metal versus Non_metal
EXTRA_KEY = "metal_expert_2"                # key of the settings in TrainConfig.extra
VERSION = 1
BASE_PRESET = "thz_metal"
PRESET = "thz_metal_me2"                    # preset name stored in config.json
CHANNEL_WINDOW: tuple[int, int] = (160, 191)    # [start, stop): channels 160 to 190 inclusive
COVERED_METAL_REPEAT = 3.0
SELECTION_METRIC = "segm_AP50_Metal"
EXCLUDED_AUGMENTATIONS: tuple[str, ...] = ("CoarseDropout", "GaussNoise")
AP50_IOU_INDEX = 0                          # COCOeval iouThrs[0] == 0.50
METADATA_FILE = "training_metadata.json"
ANALYSIS_VALUES: tuple[int, ...] = (70, 110, 120, 130)      # open metal and the three covered-metal classes
ANALYSIS_FILE = "covered_metal_analysis.csv"


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class MetalExpert2Settings:
    """The settings of a run, stored in ``config.extra["metal_expert_2"]``.

    ``channel_mean`` and ``channel_std`` hold one value per window channel (the statistics of the
    training split). Datasets and ``channel_stats`` work without them; the model needs them.
    """

    channel_start: int = CHANNEL_WINDOW[0]
    channel_stop: int = CHANNEL_WINDOW[1]
    channel_mean: tuple[float, ...] = ()
    channel_std: tuple[float, ...] = ()
    covered_metal_repeat: float = COVERED_METAL_REPEAT
    selection_metric: str = SELECTION_METRIC
    version: int = VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.channel_start < self.channel_stop <= THZ.channels:
            raise ValueError(f"channel window [{self.channel_start}, {self.channel_stop}) is not inside 0..{THZ.channels}")
        if len(self.channel_mean) != len(self.channel_std):
            raise ValueError("channel_mean and channel_std differ in length")
        if self.channel_mean and len(self.channel_mean) != self.channels:
            raise ValueError(f"{len(self.channel_mean)} statistics for {self.channels} window channels")
        if any(not (s > 0) for s in self.channel_std):
            raise ValueError("every channel standard deviation must be positive")
        if self.covered_metal_repeat < 1:
            raise ValueError("covered_metal_repeat must be at least 1")

    @property
    def channels(self) -> int:
        return self.channel_stop - self.channel_start

    @property
    def first_conv_scale(self) -> float:
        return 3.0 / self.channels

    @property
    def has_stats(self) -> bool:
        return bool(self.channel_mean)

    def with_stats(self, stats: dict) -> "MetalExpert2Settings":
        """Settings with the mean and standard deviation of a ``channel_stats`` result."""
        window = (int(stats["channel_start"]), int(stats["channel_stop"]))
        if window != (self.channel_start, self.channel_stop):
            raise ValueError(f"statistics of the channel window {window} do not fit the window "
                             f"({self.channel_start}, {self.channel_stop})")
        return replace(self, channel_mean=tuple(float(v) for v in stats["mean"]),
                       channel_std=tuple(float(v) for v in stats["std"]))

    def to_dict(self) -> dict:
        return {
            "version": self.version, "channel_start": self.channel_start, "channel_stop": self.channel_stop,
            "channels": self.channels, "first_conv_scale": self.first_conv_scale,
            "channel_mean": list(self.channel_mean), "channel_std": list(self.channel_std),
            "covered_metal_repeat": self.covered_metal_repeat, "selection_metric": self.selection_metric,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetalExpert2Settings":
        d = dict(d)
        for derived in ("channels", "first_conv_scale"):
            d.pop(derived, None)
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown metal_expert_2 settings: {sorted(unknown)}")
        for key in ("channel_mean", "channel_std"):
            if key in d:
                d[key] = tuple(float(v) for v in d[key])
        return cls(**d)


def make_me2_config(settings: MetalExpert2Settings, learning_rate: float, **overrides) -> TrainConfig:
    """Configuration of a run: the ``thz_metal`` preset with the learning rate, the settings in
    ``extra`` and the preset name ``thz_metal_me2``. ``overrides`` change single preset values."""
    if not settings.has_stats:
        raise ValueError("the model needs the channel statistics: MetalExpert2Settings.with_stats(...)")
    extra = {**overrides.pop("extra", {}), EXTRA_KEY: settings.to_dict()}
    config = make_config(BASE_PRESET, learning_rate=learning_rate, extra=extra, **overrides)
    return replace(config, preset=PRESET)


def me2_settings(config: TrainConfig) -> MetalExpert2Settings:
    """The settings stored in a configuration; ValueError for configurations of other runs."""
    data = config.extra.get(EXTRA_KEY) if isinstance(config.extra, dict) else None
    if data is None:
        raise ValueError(f"not a Metal_Expert_2 configuration: extra[{EXTRA_KEY!r}] is missing")
    if config.sensor != SENSOR or config.target != TARGET.name:
        raise ValueError(f"Metal_Expert_2 runs use sensor {SENSOR} and target {TARGET.name}, "
                         f"not {config.sensor}/{config.target}")
    return MetalExpert2Settings.from_dict(data)


def new_model_folder_me2(config: TrainConfig, now: _dt.datetime | None = None) -> str:
    """Folder name of a new run, e.g. ``THz_N2_261021_120000_METAL_ME2_256_withCovered_lr0.0001_pat25``."""
    now = now or _dt.datetime.now()
    covered = "withCovered_" if config.covered_in_training else ""
    return (f"{config.sensor}_N2_{now:%y%m%d_%H%M%S}_{config.target.upper()}_ME2_{config.target_size[0]}_"
            f"{covered}lr{config.learning_rate}_pat{config.patience}")


# ----------------------------------------------------------------------------- data
def train_transform_me2(target_size: tuple[int, int], seed: int | None = None) -> A.Compose:
    """The baseline training augmentation of the THz sensor without CoarseDropout and GaussNoise."""
    base = train_transform(SENSOR, target_size, seed=seed)
    kept = [t for t in base.transforms if type(t).__name__ not in EXCLUDED_AUGMENTATIONS]
    return A.Compose(kept, seed=seed)


def transform_names(compose: A.Compose) -> list[str]:
    """Class names of the top-level transforms of a pipeline."""
    return [type(t).__name__ for t in compose.transforms]


class MetalExpert2Dataset(WoodVitDataset):
    """THz patches of one split (target metal), cut to the channel window of ``settings``.

    ``load`` calls the baseline, which checks the 200 channels of the file, and keeps the window
    channels. With ``train=True`` the augmentation is ``train_transform_me2``.
    """

    def __init__(self, data_root: str | Path, settings: MetalExpert2Settings, split: str, *, train: bool = False,
                 target_size: tuple[int, int] = (256, 256), include_covered: bool = True, probes=None,
                 seed: int | None = None):
        super().__init__(data_root, SENSOR, TARGET, split, train=train, target_size=target_size,
                         include_covered=include_covered, probes=probes, seed=seed)
        self.settings = settings
        if train:
            self.transform = train_transform_me2(self.target_size, seed=seed)

    @property
    def channels(self) -> int:
        return self.settings.channels

    def load(self, index: int):
        image, instance_map, instances = super().load(index)
        window = image[..., self.settings.channel_start:self.settings.channel_stop]
        return np.ascontiguousarray(window), instance_map, instances


def make_datasets_me2(config: TrainConfig, data_root: str | Path) -> tuple[MetalExpert2Dataset, MetalExpert2Dataset]:
    """(training, validation) datasets of a configuration, with the recordings filter of the preset."""
    settings = me2_settings(config)
    common = dict(target_size=config.target_size, include_covered=config.covered_in_training, probes=config.probes)
    train_ds = MetalExpert2Dataset(data_root, settings, "train", train=True, seed=config.seed, **common)
    val_ds = MetalExpert2Dataset(data_root, settings, "val", train=False, **common)
    return train_ds, val_ds


def channel_stats(dataset: MetalExpert2Dataset, progress: bool = False) -> dict:
    """Mean and standard deviation (population) of every window channel over all pixels of all
    images of ``dataset`` (untransformed, native raster). Sums are accumulated in float64."""
    n_channels = dataset.channels
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)
    pixels = 0
    for index in tqdm(range(len(dataset)), desc=f"Channel statistics {dataset.split}", disable=not progress):
        image, _, _ = dataset.load(index)
        flat = image.reshape(-1, n_channels).astype(np.float64)
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        pixels += flat.shape[0]
    mean = total / pixels
    std = np.sqrt(np.maximum(total_sq / pixels - mean ** 2, 0.0))
    if not np.all(std > 0):
        raise ValueError(f"window channels without variance: {np.flatnonzero(std <= 0).tolist()}")
    return {
        "sensor": SENSOR, "split": dataset.split, "recordings": sorted({r.probe for r in dataset.records}),
        "channel_start": dataset.settings.channel_start, "channel_stop": dataset.settings.channel_stop,
        "images": len(dataset), "pixels": int(pixels), "mean": mean.tolist(), "std": std.tolist(),
        "dataset": dataset_identity(find_inventory(dataset.root)),
    }


def read_channel_stats(path: str | Path, data_root: str | Path | None = None) -> dict:
    """A ``channel_stats`` result written as JSON. With ``data_root`` the statistics must come from
    the dataset build in that folder (same inventory hash); a ValueError says otherwise."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no channel statistics at {path}; run metal_expert_2/compute_channel_stats.py first")
    stats = read_json(path)
    if data_root is not None:
        check_dataset_identity(dataset_identity(find_inventory(data_root)), stats.get("dataset"),
                               f"channel statistics {path.name}")
    return stats


# ----------------------------------------------------------------------------- sampling
def covered_metal_flags(dataset: WoodVitDataset) -> list[bool]:
    """True for every image whose ground truth holds at least one instance of a covered-metal class
    (one pass over the ground-truth PNGs; file names are not used)."""
    flags = []
    for record in dataset.records:
        class_map, instance_map = class_and_instance_maps(read_gt(record.gt_path(dataset.root)))
        instances = extract_instances(class_map, instance_map, dataset.target)
        flags.append(any(inst.value in COVERED_METAL_VALUES for inst in instances))
    return flags


def sample_weights(flags, repeat: float) -> torch.Tensor:
    """Sampling weight per image: ``repeat`` for flagged images, 1 otherwise."""
    return torch.tensor([float(repeat) if f else 1.0 for f in flags], dtype=torch.double)


def make_sampler(weights, num_samples: int, seed: int) -> WeightedRandomSampler:
    """Weighted sampling with replacement from a generator seeded with ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=int(num_samples),
                                 replacement=True, generator=generator)


def make_train_loader(dataset, flags, batch_size: int, repeat: float, seed: int, num_workers: int,
                      device: torch.device) -> DataLoader:
    """Training loader like ``train_loop.make_loader`` with the weighted sampler instead of shuffling."""
    sampler = make_sampler(sample_weights(flags, repeat), len(dataset), seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=device.type == "cuda", collate_fn=collate,
                      worker_init_fn=seed_worker if num_workers > 0 else None)


# ----------------------------------------------------------------------------- model
def set_normalisation(model: MaskRCNNWithEvalLoss, settings: MetalExpert2Settings) -> None:
    """Replace the ImageNet normalisation of the model by the window statistics."""
    if not settings.has_stats:
        raise ValueError("the model needs the channel statistics: MetalExpert2Settings.with_stats(...)")
    model.model.transform.image_mean = list(settings.channel_mean)
    model.model.transform.image_std = list(settings.channel_std)


def build_model_me2(config: TrainConfig, settings: MetalExpert2Settings, checkpoint: str | Path | None = None,
                    device=None) -> MaskRCNNWithEvalLoss:
    """The baseline model for the window channels, then: every backbone parameter trainable, the
    first convolution scaled by 3 / C (only without checkpoint), the window normalisation."""
    model = build_model(config, TARGET.num_classes, settings.channels, checkpoint=checkpoint, device=device)
    for parameter in model.model.backbone.parameters():
        parameter.requires_grad_(True)
    if checkpoint is None:
        with torch.no_grad():
            model.model.backbone.body.conv1.weight.mul_(settings.first_conv_scale)
    set_normalisation(model, settings)
    return model


# ----------------------------------------------------------------------------- evaluation
def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _mean_precision(p: np.ndarray) -> float:
    valid = p[p > -1]
    return float(valid.mean()) if valid.size else float("nan")


def coco_evaluate_ap50(coco_gt, results: list[dict], img_ids=None, iou_types=IOU_TYPES) -> tuple[dict, dict]:
    """``coco_utils.coco_evaluate`` plus the AP at IoU 0.5 of every category (``<type>_AP50_<name>``).

    Returns ({"bbox_mAP": ..., "segm_mAP": ...}, {"segm_AP_Metal": ..., "segm_AP50_Metal": ..., ...}).
    """
    metrics, per_category = {}, {}
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    for iou_type in iou_types:
        coco_dt = load_results(coco_gt, results, iou_type)
        if coco_dt is None:
            metrics.update({f"{iou_type}_{n}": 0.0 for n in METRIC_NAMES})
            for c in cats:
                per_category[f"{iou_type}_AP_{c['name']}"] = 0.0
                per_category[f"{iou_type}_AP50_{c['name']}"] = 0.0
            continue
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        if img_ids is not None:
            evaluator.params.imgIds = sorted(int(i) for i in img_ids)
        with _quiet():
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        metrics.update({f"{iou_type}_{n}": float(v) for n, v in zip(METRIC_NAMES, evaluator.stats)})
        if abs(float(evaluator.params.iouThrs[AP50_IOU_INDEX]) - 0.5) > 1e-9:
            raise RuntimeError("the IoU thresholds of COCOeval do not start at 0.5")
        precision = evaluator.eval["precision"]      # [iou, recall, category, area, maxdets]
        for k, cat_id in enumerate(evaluator.params.catIds):
            name = coco_gt.loadCats(int(cat_id))[0]["name"]
            per_category[f"{iou_type}_AP_{name}"] = _mean_precision(precision[:, :, k, 0, -1])
            per_category[f"{iou_type}_AP50_{name}"] = _mean_precision(precision[AP50_IOU_INDEX, :, k, 0, -1])
    return metrics, per_category


def validate_me2(model, loader, device, coco_gt, image_ids, config: TrainConfig,
                 progress: bool = True) -> tuple[float, dict, dict]:
    """(mean validation loss, COCO metrics, per-category AP and AP50) on a validation loader."""
    model.eval()
    total, results = 0.0, []
    for images, targets in tqdm(loader, desc="Validation", disable=not progress, leave=False):
        images, targets = to_device(images, targets, device)
        loss, batch_results = validation_step(model, images, targets, config.val_score_threshold,
                                              config.mask_threshold)
        total += loss
        results.extend(batch_results)
    metrics, per_category = coco_evaluate_ap50(coco_gt, results, img_ids=image_ids)
    return total / max(len(loader), 1), metrics, per_category


# ----------------------------------------------------------------------------- training
def train_me2(config: TrainConfig, data_root: str | Path, run: RunPaths, device=None, progress: bool = True) -> dict:
    """Train a Metal_Expert_2 model into the run folder ``run``; returns a summary.

    Same files as ``train_loop.train``; ``train_log.csv`` has the baseline columns plus the AP50 of
    both categories (segm and bbox), ``training_metadata.json`` names the selection metric.
    """
    device = resolve_device(device)
    settings = me2_settings(config)
    if run.checkpoint.exists():
        raise FileExistsError(f"{run.checkpoint} exists; choose a new model folder")
    set_seed(config.seed)

    train_ds, val_ds = make_datasets_me2(config, data_root)
    flags = covered_metal_flags(train_ds)
    train_loader = make_train_loader(train_ds, flags, config.batch_size, settings.covered_metal_repeat, config.seed,
                                     config.num_workers, device)
    val_loader = make_loader(val_ds, config.batch_size, False, config.num_workers, device)
    gt_dict, _ = build_coco_gt(data_root, SENSOR, TARGET, "val", target_size=config.target_size,
                               include_covered=config.covered_in_training, probes=config.probes)
    coco_gt = to_coco(gt_dict)
    model = build_model_me2(config, settings, device=device)
    frozen = frozen_parameters(model)
    if frozen:
        raise RuntimeError(f"{len(frozen)} parameters are frozen, e.g. {frozen[:3]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = one_cycle(optimizer, config, len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.amp else None

    write_run_config(run, config, dataset_identity(find_inventory(data_root)))     # creates the folder
    category_keys = [f"{prefix}_{name}" for prefix in ("segm_AP", "segm_AP50", "bbox_AP50") for name in TARGET.categories]
    columns = ["epoch", "train_loss", "val_loss", "lr", "seconds", *METRIC_KEYS, *category_keys]
    with run.train_log.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)

    logger = _logger(run)
    writer = SummaryWriter(log_dir=str(run.tensorboard))
    stopper = EarlyStopping(config.patience, config.min_delta)
    metric = settings.selection_metric
    best, best_epoch, epoch_seconds, stopped_early = -math.inf, 0, [], False
    try:
        logger.info(f"run {run.name}: {config.preset} ({config.sensor}, target {config.target}) on {device}; "
                    f"{len(train_ds)} training / {len(val_ds)} validation images, "
                    f"{len(train_loader)} batches of {config.batch_size} per epoch")
        logger.info(f"metal_expert_2: channels {settings.channel_start}..{settings.channel_stop - 1} "
                    f"({settings.channels}), {sum(flags)} training images with covered metal drawn "
                    f"{settings.covered_metal_repeat:g} times as often, selection by {metric}, no frozen parameters")
        for epoch in range(1, config.num_epochs + 1):
            start = time.perf_counter()
            train_loss, lr = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, writer,
                                         progress)
            val_loss, metrics, per_category = validate_me2(model, val_loader, device, coco_gt, val_ds.image_ids,
                                                           config, progress)
            seconds = time.perf_counter() - start
            epoch_seconds.append(seconds)
            values = {**metrics, **per_category}
            if metric not in values:
                raise KeyError(f"selection metric {metric!r} is not among the validation metrics")
            current = values[metric]
            if math.isnan(current):
                raise ValueError(f"{metric} is undefined on the validation split (no annotated instance)")

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": seconds,
                   **metrics, **{k: per_category.get(k, float("nan")) for k in category_keys}}
            with run.train_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([_fmt(row[c]) for c in columns])
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("LR", lr, epoch)
            for name, value in values.items():
                writer.add_scalar(f"Metrics/{name}", value, epoch)
            writer.flush()
            logger.info(f"epoch {epoch}/{config.num_epochs}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                        f"{metric} {current:.4f}, segm AP Metal {per_category.get('segm_AP_Metal', float('nan')):.4f}, "
                        f"segm mAP {metrics['segm_mAP']:.4f}, lr {lr:.3g}, {seconds:.0f} s")

            if current > best:
                best, best_epoch = current, epoch
                save_checkpoint(model, run.checkpoint)
                write_json(run.training / METADATA_FILE, {
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr,
                    "selection_metric": metric, "selection_value": current, **values,
                })
                logger.info(f"epoch {epoch}: new best {metric} {current:.4f}, checkpoint saved")
            if stopper.step(current, epoch):
                stopped_early = epoch < config.num_epochs
                logger.info(f"epoch {epoch}: no improvement of {metric} by more than {config.min_delta} "
                            f"for {config.patience} epochs, stopping (best {best:.4f} in epoch {best_epoch})")
                break
        summary = {
            "model_folder": run.name, "device": str(device), "epochs": len(epoch_seconds),
            "best_epoch": best_epoch, "selection_metric": metric, "best_value": best, "stopped_early": stopped_early,
            "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
            "train_images": len(train_ds), "val_images": len(val_ds), "covered_metal_train_images": int(sum(flags)),
        }
        logger.info("finished: " + ", ".join(f"{k} {_fmt(v)}" for k, v in summary.items()))
        return summary
    finally:
        writer.close()
        _close_logger(logger)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def smoke_step_me2(config: TrainConfig, data_root: str | Path, device=None) -> dict:
    """One training step and one validation step on one batch each; writes no files."""
    device = resolve_device(device)
    settings = me2_settings(config)
    set_seed(config.seed)
    info: dict = {"device": str(device)}

    start = time.perf_counter()
    model = build_model_me2(config, settings, device=device)
    info["model_seconds"] = time.perf_counter() - start
    info["parameters"] = sum(p.numel() for p in model.parameters())
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["frozen_parameters"] = len(frozen_parameters(model))
    info["input_channels"] = model.model.backbone.body.conv1.in_channels
    info["image_mean_range"] = (min(settings.channel_mean), max(settings.channel_mean))
    info["image_std_range"] = (min(settings.channel_std), max(settings.channel_std))

    train_ds, val_ds = make_datasets_me2(config, data_root)
    flags = covered_metal_flags(train_ds)
    info["train_images"], info["val_images"] = len(train_ds), len(val_ds)
    info["covered_metal_train_images"] = int(sum(flags))
    start = time.perf_counter()
    loader = make_train_loader(train_ds, flags, config.batch_size, settings.covered_metal_repeat, config.seed, 0, device)
    images, targets = next(iter(loader))
    val_images, val_targets = next(iter(make_loader(val_ds, config.batch_size, False, 0, device)))
    info["load_seconds_per_image"] = (time.perf_counter() - start) / (len(images) + len(val_images))
    info["image_shape"] = tuple(images[0].shape)
    info["instances"] = [len(t["labels"]) for t in targets]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.amp else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    start = time.perf_counter()
    images, targets = to_device(images, targets, device)
    info["train_losses"] = train_step(model, images, targets, optimizer, scaler)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        info["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    info["train_step_seconds"] = time.perf_counter() - start

    model.eval()
    val_images, val_targets = to_device(val_images, val_targets, device)
    info["val_loss"], results = validation_step(model, val_images, val_targets, config.val_score_threshold,
                                                config.mask_threshold)
    info["val_detections"] = len(results)
    return info


# ----------------------------------------------------------------------------- prediction
def predict_split_me2(run: RunPaths, data_root: str | Path, split: str, device=None, *, probes=None,
                      batch_size: int | None = None, progress: bool = True) -> dict:
    """Predict a split with the checkpoint of a Metal_Expert_2 run (format of ``predict.predict_split``:
    ``inference/pred_<split>.json`` and ``inference/gt_<split>.json``); returns a short summary."""
    device = resolve_device(device)
    config, run_data = read_run_config(run)
    settings = me2_settings(config)
    if not run.checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint in {run.training}")
    identity = dataset_identity(find_inventory(data_root))

    start = time.perf_counter()
    model = build_model_me2(config, settings, checkpoint=run.checkpoint, device=device)
    dataset = MetalExpert2Dataset(data_root, settings, split, train=False, target_size=config.target_size,
                                  include_covered=True, probes=probes)
    results = predict_dataset(model, dataset, device, batch_size or config.batch_size, config.mask_threshold,
                              config.num_workers, progress)
    seconds = time.perf_counter() - start

    write_predictions(run.predictions(split), results, image_ids=dataset.image_ids, sensor=SENSOR,
                      target=TARGET.name, split=split, target_size=config.target_size, dataset=identity,
                      source=f"{run.name}/{run.checkpoint.name} (metal_expert_2)")
    gt_dict, _ = build_coco_gt(data_root, SENSOR, TARGET, split, target_size=config.target_size,
                               include_covered=True, probes=probes)
    write_json(run.ground_truth(split), gt_dict)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "predictions": run.predictions(split), "ground_truth": run.ground_truth(split),
        "images": len(dataset), "detections": len(results), "seconds": seconds,
        "same_dataset_as_training": run_data.get("dataset", {}).get("sha1") == identity["sha1"],
    }


# ----------------------------------------------------------------------------- covered-metal analysis
def metal_pixel_share(images: list[ImageData], conf_threshold: float, metal_category_id: int = 1) -> dict[int, float]:
    """Share of the annotated pixels of every class value in ANALYSIS_VALUES that lie under the union
    of the Metal predictions with a score of at least ``conf_threshold``; NaN without instances."""
    covered: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    for image in images:
        annotations = [a for a in image.gt if int(a["class_value"]) in ANALYSIS_VALUES]
        if not annotations:
            continue
        rles = [p["segmentation"] for p, score in zip(image.preds, image.scores)
                if score >= conf_threshold and int(p["category_id"]) == metal_category_id]
        union = None
        if rles:
            height, width = annotations[0]["segmentation"]["size"]
            union = rle_to_mask(rle_union(rles, int(height), int(width))).astype(bool)
        for ann in annotations:
            mask = rle_to_mask(ann["segmentation"]).astype(bool)
            value = int(ann["class_value"])
            total[value] += int(mask.sum())
            if union is not None:
                covered[value] += int((mask & union).sum())
    return {v: (covered[v] / total[v] if total[v] else float("nan")) for v in ANALYSIS_VALUES}


def covered_metal_table(gt_dict: dict, prediction_data: dict, iou_thresholds=(0.5, 0.3), conf_threshold: float = 0.5,
                        check_identity: bool = True) -> pd.DataFrame:
    """Detection of open metal (70) and the covered-metal classes (110, 120, 130), one row per class
    value and IoU threshold: the number of annotated instances and the shares detected as Metal,
    detected but called Non_metal, and undetected (greedy class-agnostic matching of ``matching`` at
    ``conf_threshold``, binomial standard error ``*_se``), plus ``metal_pixel_share``.

    ``gt_dict`` and ``prediction_data`` are the files of a run (``inference/gt_<split>.json`` and
    ``pred_<split>.json``) with the metal target.
    """
    image_ids = check_inputs(gt_dict, prediction_data, check_identity)
    categories = sorted(gt_dict["categories"], key=lambda c: int(c["id"]))
    if [c["name"] for c in categories] != list(TARGET.categories):
        raise ValueError(f"the analysis needs the categories {TARGET.categories}, got {[c['name'] for c in categories]}")
    images = prepare_images(gt_dict, prediction_data["results"], image_ids)
    pixel_share = metal_pixel_share(images, conf_threshold)
    metal_row, non_metal_row = TARGET.categories
    rows = []
    for iou in iou_thresholds:
        matches = match_images(images, iou, conf_threshold)
        counts = count_instances(images, matches, categories, iou, conf_threshold)
        total = counts[counts["size"] == "total"].set_index("training_class")
        for value in ANALYSIS_VALUES:
            name = BY_VALUE[value].name
            n_metal, n_non, n_und = (int(total.loc[r, name]) for r in (metal_row, non_metal_row, UNDETECTED))
            n = n_metal + n_non + n_und
            row = {"class_value": value, "class_name": name, "iou_threshold": float(iou),
                   "confidence_threshold": float(conf_threshold), "instances": n}
            for key, count in (("detected_as_metal", n_metal), ("detected_as_non_metal", n_non), ("undetected", n_und)):
                share = count / n if n else float("nan")
                row[key] = share
                row[f"{key}_se"] = math.sqrt(share * (1.0 - share) / n) if n else float("nan")
            row["metal_pixel_share"] = pixel_share[value]
            rows.append(row)
    return pd.DataFrame(rows)
