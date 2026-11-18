"""Training of one baseline model.

``train(config, data_root, run)`` trains a Mask R-CNN on the training split and validates it
after every epoch:

* Training data with augmentation; covered recordings only if ``config.covered_in_training``.
  The validation split uses the same recordings filter, without augmentation.
* AdamW (``learning_rate``, ``weight_decay``) with a one-cycle schedule: peak ``learning_rate``,
  warm-up over ``warmup_fraction`` of all steps, one scheduler step per batch.
* On CUDA (and with ``config.amp``) the training forward pass runs under bfloat16 autocast with a
  gradient scaler. Validation runs in float32. CPU training works as well (slow).
* Validation per epoch: loss (without updating BatchNorm statistics), predictions with a score
  of at least ``val_score_threshold``, COCO evaluation against the ground truth of exactly the
  validation images.
* The checkpoint is written after the first epoch and whenever the validation mask mAP improves.
  Training stops after ``patience`` epochs in a row without an improvement of more than
  ``min_delta``.

Files of a run (see ``run_io.RunPaths``): ``training/config.json`` (written before the first
epoch, with the identity of the dataset), ``checkpoint.pth``, ``training_metadata.json`` (epoch and
metrics of the checkpoint), ``train_log.csv`` (one row per epoch, decimal point), ``train.log``
and TensorBoard scalars.
"""
from __future__ import annotations

import csv
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from woodvit_rcnn.classes import get_target
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import IOU_TYPES, METRIC_NAMES, coco_evaluate, outputs_to_results, to_coco
from woodvit_rcnn.dataset import WoodVitDataset, collate, seed_worker
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.model import build_model, resolve_device, save_checkpoint
from woodvit_rcnn.presets import TrainConfig
from woodvit_rcnn.run_io import RunPaths, dataset_identity, write_json, write_run_config

METRIC_KEYS: tuple[str, ...] = tuple(f"{t}_{n}" for t in IOU_TYPES for n in METRIC_NAMES)
SELECTION_METRIC = "segm_mAP"
TB_LOSS_EVERY = 10          # batches between two TensorBoard points of the loss components
METADATA_FILE = "training_metadata.json"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_cycle(optimizer, config: TrainConfig, steps_per_epoch: int) -> torch.optim.lr_scheduler.OneCycleLR:
    """One-cycle schedule over all training steps with peak ``config.learning_rate``.

    torch fails (division by zero) if the warm-up is exactly one step long; only in such very
    short runs the warm-up is lengthened to two steps.
    """
    total = config.num_epochs * steps_per_epoch
    pct_start = config.warmup_fraction
    if total > 2 and abs(pct_start * total - 1.0) < 1e-9:
        pct_start = 2.0 / total
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=total,
                                               pct_start=pct_start)


class EarlyStopping:
    """Stops after ``patience`` epochs in a row without an improvement of more than ``min_delta``."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -math.inf
        self.best_epoch = 0
        self.counter = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record the metric of an epoch; True if training should stop."""
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.counter = value, epoch, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ----------------------------------------------------------------------------- data
def make_datasets(config: TrainConfig, data_root: str | Path) -> tuple[WoodVitDataset, WoodVitDataset]:
    """(training, validation) datasets of a configuration."""
    common = dict(target_size=config.target_size, include_covered=config.covered_in_training, probes=config.probes,
                  channel_window=config.channel_window)
    train_ds = WoodVitDataset(data_root, config.sensor, config.target, "train", train=True, seed=config.seed, **common)
    val_ds = WoodVitDataset(data_root, config.sensor, config.target, "val", train=False, **common)
    return train_ds, val_ds


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=device.type == "cuda", collate_fn=collate,
                      worker_init_fn=seed_worker if num_workers > 0 else None)


def to_device(images, targets, device: torch.device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
               for t in targets]
    return images, targets


# ----------------------------------------------------------------------------- steps
def train_step(model, images, targets, optimizer, scaler) -> dict[str, float]:
    """One optimisation step; returns the loss components and the total loss as floats."""
    device_type = images[0].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=scaler is not None):
        losses = model(images, targets)
        loss = sum(losses.values())
    value = float(loss.item())
    if not math.isfinite(value):
        raise FloatingPointError(f"training loss is not finite ({value}): "
                                 + ", ".join(f"{k}={float(v):.4g}" for k, v in losses.items()))
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    out = {k: float(v.item()) for k, v in losses.items()}
    out["loss"] = value
    return out


@torch.no_grad()
def validation_step(model, images, targets, score_threshold: float, mask_threshold: float) -> tuple[float, list[dict]]:
    """Validation loss and COCO results of one batch (model in evaluation mode)."""
    outputs = model(images)
    losses = model(images, targets)
    image_ids = [int(t["image_id"]) for t in targets]
    results = outputs_to_results(outputs, image_ids, score_threshold=score_threshold, mask_threshold=mask_threshold)
    return float(sum(losses.values()).item()), results


def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch: int, writer=None,
                progress: bool = True) -> tuple[float, float]:
    """(mean training loss, learning rate after the epoch)."""
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"Train {epoch}", disable=not progress, leave=False)
    for step, (images, targets) in enumerate(bar):
        images, targets = to_device(images, targets, device)
        losses = train_step(model, images, targets, optimizer, scaler)
        scheduler.step()
        total += losses["loss"]
        if writer is not None and step % TB_LOSS_EVERY == 0:
            global_step = (epoch - 1) * len(loader) + step
            for name, value in losses.items():
                writer.add_scalar(f"Loss/{'total' if name == 'loss' else name}", value, global_step)
        bar.set_postfix(loss=f"{losses['loss']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total / max(len(loader), 1), float(scheduler.get_last_lr()[0])


def validate(model, loader, device, coco_gt, image_ids, config: TrainConfig,
             progress: bool = True) -> tuple[float, dict, dict]:
    """(mean validation loss, COCO metrics, per-category AP) on a validation loader."""
    model.eval()
    total, results = 0.0, []
    for images, targets in tqdm(loader, desc="Validation", disable=not progress, leave=False):
        images, targets = to_device(images, targets, device)
        loss, batch_results = validation_step(model, images, targets, config.val_score_threshold,
                                              config.mask_threshold)
        total += loss
        results.extend(batch_results)
    metrics, per_category = coco_evaluate(coco_gt, results, img_ids=image_ids)
    return total / max(len(loader), 1), metrics, per_category


# ----------------------------------------------------------------------------- logging
def _logger(run: RunPaths) -> logging.Logger:
    logger = logging.getLogger(f"woodvit_rcnn.train.{run.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(run.log_file, mode="a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


# ----------------------------------------------------------------------------- training
def train(config: TrainConfig, data_root: str | Path, run: RunPaths, device=None, progress: bool = True) -> dict:
    """Train a model into the run folder ``run``; returns a summary of the training."""
    device = resolve_device(device)
    if run.checkpoint.exists():
        raise FileExistsError(f"{run.checkpoint} exists; choose a new model folder")
    set_seed(config.seed)

    target = get_target(config.target)
    # data first (fails early, and does not draw from the global random generators)
    train_ds, val_ds = make_datasets(config, data_root)
    train_loader = make_loader(train_ds, config.batch_size, True, config.num_workers, device)
    val_loader = make_loader(val_ds, config.batch_size, False, config.num_workers, device)
    gt_dict, _ = build_coco_gt(data_root, config.sensor, target, "val", target_size=config.target_size,
                               include_covered=config.covered_in_training, probes=config.probes)
    coco_gt = to_coco(gt_dict)
    category_keys = [f"segm_AP_{c}" for c in target.categories]
    model = build_model(config, target.num_classes, config.input_channels, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = one_cycle(optimizer, config, len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.amp else None

    write_run_config(run, config, dataset_identity(find_inventory(data_root)))     # creates the folder
    columns =["epoch", "train_loss", "val_loss", "lr", "seconds", *METRIC_KEYS, *category_keys]
    with run.train_log.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)

    logger = _logger(run)
    writer = SummaryWriter(log_dir=str(run.tensorboard))
    stopper = EarlyStopping(config.patience, config.min_delta)
    best, best_epoch, epoch_seconds, stopped_early = -math.inf, 0, [], False
    try:
        logger.info(f"run {run.name}: {config.preset} ({config.sensor}, target {config.target}) on {device}; "
                    f"{len(train_ds)} training / {len(val_ds)} validation images, "
                    f"{len(train_loader)} batches of {config.batch_size} per epoch")
        for epoch in range(1, config.num_epochs + 1):
            start = time.perf_counter()
            train_loss, lr = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, writer,
                                         progress)
            val_loss, metrics, per_category = validate(model, val_loader, device, coco_gt, val_ds.image_ids, config,
                                                       progress)
            seconds = time.perf_counter() - start
            epoch_seconds.append(seconds)
            current = metrics[SELECTION_METRIC]

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": seconds,
                   **metrics, **{k: per_category.get(k, float("nan")) for k in category_keys}}
            with run.train_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([_fmt(row[c]) for c in columns])
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("LR", lr, epoch)
            for name, value in {**metrics, **per_category}.items():
                writer.add_scalar(f"Metrics/{name}", value, epoch)
            writer.flush()
            logger.info(f"epoch {epoch}/{config.num_epochs}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                        f"segm mAP {current:.4f}, bbox mAP {metrics['bbox_mAP']:.4f}, lr {lr:.3g}, {seconds:.0f} s")

            if current > best:
                best, best_epoch = current, epoch
                save_checkpoint(model, run.checkpoint)
                write_json(run.training / METADATA_FILE, {
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr,
                    **metrics, **per_category,
                })
                logger.info(f"epoch {epoch}: new best {SELECTION_METRIC} {current:.4f}, checkpoint saved")
            if stopper.step(current, epoch):
                stopped_early = epoch < config.num_epochs
                logger.info(f"epoch {epoch}: no improvement of {SELECTION_METRIC} by more than {config.min_delta} "
                            f"for {config.patience} epochs, stopping (best {best:.4f} in epoch {best_epoch})")
                break
        summary = {
            "model_folder": run.name, "device": str(device), "epochs": len(epoch_seconds),
            "best_epoch": best_epoch, f"best_{SELECTION_METRIC}": best, "stopped_early": stopped_early,
            "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
            "train_images": len(train_ds), "val_images": len(val_ds),
        }
        logger.info("finished: " + ", ".join(f"{k} {_fmt(v)}" for k, v in summary.items()))
        return summary
    finally:
        writer.close()
        _close_logger(logger)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def smoke_step(config: TrainConfig, data_root: str | Path, device=None) -> dict:
    """One training step and one validation step on one batch each; writes no files.

    Builds the model (downloads the COCO weights on first use), loads the first batches of the
    training and validation split and returns losses, detections and timings.
    """
    device = resolve_device(device)
    set_seed(config.seed)
    target = get_target(config.target)
    info: dict = {"device": str(device)}

    start = time.perf_counter()
    model = build_model(config, target.num_classes, config.input_channels, device=device)
    info["model_seconds"] = time.perf_counter() - start
    info["parameters"] = sum(p.numel() for p in model.parameters())
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds, val_ds = make_datasets(config, data_root)
    info["train_images"], info["val_images"] = len(train_ds), len(val_ds)
    start = time.perf_counter()
    images, targets = next(iter(make_loader(train_ds, config.batch_size, True, 0, device)))
    val_images, val_targets = next(iter(make_loader(val_ds, config.batch_size, False, 0, device)))
    info["load_seconds_per_image"] = (time.perf_counter() - start) / (len(images) + len(val_images))
    info["image_shape"] = tuple(images[0].shape)
    info["instances"] = [len(t["labels"]) for t in targets]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
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
