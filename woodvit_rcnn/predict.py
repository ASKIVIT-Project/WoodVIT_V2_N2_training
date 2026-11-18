"""Predictions of a trained model on a full split.

``predict_split(run, data_root, split)`` loads the checkpoint of a run, predicts every image of
the split (covered recordings included, whatever the model was trained on) and writes

* ``inference/pred_<split>.json``: all detections of the model (torchvision keeps detections with
  a score of at least 0.05, at most 100 per image), masks binarised at ``mask_threshold``, at the
  target size of the run; plus the list of all predicted images and the identity of the dataset
  (format of ``run_io.write_predictions``),
* ``inference/gt_<split>.json``: the COCO ground truth of the same images for the target of the
  run (``coco_gt.build_coco_gt``).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from woodvit_rcnn.classes import get_target
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import outputs_to_results
from woodvit_rcnn.dataset import WoodVitDataset
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.model import build_model, resolve_device
from woodvit_rcnn.run_io import RunPaths, dataset_identity, read_run_config, write_json, write_predictions
from woodvit_rcnn.sensors import get_sensor
from woodvit_rcnn.train_loop import make_loader, to_device


@torch.no_grad()
def predict_dataset(model, dataset: WoodVitDataset, device, batch_size: int, mask_threshold: float,
                    num_workers: int = 0, progress: bool = True) -> list[dict]:
    """COCO results (score >= 0) of a model in evaluation mode on all images of a dataset."""
    device = resolve_device(device)
    model.eval()
    results = []
    loader = make_loader(dataset, batch_size, False, num_workers, device)
    for images, targets in tqdm(loader, desc=f"Predict {dataset.sensor.name} {dataset.split}", disable=not progress):
        images, _ = to_device(images, [], device)
        outputs = model(images)
        image_ids = [int(t["image_id"]) for t in targets]
        results.extend(outputs_to_results(outputs, image_ids, score_threshold=0.0, mask_threshold=mask_threshold))
    return results


def predict_split(run: RunPaths, data_root: str | Path, split: str, device=None, keep_stored_order: bool = False,
                  *, probes=None, batch_size: int | None = None, progress: bool = True) -> dict:
    """Predict a split with the checkpoint of ``run``; returns a short summary.

    ``keep_stored_order`` feeds B, G, R data unflipped (only for models trained that way).
    ``probes`` restricts the split to some recordings (None = full split).
    """
    device = resolve_device(device)
    config, run_data = read_run_config(run)
    if not run.checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint in {run.training}")
    sensor = get_sensor(config.sensor)
    target = get_target(config.target)
    identity = dataset_identity(find_inventory(data_root))

    start = time.perf_counter()
    model = build_model(config, target.num_classes, config.input_channels, checkpoint=run.checkpoint,
                        device=device)
    dataset = WoodVitDataset(data_root, sensor.name, target, split, train=False, target_size=config.target_size,
                             include_covered=True, probes=probes, keep_stored_order=keep_stored_order,
                             channel_window=config.channel_window)
    results = predict_dataset(model, dataset, device, batch_size or config.batch_size, config.mask_threshold,
                              config.num_workers, progress)
    seconds = time.perf_counter() - start

    source = f"{run.name}/{run.checkpoint.name}" + (" (stored channel order)" if keep_stored_order else "")
    write_predictions(run.predictions(split), results, image_ids=dataset.image_ids, sensor=sensor.name,
                      target=target.name, split=split, target_size=config.target_size, dataset=identity,
                      source=source)
    gt_dict, _ = build_coco_gt(data_root, sensor.name, target, split, target_size=config.target_size,
                               include_covered=True, probes=probes)
    write_json(run.ground_truth(split), gt_dict)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "predictions": run.predictions(split), "ground_truth": run.ground_truth(split),
        "images": len(dataset), "detections": len(results), "seconds": seconds,
        "same_dataset_as_training": run_data.get("dataset", {}).get("sha1") == identity["sha1"],
    }
