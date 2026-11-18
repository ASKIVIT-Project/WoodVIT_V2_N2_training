"""COCO helpers: run-length encoding, conversion of model outputs, and COCO evaluation.

Areas: the ``area`` of every ground-truth annotation is its mask area. pycocotools sets the
area of loaded results from the box when a ``bbox`` is present; for the mask evaluation the
area of each result is reset to its mask area, so the size classes (small, medium, large) refer
to mask areas for ground truth and results alike. The box evaluation keeps box areas for the
results, as in the COCO protocol.

Boxes are [x, y, w, h] with w = x_max - x_min + 1 (inclusive pixel extent).
"""
from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES: tuple[str, ...] = (
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)
IOU_TYPES: tuple[str, ...] = ("bbox", "segm")


# ----------------------------------------------------------------------------- masks and RLE
def _rle_bytes(rle: dict) -> dict:
    counts = rle["counts"]
    return {"size": list(rle["size"]), "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}


def mask_to_rle(mask: np.ndarray) -> dict:
    """Binary (H, W) mask -> COCO RLE with ``counts`` as str (JSON serialisable)."""
    rle = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"].decode("utf-8")}


def rle_to_mask(rle: dict) -> np.ndarray:
    return mask_utils.decode(_rle_bytes(rle)).astype(np.uint8)


def rle_area(rle: dict) -> int:
    return int(mask_utils.area(_rle_bytes(rle)))


def rle_iou(a: dict, b: dict) -> float:
    return float(mask_utils.iou([_rle_bytes(a)], [_rle_bytes(b)], [0])[0][0])


def rle_union(rles: list[dict], height: int, width: int) -> dict:
    if not rles:
        return mask_to_rle(np.zeros((height, width), dtype=np.uint8))
    merged = mask_utils.merge([_rle_bytes(r) for r in rles], intersect=False)
    return {"size": [int(merged["size"][0]), int(merged["size"][1])], "counts": merged["counts"].decode("utf-8")}


def mask_bbox(mask: np.ndarray) -> list[float]:
    """[x, y, w, h] of a binary mask with inclusive pixel extent; [0, 0, 0, 0] for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = int(xs.min()), int(ys.min())
    return [float(x0), float(y0), float(xs.max() - x0 + 1), float(ys.max() - y0 + 1)]


# ----------------------------------------------------------------------------- model outputs
def outputs_to_results(outputs, image_ids, score_threshold: float = 0.0, mask_threshold: float = 0.5) -> list[dict]:
    """torchvision Mask R-CNN outputs -> COCO results (one dict per detection)."""
    results = []
    for output, img_id in zip(outputs, image_ids):
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()
        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            results.append({
                "image_id": int(img_id),
                "category_id": int(labels[i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
                "segmentation": mask_to_rle(masks[i, 0] > mask_threshold),
            })
    return results


# ----------------------------------------------------------------------------- COCO objects
def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def to_coco(dataset: dict) -> COCO:
    """pycocotools COCO object from a ground-truth dict, without writing a file."""
    coco = COCO()
    coco.dataset = copy.deepcopy(dataset)
    with _quiet():
        coco.createIndex()
    return coco


def load_results(coco_gt: COCO, results: list[dict], iou_type: str) -> COCO | None:
    """Results as COCO object for one IoU type; None if there are no results."""
    if not results:
        return None
    with _quiet():
        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
    if iou_type == "segm":
        for ann in coco_dt.dataset["annotations"]:
            ann["area"] = float(rle_area(ann["segmentation"]))
    return coco_dt


def coco_evaluate(coco_gt: COCO, results: list[dict], img_ids=None, iou_types=IOU_TYPES,
                  verbose: bool = False) -> tuple[dict, dict]:
    """COCO metrics and per-category AP.

    ``img_ids`` restricts the evaluation to these images (e.g. the images a model was run on).
    Returns ({"bbox_mAP": ..., "segm_mAP": ..., ...}, {"segm_AP_Wood": ..., ...}). Without any
    result all values are 0.
    """
    metrics, per_category = {}, {}
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    for iou_type in iou_types:
        coco_dt = load_results(coco_gt, results, iou_type)
        if coco_dt is None:
            metrics.update({f"{iou_type}_{n}": 0.0 for n in METRIC_NAMES})
            per_category.update({f"{iou_type}_AP_{c['name']}": 0.0 for c in cats})
            continue
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        if img_ids is not None:
            evaluator.params.imgIds = sorted(int(i) for i in img_ids)
        ctx = contextlib.nullcontext() if verbose else _quiet()
        with ctx:
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        metrics.update({f"{iou_type}_{n}": float(v) for n, v in zip(METRIC_NAMES, evaluator.stats)})
        precision = evaluator.eval["precision"]      # [iou, recall, category, area, maxdets]
        for k, cat_id in enumerate(evaluator.params.catIds):
            p = precision[:, :, k, 0, -1]
            valid = p[p > -1]
            name = coco_gt.loadCats(int(cat_id))[0]["name"]      # COCOeval turns the ids into numpy ints
            per_category[f"{iou_type}_AP_{name}"] = float(valid.mean()) if valid.size else float("nan")
    return metrics, per_category
