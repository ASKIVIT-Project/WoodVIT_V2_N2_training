"""Greedy, class-agnostic assignment of predictions to annotated instances.

One assignment serves the object surface area (OSA) and the instance counts, so both always see
the same detected and undetected instances. Per image:

1. predictions below the confidence threshold are dropped,
2. the remaining predictions are processed in descending order of confidence (equal scores keep
   their order in the result list),
3. each prediction takes the not yet assigned annotated instance with the largest mask IoU,
   irrespective of the classes (on equal IoU the instance that comes first in the ground truth),
4. if this IoU reaches the IoU threshold, the instance counts as detected; otherwise the
   prediction remains without reference and the instance stays available.

Size classes follow COCO: small below 32**2, medium below 96**2, large otherwise, in pixels of
the evaluated raster. Instances are sized by the ``area`` field of the ground truth (the mask area
in ground truth built by ``coco_gt.build_coco_gt``). Predictions are sized by their mask area;
``pred_area="box"`` uses the area of the predicted box instead (the value pycocotools assigns
when loading results that carry a box), for comparisons with evaluations that sized predictions
that way.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils

from woodvit_rcnn.coco_utils import rle_area

SIZE_NAMES: tuple[str, ...] = ("small", "medium", "large")
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2
PRED_AREA_SOURCES: tuple[str, ...] = ("mask", "box")


def size_class(area: float) -> str:
    """COCO size class of an area in pixels."""
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


@dataclass
class ImageData:
    """Ground truth and predictions of one image, prepared for repeated matching."""

    image_id: int
    gt: list[dict]              # annotations in ground-truth order
    preds: list[dict]           # predictions in descending order of score
    scores: np.ndarray          # (n_pred,)
    pred_areas: np.ndarray      # (n_pred,)
    ious: np.ndarray            # (n_pred, n_gt) mask IoU

    @property
    def gt_areas(self) -> np.ndarray:
        return np.array([float(a["area"]) for a in self.gt], dtype=float)


@dataclass(frozen=True)
class ImageMatch:
    """Assignment of one image at one IoU and confidence threshold."""

    image_id: int
    n_kept: int                 # predictions at or above the confidence threshold (a prefix of preds)
    gt_for_pred: np.ndarray     # (n_kept,) index of the assigned instance, -1 = without reference
    detected: np.ndarray        # (n_gt,) bool, instance detected


def _pred_area(pred: dict, source: str) -> float:
    if source == "mask":
        return float(rle_area(pred["segmentation"]))
    bbox = pred["bbox"]
    return float(bbox[2]) * float(bbox[3])


def prepare_images(gt_dict: dict, results: list[dict], image_ids=None, pred_area: str = "mask") -> list[ImageData]:
    """Group ground truth and predictions by image and compute the mask IoU matrices once.

    ``image_ids`` are the evaluated images (default: all images of the ground truth). A prediction
    on any other image raises a ValueError.
    """
    if pred_area not in PRED_AREA_SOURCES:
        raise ValueError(f"pred_area must be one of {PRED_AREA_SOURCES}, got {pred_area!r}")
    if image_ids is None:
        image_ids = [img["id"] for img in gt_dict["images"]]
    image_ids = sorted({int(i) for i in image_ids})
    wanted = set(image_ids)
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in gt_dict["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)
    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for res in results:
        img_id = int(res["image_id"])
        if img_id not in wanted:
            raise ValueError(f"prediction on image {img_id}, which is not among the evaluated images")
        preds_by_image[img_id].append(res)

    images = []
    for img_id in image_ids:
        gt = gt_by_image.get(img_id, [])
        preds = sorted(preds_by_image.get(img_id, []), key=lambda r: float(r["score"]), reverse=True)
        scores = np.array([float(p["score"]) for p in preds], dtype=float)
        areas = np.array([_pred_area(p, pred_area) for p in preds], dtype=float)
        if preds and gt:
            ious = np.asarray(mask_utils.iou([p["segmentation"] for p in preds], [g["segmentation"] for g in gt],
                                             [0] * len(gt)), dtype=float).reshape(len(preds), len(gt))
        else:
            ious = np.zeros((len(preds), len(gt)), dtype=float)
        images.append(ImageData(img_id, gt, preds, scores, areas, ious))
    return images


def greedy_match(ious: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy assignment on an IoU matrix whose rows are already in descending score order.

    Returns, per row, the column of the assigned instance or -1.
    """
    n_pred, n_gt = ious.shape
    gt_for_pred = np.full(n_pred, -1, dtype=int)
    if n_gt == 0:
        return gt_for_pred
    free = np.ones(n_gt, dtype=bool)
    for p in range(n_pred):
        candidates = np.where(free, ious[p], -1.0)
        g = int(np.argmax(candidates))          # first maximum = first instance in ground-truth order
        if free[g] and candidates[g] >= iou_threshold:
            free[g] = False
            gt_for_pred[p] = g
    return gt_for_pred


def match_image(image: ImageData, iou_threshold: float, score_threshold: float) -> ImageMatch:
    n_kept = int(np.count_nonzero(image.scores >= score_threshold))
    gt_for_pred = greedy_match(image.ious[:n_kept], iou_threshold)
    detected = np.zeros(len(image.gt), dtype=bool)
    detected[gt_for_pred[gt_for_pred >= 0]] = True
    return ImageMatch(image.image_id, n_kept, gt_for_pred, detected)


def match_images(images: list[ImageData], iou_threshold: float, score_threshold: float) -> list[ImageMatch]:
    """Assignment of all images at one IoU threshold and one confidence threshold."""
    return [match_image(image, iou_threshold, score_threshold) for image in images]
