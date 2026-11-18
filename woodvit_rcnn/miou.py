"""Pixel-wise IoU per class (mIoU).

Per image, all annotated masks of a class are merged into one mask, and so are all predicted
masks of that class with a confidence of at least ``score_threshold`` (default 0.5). The IoU of a
class in an image is TP / (TP + FP + FN) over the pixels; it is undefined (NaN) if neither the
ground truth nor the predictions contain the class. Per class, the IoU is averaged over the images
where it is defined; the mIoU is the mean of these class means.

Per image, also the pixel accuracy of the merged masks of all classes and, per class, pixel
accuracy, precision and recall are reported (NaN where undefined).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from woodvit_rcnn.coco_utils import rle_to_mask


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def _union(rles: list[dict], height: int, width: int) -> np.ndarray:
    merged = np.zeros((height, width), dtype=bool)
    for rle in rles:
        merged |= rle_to_mask(rle).astype(bool)
    return merged


def _pixel_counts(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(pred)) - tp
    fn = int(np.count_nonzero(gt)) - tp
    return tp, fp, fn, gt.size - tp - fp - fn


def mean_iou(gt_dict: dict, results: list[dict], image_ids=None, score_threshold: float = 0.5
             ) -> tuple[pd.DataFrame, dict]:
    """(table per image, summary) with the summary keys ``pixel_accuracy``, ``iou_<class>``,
    ``acc_<class>``, ``prec_<class>``, ``rec_<class>`` (means over images) and ``mIoU``."""
    images = {int(img["id"]): img for img in gt_dict["images"]}
    image_ids = sorted(images) if image_ids is None else sorted(int(i) for i in image_ids)
    categories = sorted(gt_dict["categories"], key=lambda c: int(c["id"]))
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for ann in gt_dict["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)
    for res in results:
        if float(res["score"]) >= score_threshold:
            pred_by_image[int(res["image_id"])].append(res)

    rows = []
    for img_id in image_ids:
        h, w = int(images[img_id]["height"]), int(images[img_id]["width"])
        gt_anns, preds = gt_by_image.get(img_id, []), pred_by_image.get(img_id, [])
        tp, fp, fn, tn = _pixel_counts(_union([a["segmentation"] for a in gt_anns], h, w),
                                       _union([p["segmentation"] for p in preds], h, w))
        row = {"image_id": img_id, "pixel_accuracy": _ratio(tp + tn, tp + fp + fn + tn)}
        class_ious = []
        for cat in categories:
            cid, name = int(cat["id"]), cat["name"]
            tp, fp, fn, tn = _pixel_counts(
                _union([a["segmentation"] for a in gt_anns if int(a["category_id"]) == cid], h, w),
                _union([p["segmentation"] for p in preds if int(p["category_id"]) == cid], h, w))
            row[f"iou_{name}"] = _ratio(tp, tp + fp + fn)
            row[f"acc_{name}"] = _ratio(tp + tn, tp + fp + fn + tn)
            row[f"prec_{name}"] = _ratio(tp, tp + fp)
            row[f"rec_{name}"] = _ratio(tp, tp + fn)
            class_ious.append(row[f"iou_{name}"])
        defined = [v for v in class_ious if not np.isnan(v)]
        row["mIoU"] = float(np.mean(defined)) if defined else float("nan")
        rows.append(row)

    columns = ["image_id", "pixel_accuracy"]
    for cat in categories:
        columns += [f"{k}_{cat['name']}" for k in ("iou", "acc", "prec", "rec")]
    table = pd.DataFrame(rows, columns=[*columns, "mIoU"])

    summary = {c: float(table[c].mean(skipna=True)) if len(table) else float("nan") for c in columns[1:]}
    class_means = [summary[f"iou_{c['name']}"] for c in categories]
    class_means = [v for v in class_means if not np.isnan(v)]
    summary["mIoU"] = float(np.mean(class_means)) if class_means else float("nan")
    return table, summary
