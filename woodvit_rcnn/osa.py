"""Object surface area (OSA): area-weighted shares of correctly sorted, missorted and undetected
material.

With the assignment of ``matching`` every annotated instance is either detected with the right
class (C), detected with the wrong class (M), or undetected (U). Each instance contributes its
whole area ``A`` (the ``area`` field of the ground truth, i.e. the mask area), irrespective of
the degree of overlap:

    C_OSA = (A_TP + A_TN) / A_tot,   M_OSA = (A_FP + A_FN) / A_tot,   U_OSA = (A_Up + A_Un) / A_tot

with C_OSA + M_OSA + U_OSA = 1 whenever there is any annotated area. Predictions without
reference do not enter the shares; they are counted per size class (sized by their own area).
"""
from __future__ import annotations

import pandas as pd

from woodvit_rcnn.matching import SIZE_NAMES, ImageData, ImageMatch, size_class

SHARES: tuple[str, ...] = ("C_OSA", "M_OSA", "U_OSA")
COUNT_PREFIXES: tuple[str, ...] = ("correct", "wrong", "undetected", "pred_without_gt", "gt_instances")
AREAS: tuple[str, ...] = ("correct_area", "wrong_area", "undetected_area")
TOTALS: tuple[str, ...] = ("instance_gt_total", "instance_correct_total", "instance_wrong_total",
                           "instance_undetected_total", "instance_pred_without_gt_total")
COLUMNS: tuple[str, ...] = (
    "iou_threshold", "conf_threshold", "gt_total_area", *SHARES, *TOTALS,
    *(f"{p}_{s}" for p in COUNT_PREFIXES for s in SIZE_NAMES), *AREAS,
)


def _shares(row: dict) -> None:
    total = row["gt_total_area"]
    for share, area in zip(SHARES, AREAS):
        row[share] = row[area] / total if total > 0 else 0.0


def _image_row(image: ImageData, match: ImageMatch) -> dict:
    row = {f"{p}_{s}": 0 for p in COUNT_PREFIXES for s in SIZE_NAMES}
    row.update({a: 0.0 for a in AREAS})
    gt_areas = image.gt_areas
    row["gt_total_area"] = float(gt_areas.sum())
    for g, area in enumerate(gt_areas):
        size = size_class(area)
        row[f"gt_instances_{size}"] += 1
        if not match.detected[g]:
            row[f"undetected_{size}"] += 1
            row["undetected_area"] += float(area)
    for p in range(match.n_kept):
        g = int(match.gt_for_pred[p])
        if g < 0:
            row[f"pred_without_gt_{size_class(image.pred_areas[p])}"] += 1
            continue
        area = float(gt_areas[g])
        if int(image.preds[p]["category_id"]) == int(image.gt[g]["category_id"]):
            row[f"correct_{size_class(area)}"] += 1
            row["correct_area"] += area
        else:
            row[f"wrong_{size_class(area)}"] += 1
            row["wrong_area"] += area
    for total, prefix in zip(TOTALS, ("gt_instances", "correct", "wrong", "undetected", "pred_without_gt")):
        row[total] = sum(row[f"{prefix}_{s}"] for s in SIZE_NAMES)
    _shares(row)
    return row


def osa_per_image(images: list[ImageData], matches: list[ImageMatch], iou_threshold: float,
                  conf_threshold: float) -> pd.DataFrame:
    """One row per image (shares relative to the annotated area of that image)."""
    rows = []
    for image, match in zip(images, matches):
        row = {"image_id": image.image_id, "iou_threshold": iou_threshold, "conf_threshold": conf_threshold}
        row.update(_image_row(image, match))
        rows.append(row)
    return pd.DataFrame(rows, columns=["image_id", *COLUMNS])


def osa_summary(images: list[ImageData], matches: list[ImageMatch], iou_threshold: float,
                conf_threshold: float) -> dict:
    """Totals over all images; the shares are computed from the summed areas."""
    summary = {c: 0 for c in COLUMNS}
    for image, match in zip(images, matches):
        row = _image_row(image, match)
        for key in COLUMNS:
            if key in row and key not in SHARES:
                summary[key] += row[key]
    summary["iou_threshold"] = float(iou_threshold)
    summary["conf_threshold"] = float(conf_threshold)
    _shares(summary)
    return summary


def osa_series(summaries: list[dict], column_prefix: str, key: str) -> pd.DataFrame:
    """Summaries of a threshold series as one table: a row per quantity, a column per threshold.

    ``key`` is the varied threshold (``iou_threshold`` or ``conf_threshold``); the columns are
    named ``<column_prefix><threshold:.2f>``, e.g. ``iou_0.50``.
    """
    table = {"metric": list(COLUMNS)}
    for summary in summaries:
        table[f"{column_prefix}{summary[key]:.2f}"] = [summary[c] for c in COLUMNS]
    return pd.DataFrame(table)
