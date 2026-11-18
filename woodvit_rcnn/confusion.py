"""Confusion counts and derived metrics of a binary target (Wood versus Non_wood, or Metal versus
Non_metal), with category 1 as the positive class.

From the instance counts (rows: predicted class or ``undetected``; columns ``GT_<category>``):

    TP = positive predicted as positive      FN = positive predicted as negative
    FP = negative predicted as positive      TN = negative predicted as negative
    U_p = undetected positive                U_n = undetected negative

Metrics (value used when the denominator is zero in brackets):

    recall_cat1         = TP / (TP + FN + U_p)                          (0)
    precision_cat1      = TP / (TP + FP)                                (1)
    f1_strict           = 2 TP / (2 TP + FP + FN + U_p)                 (1)
    accuracy_classified = (TP + TN) / (TP + TN + FP + FN)               (1)
    coverage            = (TP + TN + FP + FN) / (all annotated)         (1)
    specificity         = TN / (TN + FP + U_n)                          (1)
    norm_mcc_classified = (MCC(TP, FP, FN, TN) + 1) / 2
    norm_mcc_strict     = (MCC(TP, FP + U_n, FN + U_p, TN) + 1) / 2     (undetected count as errors)

An undefined MCC (a zero factor in the denominator) is set to -1, i.e. nMCC = 0.
"""
from __future__ import annotations

import math

import pandas as pd

from woodvit_rcnn.instance_counts import UNDETECTED

COUNTS: tuple[str, ...] = ("TP", "FN", "U_p", "FP", "TN", "U_n")
METRICS: tuple[str, ...] = ("recall_cat1", "precision_cat1", "coverage", "accuracy_classified", "specificity",
                            "f1_strict", "norm_mcc_classified", "norm_mcc_strict")


def _div(num: float, den: float, zero_result: float) -> float:
    return float(num) / float(den) if den != 0 else float(zero_result)


def mcc(tp: float, fp: float, fn: float, tn: float) -> float:
    """Matthews correlation coefficient; -1 if undefined."""
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _div(tp * tn - fp * fn, den, -1.0)


def nmcc(tp: float, fp: float, fn: float, tn: float) -> float:
    return (mcc(tp, fp, fn, tn) + 1.0) / 2.0


def confusion_metrics(TP, FN, U_p, FP, TN, U_n) -> dict:
    """Metrics of one set of confusion counts (argument names as in the paper)."""
    detected = TP + TN + FP + FN
    return {
        "recall_cat1": _div(TP, TP + FN + U_p, 0.0),
        "precision_cat1": _div(TP, TP + FP, 1.0),
        "coverage": _div(detected, detected + U_p + U_n, 1.0),
        "accuracy_classified": _div(TP + TN, detected, 1.0),
        "specificity": _div(TN, FP + TN + U_n, 1.0),
        "f1_strict": _div(2 * TP, 2 * TP + FP + FN + U_p, 1.0),
        "norm_mcc_classified": nmcc(TP, FP, FN, TN),
        "norm_mcc_strict": nmcc(TP, FP + U_n, FN + U_p, TN),
    }


def confusion_counts(rows: pd.DataFrame, positive: str, negative: str) -> dict:
    """TP, FN, U_p, FP, TN, U_n from the count rows of one threshold pair and one size."""
    def cell(row_class: str, gt_class: str) -> int:
        return int(rows.loc[rows["training_class"] == row_class, f"GT_{gt_class}"].sum())

    return {
        "TP": cell(positive, positive), "FN": cell(negative, positive), "U_p": cell(UNDETECTED, positive),
        "FP": cell(positive, negative), "TN": cell(negative, negative), "U_n": cell(UNDETECTED, negative),
    }


def confusion_table(counts: pd.DataFrame, categories: list[dict]) -> pd.DataFrame:
    """Counts and metrics per (IoU threshold, confidence threshold, size) of a count table."""
    if len(categories) != 2:
        raise ValueError(f"confusion metrics need a binary target, got {len(categories)} categories")
    by_id = {int(c["id"]): c["name"] for c in categories}
    positive, negative = by_id[1], by_id[2]
    rows = []
    for (iou, conf, size), group in counts.groupby(["iou_threshold", "confidence_threshold", "size"], sort=True):
        cm = confusion_counts(group, positive, negative)
        rows.append({"iou_threshold": iou, "confidence_threshold": conf, "size": size, **cm,
                     **confusion_metrics(**cm)})
    return pd.DataFrame(rows, columns=["iou_threshold", "confidence_threshold", "size", *COUNTS, *METRICS])
