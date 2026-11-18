"""Detection of covered metal (class values 110, 120, 130) and of the objects that cover it, for any
prediction file of a binary target, per recording group and per class value.

The paper metrics hardly show covered metal: it is a few per cent of the annotated area. This
report answers the question directly. With the greedy class-agnostic matching of ``matching`` at
one confidence threshold, every annotated instance of a class value is either detected and given
category 1, detected and given category 2, or undetected. The table lists these shares (with a
binomial standard error) per IoU threshold, per recording group (``rec48``, ``rec49``, ``rec51``,
``other``, ``all``) and per class value, plus the share of the annotated pixels of the class value
that lie under predictions of the "interest" category: the category that covered metal belongs
to in the target of the ground truth (Non_wood in the main target, Metal in the metal target).

``occluded_summary`` pools the three covered-metal classes and names the one number the late fusion
is tuned for: the share of covered-metal instances detected and given the interest category.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from woodvit_rcnn.classes import BY_VALUE, COVERED_METAL_VALUES, MAIN, METAL, Target
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import rle_to_mask, rle_union
from woodvit_rcnn.evaluate import check_inputs, data_root_for
from woodvit_rcnn.instance_counts import UNDETECTED, count_instances
from woodvit_rcnn.inventory import COVERED_TOP_RECORDINGS
from woodvit_rcnn.matching import ImageData, match_images, prepare_images
from woodvit_rcnn.run_io import split_image_id, write_json
from woodvit_rcnn.sensors import LF_INDEX

COVERED_VALUES: tuple[int, ...] = tuple(sorted(COVERED_METAL_VALUES))      # 110, 120, 130
COVERED_RECORDINGS: tuple[int, ...] = (48, 49, 51)      # covered metal: under chipboard, solid wood, plastic
FOCUS_VALUES: tuple[int, ...] = (110, 120, 130, 70, 10, 20, 80)
COVER_OF: dict[int, tuple[int, int]] = {48: (10, 110), 49: (20, 120), 51: (80, 130)}    # recording -> (cover, metal)
GROUP_OTHER = "other"
GROUP_ALL = "all"
TABLE_FILE = "occluded_cases.csv"
SUMMARY_FILE = "occluded_summary.json"
DEFAULT_IOUS: tuple[float, ...] = (0.5, 0.3)
DEFAULT_CONFS: tuple[float, ...] = (0.5, 0.8)
KEY_COLUMNS: tuple[str, ...] = ("group", "class_value", "class_name", "iou_threshold", "confidence_threshold",
                                "interest_category", "instances", "n_cat1", "n_cat2", "n_undetected")


def recording_groups(recordings=COVERED_RECORDINGS) -> dict[str, tuple[int, ...] | None]:
    """Group name -> recordings; None stands for all recordings, ``other`` for all but the covered ones."""
    groups: dict[str, tuple[int, ...] | None] = {f"rec{r}": (int(r),) for r in recordings}
    groups[GROUP_OTHER] = ()
    groups[GROUP_ALL] = None
    return groups


def _in_group(recording: int, group: tuple[int, ...] | None, covered) -> bool:
    if group is None:
        return True
    if group == ():
        return recording not in covered
    return recording in group


def target_of(gt_dict: dict) -> Target:
    """The training target of a ground truth (from its info, else from its category names)."""
    name = (gt_dict.get("info") or {}).get("target")
    names = tuple(c["name"] for c in sorted(gt_dict["categories"], key=lambda c: int(c["id"])))
    for target in (MAIN, METAL):
        if name == target.name or (name is None and names == target.categories):
            return target
    raise ValueError(f"the report needs the main or the metal target, got {name!r} with categories {names}")


def pixel_share_under_category(images: list[ImageData], conf_threshold: float, category_id: int,
                               values) -> dict[int, float]:
    """Share of the annotated pixels of every class value in ``values`` under the union of the
    predictions of ``category_id`` with a score of at least ``conf_threshold``; NaN without pixels."""
    values = tuple(int(v) for v in values)
    covered = {v: 0 for v in values}
    total = {v: 0 for v in values}
    for image in images:
        annotations = [a for a in image.gt if int(a["class_value"]) in values]
        if not annotations:
            continue
        rles = [p["segmentation"] for p, score in zip(image.preds, image.scores)
                if score >= conf_threshold and int(p["category_id"]) == int(category_id)]
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
    return {v: (covered[v] / total[v] if total[v] else math.nan) for v in values}


def _share(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    share = count / n
    return share, math.sqrt(share * (1.0 - share) / n)


def occluded_case_table(gt_dict: dict, prediction_data: dict, iou_thresholds=DEFAULT_IOUS,
                        conf_thresholds=DEFAULT_CONFS, values=FOCUS_VALUES, groups: dict | None = None,
                        check_identity: bool = True) -> pd.DataFrame:
    """One row per (group, class value, IoU threshold, confidence threshold).

    Columns: the keys, ``detected_as_<category 1>``, ``detected_as_<category 2>``, ``undetected``
    (shares of the annotated instances, each with ``_se``), ``detected_as_interest`` (the share of the
    interest category again) and ``pixel_share_under_interest``. ``groups`` defaults to
    ``recording_groups()``.
    """
    image_ids = check_inputs(gt_dict, prediction_data, check_identity)
    target = target_of(gt_dict)
    categories = sorted(gt_dict["categories"], key=lambda c: int(c["id"]))
    names = [c["name"] for c in categories]
    if len(names) != 2:
        raise ValueError(f"the report needs a binary target, got {names}")
    interest_id = target.category_id(COVERED_VALUES[0])
    interest = names[interest_id - 1]
    groups = recording_groups() if groups is None else groups
    covered_recs = {r for g in groups.values() if g for r in g}
    values = tuple(int(v) for v in values)
    images = prepare_images(gt_dict, prediction_data["results"], image_ids)
    recording = {img.image_id: split_image_id(img.image_id)[1] for img in images}

    rows = []
    for group_name, group in groups.items():
        subset = [img for img in images if _in_group(recording[img.image_id], group, covered_recs)]
        for conf in conf_thresholds:
            pixel_share = pixel_share_under_category(subset, conf, interest_id, values)
            for iou in iou_thresholds:
                matches = match_images(subset, iou, conf)
                counts = count_instances(subset, matches, categories, iou, conf)
                total = counts[counts["size"] == "total"].set_index("training_class")
                for value in values:
                    name = BY_VALUE[value].name
                    n1, n2, nu = (int(total.loc[r, name]) for r in (names[0], names[1], UNDETECTED))
                    n = n1 + n2 + nu
                    row = {"group": group_name, "class_value": value, "class_name": name, "iou_threshold": float(iou),
                           "confidence_threshold": float(conf), "interest_category": interest, "instances": n,
                           "n_cat1": n1, "n_cat2": n2, "n_undetected": nu}
                    for key, count in ((f"detected_as_{names[0]}", n1), (f"detected_as_{names[1]}", n2),
                                       ("undetected", nu)):
                        row[key], row[f"{key}_se"] = _share(count, n)
                    row["detected_as_interest"] = row[f"detected_as_{interest}"]
                    row["pixel_share_under_interest"] = pixel_share[value]
                    rows.append(row)
    return pd.DataFrame(rows)


def covered_detection_rates(gt_dict: dict, results: list[dict], image_ids=None, iou_thresholds=DEFAULT_IOUS,
                            conf_threshold: float = 0.5) -> dict:
    """Pooled detection of the covered-metal instances (110, 120, 130) over all images, e.g. during
    validation: ``covered_instances``, and per IoU threshold ``covered_detected_iou<t>`` (detected
    and given the interest category), ``covered_found_iou<t>`` (detected, any category) and
    ``covered_undetected_iou<t>`` as shares; NaN without covered-metal instances."""
    prediction_data = {"results": results, "image_ids": image_ids}
    if image_ids is None:
        prediction_data.pop("image_ids")
    table = occluded_case_table(gt_dict, prediction_data, iou_thresholds, (conf_threshold,), COVERED_VALUES,
                                groups={GROUP_ALL: None}, check_identity=False)
    cat1 = _cat1(table)
    out: dict = {}
    for iou in iou_thresholds:
        rows = table[table["iou_threshold"] == float(iou)]
        interest_col = "n_cat1" if str(rows["interest_category"].iloc[0]) == cat1 else "n_cat2"
        n = int(rows["instances"].sum())
        n_interest = int(rows[interest_col].sum())
        found = int(rows["n_cat1"].sum() + rows["n_cat2"].sum())
        undetected = int(rows["n_undetected"].sum())
        tag = f"{int(round(iou * 100)):02d}"
        out["covered_instances"] = n
        out[f"covered_detected_iou{tag}"] = n_interest / n if n else math.nan
        out[f"covered_found_iou{tag}"] = found / n if n else math.nan
        out[f"covered_undetected_iou{tag}"] = undetected / n if n else math.nan
    return out


def _cat1(table: pd.DataFrame) -> str:
    """Name of category 1 from the ``detected_as_<name>`` columns of a table."""
    names = [c[len("detected_as_"):] for c in table.columns
             if c.startswith("detected_as_") and not c.endswith("_se") and c != "detected_as_interest"]
    if len(names) != 2:
        raise ValueError(f"cannot read the category names from the columns {list(table.columns)}")
    return names[0]


def occluded_summary(table: pd.DataFrame, iou: float = 0.5, conf: float = 0.5) -> dict:
    """The numbers the late fusion is tuned and judged by, at one IoU and confidence threshold:

    * ``covered_metal_detected``: share of the 110/120/130 instances (group ``all``) detected and
      given the interest category; ``covered_metal_found_any_class``, ``covered_metal_undetected``;
    * the same three per class value (``covered_<value>_detected`` ...);
    * ``cover_wood_correct``: share of the cover instances (10 in recording 48, 20 in 49) detected and
      given the category that is not the interest category (Wood in the main target);
    * ``pixel_share_covered_metal``: pooled pixel share of 110/120/130 under interest predictions;
    * ``covered_instances``, ``iou_threshold``, ``confidence_threshold``.
    """
    sel = table[(table["iou_threshold"] == float(iou)) & (table["confidence_threshold"] == float(conf))]
    if sel.empty:
        raise ValueError(f"the table holds no rows at IoU {iou} and confidence {conf}")
    cat1 = _cat1(table)
    interest = str(sel["interest_category"].iloc[0])
    interest_col = "n_cat1" if interest == cat1 else "n_cat2"
    other_col = "n_cat2" if interest == cat1 else "n_cat1"

    def pooled(rows: pd.DataFrame) -> tuple[int, int, int, int]:
        n = int(rows["instances"].sum())
        return n, int(rows[interest_col].sum()), int(rows["n_cat1"].sum() + rows["n_cat2"].sum()), int(rows["n_undetected"].sum())

    def ratio(a: int, n: int) -> float:
        return a / n if n else math.nan

    all_rows = sel[sel["group"] == GROUP_ALL]
    covered = all_rows[all_rows["class_value"].isin(COVERED_VALUES)]
    n, n_interest, found, undetected = pooled(covered)
    out = {"iou_threshold": float(iou), "confidence_threshold": float(conf), "interest_category": interest,
           "covered_instances": n, "covered_metal_detected": ratio(n_interest, n),
           "covered_metal_found_any_class": ratio(found, n), "covered_metal_undetected": ratio(undetected, n)}
    for value in COVERED_VALUES:
        rows = covered[covered["class_value"] == value]
        n_v, i_v, f_v, u_v = pooled(rows)
        out[f"covered_{value}_instances"] = n_v
        out[f"covered_{value}_detected"] = ratio(i_v, n_v)
        out[f"covered_{value}_found_any_class"] = ratio(f_v, n_v)
        out[f"covered_{value}_undetected"] = ratio(u_v, n_v)
    cover_n = cover_correct = 0
    for rec, (cover_value, _) in COVER_OF.items():
        rows = sel[(sel["group"] == f"rec{rec}") & (sel["class_value"] == cover_value)]
        if rows.empty or cover_value not in (10, 20):
            continue
        cover_n += int(rows["instances"].sum())
        cover_correct += int(rows[other_col].sum())
    out["cover_wood_instances"] = cover_n
    out["cover_wood_correct"] = ratio(cover_correct, cover_n)
    pixels = covered["pixel_share_under_interest"]
    weights = covered["instances"]
    out["pixel_share_covered_metal"] = (float((pixels.fillna(0) * weights).sum() / weights.sum())
                                        if int(weights.sum()) else math.nan)
    return out


def write_occluded_report(gt_dict: dict, prediction_data: dict, out_dir: str | Path, iou_thresholds=DEFAULT_IOUS,
                          conf_thresholds=DEFAULT_CONFS, check_identity: bool = True) -> tuple[pd.DataFrame, dict]:
    """Write ``occluded_cases.csv`` and ``occluded_summary.json`` (one summary per confidence
    threshold at IoU ``iou_thresholds[0]``, keyed ``conf<value>``) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = occluded_case_table(gt_dict, prediction_data, iou_thresholds, conf_thresholds, check_identity=check_identity)
    table.to_csv(out_dir / TABLE_FILE, index=False)
    summaries = {f"conf{conf:g}": occluded_summary(table, iou_thresholds[0], conf) for conf in conf_thresholds}
    write_json(out_dir / SUMMARY_FILE, summaries)
    return table, summaries


def read_occluded_summary(metrics_dir: str | Path, conf: float = 0.5) -> dict:
    """The summary of one confidence threshold from ``occluded_summary.json``."""
    path = Path(metrics_dir) / SUMMARY_FILE
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = f"conf{conf:g}"
    if key not in data:
        raise KeyError(f"{path}: no summary at confidence {conf}")
    return data[key]


def ground_truth_for_predictions(prediction_data: dict, data_root: str | Path, ef_root: str | Path | None = None,
                                 progress: bool = False) -> dict:
    """The COCO ground truth a prediction file is evaluated against: the RGB ground truth with
    late-fusion image ids for fused results (sensor ``LF``), else the ground truth of the file's
    sensor and target, at the file's target size, covered recordings included."""
    sensor, target = prediction_data["sensor"], prediction_data["target"]
    size = tuple(int(v) for v in prediction_data["target_size"])
    if sensor == "LF":
        gt, _ = build_coco_gt(data_root, "RGB", "main", prediction_data["split"], target_size=size,
                              include_covered=True, image_sensor_index=LF_INDEX, progress=progress)
        return gt
    gt, _ = build_coco_gt(data_root_for(sensor, data_root, ef_root), sensor, target, prediction_data["split"],
                          target_size=size, include_covered=True, progress=progress)
    return gt
