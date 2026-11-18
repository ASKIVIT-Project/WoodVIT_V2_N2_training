"""Evaluation of a prediction file against its ground truth.

``evaluate_run`` writes into one folder (all CSV files with a decimal point and comma separator):

* ``summary_metrics.csv``: COCO metrics of boxes and masks, AP per category, the pixel-wise IoU
  per class and the mIoU (predictions with a confidence of at least 0.5),
* ``per_img_metrics.csv``: pixel-wise IoU per image,
* ``iou_metric.csv`` / ``conf_metric.csv``: OSA and instance totals over the IoU thresholds (at
  the operating confidence 0.8) and over the confidence thresholds (at the operating IoU 0.5),
  with a plot of C-, M- and U-OSA each,
* ``instance_counts_subclasses.csv`` / ``confidence_instance_counts_subclasses.csv``: instance
  counts per predicted class, annotated category, annotated subclass and size, same two series,
* ``iou_cm_metrics.csv`` / ``conf_cm_metrics.csv``: confusion counts and metrics, same two series,
* ``per_img_own_instance_metrics.csv``: OSA per image at the operating point,
* ``operating_point.csv``: the columns of the paper table at IoU 0.5 and confidence 0.8,
* ``evaluation.json``: provenance (prediction source, dataset identity, thresholds).

Runs on the subclass target stop after the COCO part (the instance metrics need a binary target).
"""
from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import pandas as pd

from woodvit_rcnn import __version__
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import METRIC_NAMES, coco_evaluate, to_coco
from woodvit_rcnn.confusion import confusion_table
from woodvit_rcnn.instance_counts import count_instances
from woodvit_rcnn.matching import ImageData, match_images, prepare_images
from woodvit_rcnn.miou import mean_iou
from woodvit_rcnn.osa import SHARES, osa_per_image, osa_series, osa_summary
from woodvit_rcnn.plot_utils import plot_osa_series
from woodvit_rcnn.run_io import (TRAINED_MODELS, RunPaths, check_dataset_identity, find_run, read_predictions,
                                 read_run_config, write_json)

THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 21))
OPERATING_IOU = 0.5
OPERATING_CONF = 0.8
MIOU_SCORE_THRESHOLD = 0.5
LATE_FUSION = "Late_Fusion"         # <results>/Late_Fusion/<split>/<parameter folder>/
LATE_FUSION_NO_EF = "Late_Fusion_noEF"      # the same for the late fusion without the early-fusion model
LATE_FUSION_LEVELS: dict[str, str] = {LATE_FUSION: "LF", LATE_FUSION_NO_EF: "LF_noEF"}  # folder -> column Fusion

FILES: dict[str, str] = {
    "summary": "summary_metrics.csv",
    "per_image_miou": "per_img_metrics.csv",
    "osa_iou": "iou_metric.csv",
    "osa_conf": "conf_metric.csv",
    "counts_iou": "instance_counts_subclasses.csv",
    "counts_conf": "confidence_instance_counts_subclasses.csv",
    "cm_iou": "iou_cm_metrics.csv",
    "cm_conf": "conf_cm_metrics.csv",
    "per_image_osa": "per_img_own_instance_metrics.csv",
    "operating_point": "operating_point.csv",
    "plot_iou": "iou_area_pct_over_thresholds.png",
    "plot_conf": "conf_area_pct_over_thresholds.png",
    "info": "evaluation.json",
}

# Paper names of the confusion metrics (columns of operating_point.csv) and their column in *_cm_metrics.csv.
CM_PAPER_NAMES: dict[str, str] = {
    "Coverage": "coverage", "Accuracy": "accuracy_classified", "Precision": "precision_cat1",
    "Recall": "recall_cat1", "F1": "f1_strict", "nMCC": "norm_mcc_strict",
    "nMCC_classified": "norm_mcc_classified", "Specificity": "specificity",
}
COCO_TABLE: tuple[str, ...] = ("mIoU", "bbox_mAP", "segm_mAP", "segm_mAP50", "segm_mAP75",
                               "segm_mAP_small", "segm_mAP_medium", "segm_mAP_large")
OPERATING_COLUMNS: tuple[str, ...] = ("iou_threshold", "conf_threshold", *SHARES, *CM_PAPER_NAMES, *COCO_TABLE)


# ----------------------------------------------------------------------------- checks
def _plain(value):
    return [int(v) for v in value] if isinstance(value, (list, tuple)) else value


def check_inputs(gt_dict: dict, prediction_data: dict, check_identity: bool = True) -> list[int]:
    """Consistency of predictions and ground truth; returns the evaluated image ids.

    Checks the dataset identity, split, target and raster, that all predicted images belong to
    the ground truth, and that all predictions lie on predicted images and use categories of the
    ground truth.
    """
    info = gt_dict.get("info") or {}
    if check_identity:
        check_dataset_identity(info.get("dataset"), prediction_data.get("dataset"), "predictions")
    for key in ("split", "target", "target_size"):
        a, b = info.get(key), prediction_data.get(key)
        if a is not None and b is not None and _plain(a) != _plain(b):
            raise ValueError(f"{key} of predictions ({b!r}) and ground truth ({a!r}) differ")
    gt_images = {int(img["id"]) for img in gt_dict["images"]}
    image_ids = prediction_data.get("image_ids")
    image_ids = sorted(gt_images) if image_ids is None else sorted({int(i) for i in image_ids})
    outside = sorted(set(image_ids) - gt_images)
    if outside:
        raise ValueError(f"{len(outside)} predicted images are not in the ground truth, e.g. {outside[:3]}")
    predicted = set(image_ids)
    category_ids = {int(c["id"]) for c in gt_dict["categories"]}
    for res in prediction_data["results"]:
        if int(res["image_id"]) not in predicted:
            raise ValueError(f"prediction on image {res['image_id']}, which is not listed in image_ids")
        if int(res["category_id"]) not in category_ids:
            raise ValueError(f"prediction category {res['category_id']} is not a category of the ground truth")
    return image_ids


# ----------------------------------------------------------------------------- evaluation
def _series(images: list[ImageData], categories: list[dict], pairs) -> tuple[list[dict], pd.DataFrame]:
    summaries, counts = [], []
    for iou, conf in pairs:
        matches = match_images(images, iou, conf)
        summaries.append(osa_summary(images, matches, iou, conf))
        counts.append(count_instances(images, matches, categories, iou, conf))
    return summaries, pd.concat(counts, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def summary_rows(coco_metrics: dict, per_category: dict, iou_summary: dict) -> list[dict]:
    rows = []
    for name in METRIC_NAMES:
        rows += [{"metric": f"bbox_{name}", "value": coco_metrics[f"bbox_{name}"]},
                 {"metric": f"segm_{name}", "value": coco_metrics[f"segm_{name}"]}]
    for iou_type in ("bbox", "segm"):
        rows += [{"metric": k, "value": v} for k, v in per_category.items() if k.startswith(f"{iou_type}_")]
    rows += [{"metric": k, "value": v} for k, v in iou_summary.items()]
    return rows


def evaluate_run(gt_dict: dict, prediction_data: dict, out_dir: str | Path, *, prefix: str = "",
                 iou_thresholds=THRESHOLDS, conf_thresholds=THRESHOLDS, operating_iou: float = OPERATING_IOU,
                 operating_conf: float = OPERATING_CONF, miou_score_threshold: float = MIOU_SCORE_THRESHOLD,
                 pred_area: str = "mask", plots: bool = True, check_identity: bool = True) -> dict:
    """Evaluate predictions (format of ``run_io.write_predictions``) against a COCO ground truth
    (``coco_gt.build_coco_gt``) and write the metric files to ``out_dir``.

    Returns ``{"summary": {...}, "operating_point": {...} or None, "tables": {...}}``.
    """
    image_ids = check_inputs(gt_dict, prediction_data, check_identity)
    results = prediction_data["results"]
    categories = sorted(gt_dict["categories"], key=lambda c: int(c["id"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def path(key: str) -> Path:
        return out_dir / (f"{prefix}_{FILES[key]}" if prefix else FILES[key])

    coco_metrics, per_category = coco_evaluate(to_coco(gt_dict), results, img_ids=image_ids)
    per_image_iou, iou_summary = mean_iou(gt_dict, results, image_ids, miou_score_threshold)
    summary_df = pd.DataFrame(summary_rows(coco_metrics, per_category, iou_summary), columns=["metric", "value"])
    _write_csv(summary_df, path("summary"))
    _write_csv(per_image_iou, path("per_image_miou"))
    summary = dict(zip(summary_df["metric"], summary_df["value"]))
    out = {"summary": summary, "operating_point": None, "tables": {"summary": summary_df}}

    binary = len(categories) == 2
    if binary:
        images = prepare_images(gt_dict, results, image_ids, pred_area)
        osa_iou, counts_iou = _series(images, categories, [(t, operating_conf) for t in iou_thresholds])
        osa_conf, counts_conf = _series(images, categories, [(operating_iou, t) for t in conf_thresholds])
        tables = {
            "osa_iou": osa_series(osa_iou, "iou_", "iou_threshold"),
            "osa_conf": osa_series(osa_conf, "conf_", "conf_threshold"),
            "counts_iou": counts_iou, "counts_conf": counts_conf,
            "cm_iou": confusion_table(counts_iou, categories),
            "cm_conf": confusion_table(counts_conf, categories),
        }
        matches = match_images(images, operating_iou, operating_conf)
        tables["per_image_osa"] = osa_per_image(images, matches, operating_iou, operating_conf)
        osa_op = osa_summary(images, matches, operating_iou, operating_conf)
        cm_op = confusion_table(count_instances(images, matches, categories, operating_iou, operating_conf), categories)
        cm_op = cm_op[cm_op["size"] == "total"].iloc[0]
        operating = {"iou_threshold": float(operating_iou), "conf_threshold": float(operating_conf)}
        operating.update({k: osa_op[k] for k in SHARES})
        operating.update({k: float(cm_op[v]) for k, v in CM_PAPER_NAMES.items()})
        operating.update({k: summary[k] for k in COCO_TABLE})
        operating.update({k: v for k, v in summary.items() if k.startswith("segm_AP_")})
        tables["operating_point"] = pd.DataFrame([operating])
        for key, df in tables.items():
            _write_csv(df, path(key))
        if plots:
            plot_osa_series(tables["osa_iou"], "iou_", f"IoU threshold (confidence {operating_conf:.2f})",
                            path("plot_iou"))
            plot_osa_series(tables["osa_conf"], "conf_", f"Confidence threshold (IoU {operating_iou:.2f})",
                            path("plot_conf"))
        out["operating_point"] = operating
        out["tables"].update(tables)

    write_json(path("info"), {
        "code_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "prediction_source": prediction_data.get("source"),
        "sensor": prediction_data.get("sensor"), "target": prediction_data.get("target"),
        "split": prediction_data.get("split"), "dataset": prediction_data.get("dataset"),
        "images": len(image_ids), "predictions": len(results), "annotations": len(gt_dict["annotations"]),
        "instance_metrics": binary, "iou_thresholds": list(iou_thresholds), "conf_thresholds": list(conf_thresholds),
        "operating_iou": operating_iou, "operating_conf": operating_conf,
        "miou_score_threshold": miou_score_threshold, "pred_area": pred_area,
    })
    return out


# ----------------------------------------------------------------------------- runs on disk
def data_root_for(sensor: str, data_root: str | Path, ef_root: str | Path | None = None) -> Path:
    """Dataset folder of a sensor: the early-fusion folder for EF, the patch dataset otherwise."""
    if sensor == "EF":
        if ef_root is None:
            raise ValueError("the EF ground truth needs the early-fusion folder (ef_root)")
        return Path(ef_root)
    return Path(data_root)


def evaluate_model_run(results_root: str | Path, model_folder: str, split: str, data_root: str | Path,
                       ef_root: str | Path | None = None, progress: bool = False, **kwargs) -> dict:
    """Evaluate the saved predictions of a run (``<run>/inference/pred_<split>.json``) into
    ``<run>/metrics/<split>/``. The ground truth is rebuilt from the dataset."""
    run = find_run(results_root, model_folder)
    config, _ = read_run_config(run)
    predictions = read_predictions(run.predictions(split))
    if predictions.get("sensor") != config.sensor or predictions.get("target") != config.target:
        raise ValueError(f"{run.predictions(split)}: sensor/target {predictions.get('sensor')}/"
                         f"{predictions.get('target')} differ from the run ({config.sensor}/{config.target})")
    gt, _ = build_coco_gt(data_root_for(config.sensor, data_root, ef_root), config.sensor, config.target, split,
                          target_size=tuple(predictions["target_size"]), include_covered=True, progress=progress)
    return evaluate_run(gt, predictions, run.metrics(split), **kwargs)


def list_runs(results_root: str | Path) -> list[RunPaths]:
    """All run folders under ``<results>/Trained_Models/<sensor>/``."""
    base = Path(results_root) / TRAINED_MODELS
    return [RunPaths(p) for p in sorted(base.glob("*/*")) if (p / "training" / "config.json").is_file()]


# ----------------------------------------------------------------------------- reading metric files
def read_summary(metrics_dir: str | Path, prefix: str = "") -> dict:
    name = f"{prefix}_{FILES['summary']}" if prefix else FILES["summary"]
    df = pd.read_csv(Path(metrics_dir) / name)
    return dict(zip(df["metric"], df["value"].astype(float)))


def metrics_at(metrics_dir: str | Path, iou: float = OPERATING_IOU, conf: float = OPERATING_CONF,
               prefix: str = "") -> dict:
    """Paper-table values of one evaluation folder at an IoU and a confidence threshold.

    The instance metrics are read from the confidence series if ``iou`` is its IoU threshold, else
    from the IoU series if ``conf`` is its confidence threshold. Folders without instance metrics
    (subclass runs) give NaN for them.
    """
    metrics_dir = Path(metrics_dir)

    def file(key: str) -> Path:
        return metrics_dir / (f"{prefix}_{FILES[key]}" if prefix else FILES[key])

    summary = read_summary(metrics_dir, prefix)
    row = {"iou_threshold": float(iou), "conf_threshold": float(conf)}
    row.update({k: math.nan for k in (*SHARES, *CM_PAPER_NAMES)})
    row.update({k: summary.get(k, math.nan) for k in COCO_TABLE})
    row.update({k: v for k, v in summary.items() if k.startswith("segm_AP_")})
    if not file("osa_conf").is_file():
        return row
    chosen = None
    for osa_key, cm_key, column in (("osa_conf", "cm_conf", f"conf_{conf:.2f}"), ("osa_iou", "cm_iou", f"iou_{iou:.2f}")):
        osa = pd.read_csv(file(osa_key))
        if column not in osa.columns:
            continue
        values = dict(zip(osa["metric"], osa[column].astype(float)))
        if math.isclose(values["iou_threshold"], iou) and math.isclose(values["conf_threshold"], conf):
            chosen = (values, pd.read_csv(file(cm_key)))
            break
    if chosen is None:
        raise ValueError(f"{metrics_dir}: no series contains IoU {iou} with confidence {conf}")
    values, cm = chosen
    row.update({k: values[k] for k in SHARES})
    sel = cm[(cm["size"] == "total") & cm["iou_threshold"].sub(iou).abs().lt(1e-9)
             & cm["confidence_threshold"].sub(conf).abs().lt(1e-9)]
    row.update({k: float(sel.iloc[0][v]) for k, v in CM_PAPER_NAMES.items()})
    return row


SENSOR_ORDER: tuple[str, ...] = ("RGB", "NIR", "Thermo", "THz", "EF", "LF")


def combined_table(results_root: str | Path, split: str = "test", iou: float = OPERATING_IOU,
                   conf: float = OPERATING_CONF) -> pd.DataFrame:
    """One row per evaluated run and late-fusion folder of a split (the paper table). The column
    ``Fusion`` is ``LF`` for the late fusion of five models and ``LF_noEF`` for the late fusion
    without the early-fusion model."""
    rows = []
    for run in list_runs(results_root):
        metrics_dir = run.metrics(split)
        if not (metrics_dir / FILES["summary"]).is_file():
            continue
        config, _ = read_run_config(run)
        level = "EF" if config.sensor == "EF" else "None"
        rows.append({"Model": run.name, "Sensor": config.sensor, "Fusion": level, "Target": config.target,
                     **metrics_at(metrics_dir, iou, conf)})
    for root_name, level in LATE_FUSION_LEVELS.items():
        lf_root = Path(results_root) / root_name / split
        for folder in sorted(p for p in lf_root.glob("*") if (p / "metrics" / FILES["summary"]).is_file()):
            rows.append({"Model": folder.name, "Sensor": "LF", "Fusion": level, "Target": "main",
                         **metrics_at(folder / "metrics", iou, conf)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_order"] = df["Sensor"].map({s: i for i, s in enumerate(SENSOR_ORDER)})
    return df.sort_values(["_order", "Target", "Model"], kind="stable").drop(columns="_order").reset_index(drop=True)
