"""Covered-metal report of a prediction file: detection of the class values 110, 120, 130 (and of
the objects covering them) per recording group and per class value.

PREDICTIONS is either the folder name of a trained model (its inference/pred_<split>.json, the ground
truth of its sensor and target) or the path of a fused prediction file (pred_fused.json of the late
fusion or the late fusion 2; the RGB ground truth with late-fusion image ids). Output: occluded_cases.csv
and occluded_summary.json next to the other metrics (<run>/metrics/<split>/ or <folder>/metrics/).

Run:  python metrics/main_occluded_cases.py
"""
from pathlib import Path

import pandas as pd

import paths
from woodvit_rcnn.occluded_eval import ground_truth_for_predictions, write_occluded_report
from woodvit_rcnn.run_io import find_run, read_predictions

PREDICTIONS = "RGB_N2_YYMMDD_HHMMSS_MAIN_256_lr5e-05_pat25"     # run folder name, or a path to a pred_fused.json
SPLITS = ("val", "test")        # only for a run folder; a fused file holds one split
IOU_THRESHOLDS = (0.5, 0.3)
CONF_THRESHOLDS = (0.5, 0.8)
RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT
EF_ROOT = paths.EF_ROOT
SHOWN = ("group", "class_value", "class_name", "iou_threshold", "instances", "detected_as_interest", "undetected",
         "pixel_share_under_interest")


def report(prediction_file: Path, out_dir: Path) -> None:
    predictions = read_predictions(prediction_file)
    gt = ground_truth_for_predictions(predictions, DATA_ROOT, EF_ROOT, progress=True)
    table, summaries = write_occluded_report(gt, predictions, out_dir, IOU_THRESHOLDS, CONF_THRESHOLDS)
    shown = table[(table["confidence_threshold"] == CONF_THRESHOLDS[0]) & (table["group"] != "other")]
    print(f"\n{prediction_file} -> {out_dir}")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200):
        print(shown[list(SHOWN)].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for key, summary in summaries.items():
        print(f"  {key}: covered metal detected as {summary['interest_category']} "
              f"{summary['covered_metal_detected']:.3f}, found any class {summary['covered_metal_found_any_class']:.3f}, "
              f"undetected {summary['covered_metal_undetected']:.3f} ({summary['covered_instances']} instances); "
              f"cover wood correct {summary['cover_wood_correct']:.3f}")


def main():
    if PREDICTIONS.endswith(".json"):
        file = Path(PREDICTIONS)
        report(file, file.parent / "metrics")
        return
    run = find_run(RESULTS_ROOT, PREDICTIONS)
    for split in SPLITS:
        report(run.predictions(split), run.metrics(split))


if __name__ == "__main__":
    main()
