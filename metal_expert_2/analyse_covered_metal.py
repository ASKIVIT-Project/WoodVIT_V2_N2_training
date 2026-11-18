"""Detection of the covered-metal subclasses by a model with the metal target (Metal_Expert_2).

For open metal (class value 70) and the three covered-metal classes (110 under chipboard, 120 under
solid wood, 130 under plastic), per split and IoU threshold at confidence 0.5:

* the share of the annotated instances detected as Metal, detected but called Non_metal, and not
  detected at all (greedy class-agnostic matching of woodvit_rcnn.matching, as in the paper metrics),
  each with its binomial standard error,
* the share of the annotated pixels under Metal predictions.

Prerequisite: main_inference_metal_expert_2.py has written <run>/inference/pred_<split>.json and
gt_<split>.json. Output: <run>/metrics/<split>/covered_metal_analysis.csv and the table on screen.
"""
import pandas as pd

import paths

from woodvit_rcnn.metal_expert_2 import ANALYSIS_FILE, covered_metal_table
from woodvit_rcnn.run_io import find_run, read_json, read_predictions

MODEL_FOLDER = "THz_N2_YYMMDD_HHMMSS_METAL_ME2_256_withCovered_lr0.0001_pat25"    # folder under RESULTS_ROOT/Trained_Models/THz/
SPLITS = ("val", "test")
IOU_THRESHOLDS = (0.5, 0.3)
CONFIDENCE = 0.5
RESULTS_ROOT = paths.RESULTS_ROOT

SHOWN_COLUMNS = ("class_value", "class_name", "iou_threshold", "instances", "detected_as_metal", "detected_as_metal_se",
                 "detected_as_non_metal", "undetected", "metal_pixel_share")


def main():
    run = find_run(RESULTS_ROOT, MODEL_FOLDER)
    for split in SPLITS:
        predictions = read_predictions(run.predictions(split))
        gt_dict = read_json(run.ground_truth(split))
        table = covered_metal_table(gt_dict, predictions, IOU_THRESHOLDS, CONFIDENCE)
        out = run.metrics(split) / ANALYSIS_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\n{MODEL_FOLDER} [{split}], confidence {CONFIDENCE} -> {out}")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(table[list(SHOWN_COLUMNS)].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
