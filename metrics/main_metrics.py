"""Evaluate the predictions of one trained model.

Prerequisite: inference/main_inference.py has written <run>/inference/pred_<split>.json.
The ground truth is rebuilt from the dataset (EF runs: from the early-fusion folder). Output:
<run>/metrics/<split>/ (see woodvit_rcnn/evaluate.py for the list of files).

Run:  python metrics/main_metrics.py
"""
import time

import paths
from woodvit_rcnn.evaluate import MIOU_SCORE_THRESHOLD, OPERATING_CONF, OPERATING_IOU, THRESHOLDS, evaluate_model_run

RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT
EF_ROOT = paths.EF_ROOT

MODEL_FOLDER = "RGB_N2_YYMMDD_HHMMSS_MAIN_256_lr5e-05_pat25"     # folder under RESULTS_ROOT/Trained_Models/<sensor>/
SPLITS = ("val", "test")

OPERATING_IOU_THRESHOLD = OPERATING_IOU         # IoU of the confidence series and of the operating point
OPERATING_CONF_THRESHOLD = OPERATING_CONF       # confidence of the IoU series and of the operating point
SERIES_THRESHOLDS = THRESHOLDS                  # 0.05, 0.10, ..., 1.00
MIOU_SCORE = MIOU_SCORE_THRESHOLD               # minimum confidence of the predictions in the mIoU
PLOTS = True


if __name__ == "__main__":
    for split in SPLITS:
        start = time.time()
        out = evaluate_model_run(RESULTS_ROOT, MODEL_FOLDER, split, DATA_ROOT, EF_ROOT, progress=True,
                                 iou_thresholds=SERIES_THRESHOLDS, conf_thresholds=SERIES_THRESHOLDS,
                                 operating_iou=OPERATING_IOU_THRESHOLD, operating_conf=OPERATING_CONF_THRESHOLD,
                                 miou_score_threshold=MIOU_SCORE, plots=PLOTS)
        summary = out["summary"]
        print(f"\n{MODEL_FOLDER} [{split}]  segm mAP {summary['segm_mAP']:.3f}  bbox mAP {summary['bbox_mAP']:.3f}"
              f"  mIoU {summary['mIoU']:.3f}")
        if out["operating_point"] is not None:
            op = out["operating_point"]
            print("  " + "  ".join(f"{k} {op[k]:.3f}" for k in ("C_OSA", "M_OSA", "U_OSA", "Coverage", "Accuracy",
                                                              "Precision", "Recall", "F1", "nMCC")))
        print(f"  took {time.time() - start:.0f} s")
