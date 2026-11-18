"""Collect the paper-table metrics of all evaluated runs and late-fusion folders into one table.

Reads <run>/metrics/<SPLIT>/ of every run under RESULTS_ROOT/Trained_Models/ and the metrics/ of
every folder under RESULTS_ROOT/Late_Fusion_noEF/<SPLIT>/ (late fusion of four models, column
Fusion = LF_noEF) and RESULTS_ROOT/Late_Fusion/<SPLIT>/ (five models, LF). The instance metrics are taken at
IOU_THRESHOLD and CONF_THRESHOLD: either IOU_THRESHOLD is the IoU of the confidence series (0.5)
or CONF_THRESHOLD is the confidence of the IoU series (0.8).

Output: RESULTS_ROOT/combined_metrics/metrics_<split>_iou<iou>_conf<conf>.csv (3 decimals).

Run:  python metrics/extract_wanted_metrics.py
"""
from pathlib import Path

import paths
from woodvit_rcnn.evaluate import combined_table

RESULTS_ROOT = paths.RESULTS_ROOT
SPLIT = "test"
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.8
DECIMALS = 3


if __name__ == "__main__":
    table = combined_table(RESULTS_ROOT, SPLIT, IOU_THRESHOLD, CONF_THRESHOLD)
    if table.empty:
        raise SystemExit(f"no evaluated runs for split {SPLIT!r} under {RESULTS_ROOT}")
    out_dir = Path(RESULTS_ROOT) / "combined_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"metrics_{SPLIT}_iou{IOU_THRESHOLD:.2f}_conf{CONF_THRESHOLD:.2f}.csv"
    table.round(DECIMALS).to_csv(out, index=False, float_format=f"%.{DECIMALS}f")
    print(table.round(DECIMALS).to_string(index=False))
    print(f"\nSaved to {out}")
