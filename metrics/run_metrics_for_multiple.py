"""Evaluate several trained models in one go (same output as metrics/main_metrics.py per model).

MODEL_FOLDERS = None evaluates every run under RESULTS_ROOT/Trained_Models/ that has predictions
for the split.

Run:  python metrics/run_metrics_for_multiple.py
"""
import time

import paths
from woodvit_rcnn.evaluate import evaluate_model_run, list_runs

RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT
EF_ROOT = paths.EF_ROOT

MODEL_FOLDERS = None        # e.g. ["RGB_N2_...", "NIR_N2_..."]; None = all runs
SPLITS = ("val", "test")
SKIP_EVALUATED = False      # skip runs that already have <run>/metrics/<split>/summary_metrics.csv
PLOTS = True


if __name__ == "__main__":
    runs = list_runs(RESULTS_ROOT)
    if MODEL_FOLDERS is not None:
        runs = [r for r in runs if r.name in set(MODEL_FOLDERS)]
        missing = set(MODEL_FOLDERS) - {r.name for r in runs}
        if missing:
            raise FileNotFoundError(f"runs not found under {RESULTS_ROOT}: {sorted(missing)}")
    for split in SPLITS:
        for run in runs:
            if not run.predictions(split).is_file():
                print(f"skip {run.name} [{split}]: no predictions")
                continue
            if SKIP_EVALUATED and (run.metrics(split) / "summary_metrics.csv").is_file():
                print(f"skip {run.name} [{split}]: already evaluated")
                continue
            start = time.time()
            out = evaluate_model_run(RESULTS_ROOT, run.name, split, DATA_ROOT, EF_ROOT, plots=PLOTS)
            print(f"{run.name} [{split}]: segm mAP {out['summary']['segm_mAP']:.3f} ({time.time() - start:.0f} s)")
