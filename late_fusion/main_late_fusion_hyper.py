"""Grid search over the late-fusion parameters (the paper values are part of the grid).

Uses the models of config_fusion.py. Every parameter set is fused and evaluated into its own
folder RESULTS_ROOT/Late_Fusion_noEF/<split>/<parameter folder>/ (Late_Fusion/ with an EF model).
Afterwards,
late_fusion/find_best_lf_params.py lists the best parameter sets.

Run:  python late_fusion/main_late_fusion_hyper.py
"""
import time

from config_fusion import DATA_ROOT, EF_ROOT, MODEL_FOLDERS, RESULTS_ROOT
from woodvit_rcnn.fusion import fusion_folder, load_fusion_inputs, parameter_grid, run_late_fusion

SPLIT = "val"
SCORE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)            # input confidence (paper: 0.5)
IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)              # grouping IoU (paper: 0.6)
MIN_MODELS = (2, 3)                                     # models per added pixel (paper: 3)
SINGLE_SCORE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)     # single predictions, not below the input confidence (paper: 0.5)
SCORE_FUSIONS = ("max",)
SKIP_EXISTING = True        # skip parameter sets whose folder already holds metrics/operating_point.csv
PLOTS = False


if __name__ == "__main__":
    grid = parameter_grid(SCORE_THRESHOLDS, IOU_THRESHOLDS, MIN_MODELS, SINGLE_SCORE_THRESHOLDS, SCORE_FUSIONS)
    inputs = load_fusion_inputs(RESULTS_ROOT, MODEL_FOLDERS, SPLIT, DATA_ROOT, EF_ROOT, progress=True)
    print(f"Late fusion of {', '.join(inputs.roles)} [{SPLIT}], {len(grid)} parameter sets")
    for k, params in enumerate(grid, start=1):
        folder = fusion_folder(RESULTS_ROOT, SPLIT, params, inputs.target_size, inputs.roles)
        if SKIP_EXISTING and (folder / "metrics" / "operating_point.csv").is_file():
            print(f"[{k}/{len(grid)}] skip {folder.name}")
            continue
        start = time.time()
        run_late_fusion(inputs, params, RESULTS_ROOT, plots=PLOTS)
        print(f"[{k}/{len(grid)}] {folder.name} ({time.time() - start:.0f} s)")
