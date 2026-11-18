"""Late fusion of the models in config_fusion.py with the parameters set there.

Output: RESULTS_ROOT/Late_Fusion_noEF/<split>/<parameter folder>/ for the four models of the paper
(EF entry empty), RESULTS_ROOT/Late_Fusion/<split>/<parameter folder>/ with an EF model, each with
pred_fused.json, config_fusion.json and metrics/ (same files as metrics/main_metrics.py writes for
a model).

Run:  python late_fusion/late_fusion.py
"""
import time

from config_fusion import (DATA_ROOT, EF_ROOT, IOU_THRESHOLD, MIN_MODELS_PER_PIXEL, MODEL_FOLDERS, RESULTS_ROOT,
                           SCORE_FUSION, SCORE_THRESHOLD, SINGLE_SCORE_THRESHOLD, SPLIT)
from woodvit_rcnn.fusion import FusionParams, load_fusion_inputs, run_late_fusion

SPLITS = (SPLIT,)       # e.g. ("val", "test") to apply the chosen parameters to both
PLOTS = True


if __name__ == "__main__":
    params = FusionParams(score_threshold=SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                          min_models_per_pixel=MIN_MODELS_PER_PIXEL, single_score_threshold=SINGLE_SCORE_THRESHOLD,
                          score_fusion=SCORE_FUSION)
    for split in SPLITS:
        start = time.time()
        inputs = load_fusion_inputs(RESULTS_ROOT, MODEL_FOLDERS, split, DATA_ROOT, EF_ROOT, progress=True)
        folder = run_late_fusion(inputs, params, RESULTS_ROOT, plots=PLOTS, progress=True)
        print(f"Late fusion of {', '.join(inputs.roles)} [{split}] saved to {folder} "
              f"({time.time() - start:.0f} s)")
