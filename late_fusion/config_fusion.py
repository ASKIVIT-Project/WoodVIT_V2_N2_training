"""Settings of the late fusion: the fused models and the fusion parameters.

The model folders are run folders under RESULTS_ROOT/Trained_Models/<sensor>/ whose predictions
for SPLIT exist (inference/main_inference.py). The parameter defaults are the values of the paper.

The paper fuses four models: RGB, NIR, Thermo and the THz metal expert, with the EF entry left
empty. Results then go to RESULTS_ROOT/Late_Fusion_noEF/. With an EF run folder entered, five
models are fused and the results go to RESULTS_ROOT/Late_Fusion/.
"""
import paths

RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT         # the RGB ground truth of the fusion is built from here
EF_ROOT = paths.EF_ROOT             # needed to check the source of the EF predictions (five models only)

# role -> model folder. Roles: RGB, NIR, Thermo (target main), THz_metal (THz, target metal) and,
# optionally, EF (target main). An empty EF entry fuses the four other models (paper).
MODEL_FOLDERS = {
    "RGB": "",
    "NIR": "",
    "Thermo": "",
    "THz_metal": "",
    "EF": "",       # empty: fusion without the early-fusion model
}
SPLIT = "val"       # val or test; the parameters below are applied unchanged to both

# Fusion parameters (paper values)
SCORE_THRESHOLD = 0.5           # minimum confidence of a prediction to enter the fusion
IOU_THRESHOLD = 0.6             # predictions of different models with at least this mask IoU form a group
MIN_MODELS_PER_PIXEL = 3        # other models that must agree on a pixel to add it to the RGB mask
SINGLE_SCORE_THRESHOLD = 0.5    # minimum confidence of a prediction that overlaps with no other model
SCORE_FUSION = "max"            # fused score: max (paper) or mean
