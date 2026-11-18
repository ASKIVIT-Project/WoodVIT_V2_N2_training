"""Predictions of a trained Metal_Expert_2 model on full splits (covered recordings included).

Writes <run>/inference/pred_<split>.json (all detections, masks at the target size) and
<run>/inference/gt_<split>.json (COCO ground truth of the same images), in the format of the baseline
inference: the input of metrics/main_metrics.py, of analyse_covered_metal.py and of the late fusion.
The baseline inference cannot load these checkpoints (31 input channels, own normalisation).
"""
import paths

from woodvit_rcnn.metal_expert_2 import predict_split_me2
from woodvit_rcnn.run_io import find_run

MODEL_FOLDER = "THz_N2_YYMMDD_HHMMSS_METAL_ME2_256_withCovered_lr0.0001_pat25"    # folder name of the trained model
SPLITS = ("val", "test")
PROBES = None           # None: the full split; a list of recordings only for quick checks
RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    run = find_run(RESULTS_ROOT, MODEL_FOLDER)
    for split in SPLITS:
        info = predict_split_me2(run, DATA_ROOT, split, device=DEVICE, probes=PROBES)
        print(f"{split}: {info['detections']} detections on {info['images']} images in {info['seconds']:.0f} s "
              f"-> {info['predictions']}")
        if not info["same_dataset_as_training"]:
            print(f"  note: {DATA_ROOT} is not the dataset build the model was trained on")
    print(f"Next: set MODEL_FOLDER = {MODEL_FOLDER!r} in metrics/main_metrics.py and in analyse_covered_metal.py")


if __name__ == "__main__":
    main()
