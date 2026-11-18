"""Predictions of a trained model on full splits (covered recordings included).

Writes <run>/inference/pred_<split>.json (all detections, masks at the target size) and
<run>/inference/gt_<split>.json (COCO ground truth of the same images), the input of
metrics/main_metrics.py and of the late fusion.
"""
import paths

from woodvit_rcnn.predict import predict_split
from woodvit_rcnn.run_io import find_run, read_run_config

MODEL_FOLDER = "RGB_N2_YYMMDD_HHMMSS_MAIN_256_lr5e-05_pat25"     # folder name of the trained model
SPLITS = ("val", "test")
PROBES = None           # None: the full split (as in the paper); a list of recordings only for quick checks
RESULTS_ROOT = paths.RESULTS_ROOT
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    run = find_run(RESULTS_ROOT, MODEL_FOLDER)
    config, _ = read_run_config(run)
    data_root = paths.EF_ROOT if config.sensor == "EF" else paths.DATA_ROOT
    for split in SPLITS:
        info = predict_split(run, data_root, split, device=DEVICE, probes=PROBES)
        print(f"{split}: {info['detections']} detections on {info['images']} images in {info['seconds']:.0f} s "
              f"-> {info['predictions']}")
        if not info["same_dataset_as_training"]:
            print(f"  note: {data_root} is not the dataset build the model was trained on")


if __name__ == "__main__":
    main()
