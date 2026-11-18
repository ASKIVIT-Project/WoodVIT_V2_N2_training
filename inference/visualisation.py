"""Predictions next to the ground truth as PNG images, one per patch.

Needs the files of inference/main_inference.py. The background is the preview PNG of the patch
(or, without preview, an image made from the sensor data). Output by default in
<run>/visualisation/<split>/<image_id>_comparison.png.
"""
import paths

from woodvit_rcnn.run_io import find_run, read_run_config
from woodvit_rcnn.visualize import visualize_split

MODEL_FOLDER = "RGB_N2_YYMMDD_HHMMSS_MAIN_256_lr5e-05_pat25"     # folder name of the trained model
SPLIT = "test"
SCORE_THRESHOLD = 0.5   # predictions below this score are not drawn
MAX_IMAGES = None       # None: all images of the split
OUT_DIR = None          # None: <run>/visualisation/<split>
RESULTS_ROOT = paths.RESULTS_ROOT


def main():
    run = find_run(RESULTS_ROOT, MODEL_FOLDER)
    config, _ = read_run_config(run)
    data_root = paths.EF_ROOT if config.sensor == "EF" else paths.DATA_ROOT
    written = visualize_split(run, data_root, SPLIT, out_dir=OUT_DIR, score_threshold=SCORE_THRESHOLD,
                              max_images=MAX_IMAGES)
    print(f"{len(written)} images written to {written[0].parent if written else OUT_DIR}")


if __name__ == "__main__":
    main()
