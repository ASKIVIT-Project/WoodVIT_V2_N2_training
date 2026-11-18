"""Train one baseline model with the configuration in config_train.py.

Run smoke_test.py first. The run folder is
<RESULTS_ROOT>/Trained_Models/<sensor>/<MODEL_FOLDER>/training/ with config.json, checkpoint.pth,
training_metadata.json, train_log.csv, train.log and the TensorBoard logs.
The early-fusion model reads its data from paths.EF_ROOT (see preprocessing/create_ef_data.py),
all other models from paths.DATA_ROOT.
"""
import paths
from config_train import CONFIG, MODEL_FOLDER

from woodvit_rcnn.run_io import new_model_folder, run_paths
from woodvit_rcnn.train_loop import train

RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.EF_ROOT if CONFIG.sensor == "EF" else paths.DATA_ROOT
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    run = run_paths(RESULTS_ROOT, CONFIG.sensor, MODEL_FOLDER or new_model_folder(CONFIG))
    print(f"Training {CONFIG.preset} on {DATA_ROOT}\n  -> {run.root}")
    summary = train(CONFIG, DATA_ROOT, run, device=DEVICE)
    print(f"Done: best segm mAP {summary['best_segm_mAP']:.4f} in epoch {summary['best_epoch']} "
          f"of {summary['epochs']} ({summary['seconds_per_epoch']:.0f} s per epoch)")
    print(f"Next: set MODEL_FOLDER = {run.name!r} in inference/main_inference.py")


if __name__ == "__main__":
    main()
