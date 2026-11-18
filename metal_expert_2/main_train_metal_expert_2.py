"""Train Metal_Expert_2: one run per learning rate in config_metal_expert_2.py, one after the other.

Run compute_channel_stats.py and smoke_test_metal_expert_2.py first. Every run is a normal run folder
<RESULTS_ROOT>/Trained_Models/THz/THz_N2_<date>_METAL_ME2_256_withCovered_lr<lr>_pat25/ with
config.json (the Metal_Expert_2 settings in "extra"), checkpoint.pth, training_metadata.json,
train_log.csv, train.log and the TensorBoard logs. A failed run does not stop the following ones.
"""
import traceback

import paths
from config_metal_expert_2 import CHANNEL_STATS_FILE, LEARNING_RATES, OVERRIDES, SETTINGS

from woodvit_rcnn.metal_expert_2 import make_me2_config, new_model_folder_me2, read_channel_stats, train_me2
from woodvit_rcnn.run_io import run_paths

RESULTS_ROOT = paths.RESULTS_ROOT
DATA_ROOT = paths.DATA_ROOT
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    # the statistics must come from the dataset build in DATA_ROOT (same inventory hash)
    settings = SETTINGS.with_stats(read_channel_stats(RESULTS_ROOT / CHANNEL_STATS_FILE, DATA_ROOT))
    summaries = []
    for learning_rate in LEARNING_RATES:
        config = make_me2_config(settings, learning_rate, **OVERRIDES)
        run = run_paths(RESULTS_ROOT, config.sensor, new_model_folder_me2(config))
        print(f"Training {config.preset} with learning rate {learning_rate:g} on {DATA_ROOT}\n  -> {run.root}", flush=True)
        try:
            summary = train_me2(config, DATA_ROOT, run, device=DEVICE)
        except Exception:
            traceback.print_exc()
            print(f"FAILED: {run.name}; continuing with the next learning rate", flush=True)
            continue
        summaries.append(summary)
        print(f"Done: best {summary['selection_metric']} {summary['best_value']:.4f} in epoch {summary['best_epoch']} "
              f"of {summary['epochs']} ({summary['seconds_per_epoch']:.0f} s per epoch)", flush=True)

    print("\nRuns (validation split):")
    for s in summaries:
        print(f"  {s['model_folder']}: {s['selection_metric']} {s['best_value']:.4f} in epoch {s['best_epoch']}")
    print("Next: set MODEL_FOLDER to the run with the best validation value in main_inference_metal_expert_2.py")


if __name__ == "__main__":
    main()
