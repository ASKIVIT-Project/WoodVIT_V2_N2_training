"""Quick check of the Metal_Expert_2 training setup; run this before main_train_metal_expert_2.py.

Reads the channel statistics, builds the model (the first run downloads the COCO weights, about
180 MB) and runs one training step and one validation step on a single batch of the first learning
rate in config_metal_expert_2.py. Nothing is written to the results folder.
"""
import sys

import torch
import torchvision

import paths
from config_metal_expert_2 import CHANNEL_STATS_FILE, LEARNING_RATES, OVERRIDES, SETTINGS

from woodvit_rcnn import __version__
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.metal_expert_2 import make_me2_config, read_channel_stats, smoke_step_me2

DATA_ROOT = paths.DATA_ROOT
STATS_FILE = paths.RESULTS_ROOT / CHANNEL_STATS_FILE
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} "
          f"| woodvit_rcnn {__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (torch built for CUDA {torch.version.cuda})")
    else:
        print("No GPU found: training would run on the CPU (very slow).")

    try:
        # the statistics must come from the dataset build in DATA_ROOT (same inventory hash)
        settings = SETTINGS.with_stats(read_channel_stats(STATS_FILE, DATA_ROOT))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"FAIL: {e}")
    config = make_me2_config(settings, LEARNING_RATES[0], **OVERRIDES)
    print(f"\nPreset {config.preset}: sensor {config.sensor}, target {config.target}, "
          f"covered in training {config.covered_in_training}, batch {config.batch_size}, "
          f"{config.num_epochs} epochs, probes {config.probes or 'all'}, learning rates {LEARNING_RATES}")
    print(f"Metal_Expert_2: channels {settings.channel_start}..{settings.channel_stop - 1} ({settings.channels}), "
          f"covered-metal repeat {settings.covered_metal_repeat:g}, selection by {settings.selection_metric}")
    print(f"Dataset: {DATA_ROOT}")
    try:
        print(f"Inventory: {find_inventory(DATA_ROOT)}")
    except FileNotFoundError as e:
        sys.exit(f"FAIL: {e}\n  -> set DATA_ROOT in paths.py")

    info = smoke_step_me2(config, DATA_ROOT, device=DEVICE)
    print(f"\nModel built in {info['model_seconds']:.1f} s: {info['parameters'] / 1e6:.1f} M parameters, "
          f"{info['trainable_parameters'] / 1e6:.1f} M trainable, {info['frozen_parameters']} frozen, "
          f"{info['input_channels']} input channels, on {info['device']}")
    print(f"  normalisation: mean {info['image_mean_range'][0]:.4f} .. {info['image_mean_range'][1]:.4f}, "
          f"std {info['image_std_range'][0]:.4f} .. {info['image_std_range'][1]:.4f}")
    print(f"Images: {info['train_images']} training ({info['covered_metal_train_images']} with covered metal), "
          f"{info['val_images']} validation; {info['load_seconds_per_image']:.2f} s per image to load; "
          f"shape {info['image_shape']}")
    losses = info["train_losses"]
    print(f"Training step: loss {losses['loss']:.4f} in {info['train_step_seconds']:.2f} s "
          f"(instances per image {info['instances']})")
    print("  " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items() if k != "loss"))
    if "peak_memory_gb" in info:
        print(f"  peak GPU memory {info['peak_memory_gb']:.2f} GB")
    print(f"Validation step: loss {info['val_loss']:.4f}, {info['val_detections']} detections")
    if info["frozen_parameters"]:
        sys.exit("FAIL: the model has frozen parameters")
    print("\nSMOKE TEST PASSED - start the training with metal_expert_2/main_train_metal_expert_2.py")


if __name__ == "__main__":
    main()
