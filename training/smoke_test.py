"""Quick check of the training setup; run this before main_train.py.

Checks the environment, the dataset folder of the configuration in config_train.py, builds the
model (the first run downloads the COCO weights, about 180 MB) and runs one training step and one
validation step on a single batch. Nothing is written to the results folder.
"""
import sys

import torch
import torchvision

import paths
from config_train import CONFIG

from woodvit_rcnn import __version__
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.train_loop import smoke_step

DATA_ROOT = paths.EF_ROOT if CONFIG.sensor == "EF" else paths.DATA_ROOT
DEVICE = None           # None: the GPU if available, else the CPU


def main():
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} "
          f"| woodvit_rcnn {__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (torch built for CUDA {torch.version.cuda})")
    else:
        print("No GPU found: training would run on the CPU (very slow).")

    print(f"\nPreset {CONFIG.preset}: sensor {CONFIG.sensor}, target {CONFIG.target}, "
          f"covered in training {CONFIG.covered_in_training}, batch {CONFIG.batch_size}, "
          f"{CONFIG.num_epochs} epochs, probes {CONFIG.probes or 'all'}")
    print(f"Dataset: {DATA_ROOT}")
    try:
        print(f"Inventory: {find_inventory(DATA_ROOT)}")
    except FileNotFoundError as e:
        sys.exit(f"FAIL: {e}\n  -> set DATA_ROOT (or EF_ROOT) in paths.py")

    info = smoke_step(CONFIG, DATA_ROOT, device=DEVICE)
    print(f"\nModel built in {info['model_seconds']:.1f} s: {info['parameters'] / 1e6:.1f} M parameters, "
          f"{info['trainable_parameters'] / 1e6:.1f} M trainable, on {info['device']}")
    print(f"Images: {info['train_images']} training, {info['val_images']} validation; "
          f"{info['load_seconds_per_image']:.2f} s per image to load; shape {info['image_shape']}")
    losses = info["train_losses"]
    print(f"Training step: loss {losses['loss']:.4f} in {info['train_step_seconds']:.2f} s "
          f"(instances per image {info['instances']})")
    print("  " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items() if k != "loss"))
    if "peak_memory_gb" in info:
        print(f"  peak GPU memory {info['peak_memory_gb']:.2f} GB")
    print(f"Validation step: loss {info['val_loss']:.4f}, {info['val_detections']} detections")
    print("\nSMOKE TEST PASSED - start the training with training/main_train.py")


if __name__ == "__main__":
    main()
