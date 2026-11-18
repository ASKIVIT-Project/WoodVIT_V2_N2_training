"""Per-channel mean and standard deviation of the THz window channels over all pixels of the
training split (covered recordings included): the input normalisation of Metal_Expert_2.

Writes <RESULTS_ROOT>/<CHANNEL_STATS_FILE> (JSON, see config_metal_expert_2.py). Run once before
smoke_test_metal_expert_2.py and main_train_metal_expert_2.py; the values are copied into the
configuration of every run.
"""
import paths
from config_metal_expert_2 import CHANNEL_STATS_FILE, SETTINGS

from woodvit_rcnn.metal_expert_2 import MetalExpert2Dataset, channel_stats
from woodvit_rcnn.run_io import write_json

DATA_ROOT = paths.DATA_ROOT
OUT_FILE = paths.RESULTS_ROOT / CHANNEL_STATS_FILE
PROBES = None           # None: the full training split


def main():
    dataset = MetalExpert2Dataset(DATA_ROOT, SETTINGS, "train", train=False, include_covered=True, probes=PROBES)
    print(f"Channel statistics of {len(dataset)} THz training images, channels "
          f"{SETTINGS.channel_start}..{SETTINGS.channel_stop - 1} ({SETTINGS.channels})")
    stats = channel_stats(dataset, progress=True)
    write_json(OUT_FILE, stats)
    mean, std = stats["mean"], stats["std"]
    peak = stats["channel_start"] + std.index(max(std))
    print(f"{stats['pixels']} pixels per channel; mean {min(mean):.4f} .. {max(mean):.4f}, "
          f"std {min(std):.4f} .. {max(std):.4f} (largest std at channel {peak})")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
