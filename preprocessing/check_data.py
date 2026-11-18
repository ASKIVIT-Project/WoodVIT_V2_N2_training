"""Check a dataset folder before training.

Compares the files with the inventory, checks shape, channels and value range of the sensor
data, and counts the ground-truth instances per sensor, split and class. Results go to
OUT_DIR (CSV files and summary.txt).
"""
import paths
from woodvit_rcnn.data_check import check_dataset, write_report

DATASET_ROOT = paths.DATA_ROOT      # or paths.EF_ROOT for the early-fusion input
SENSORS = None                      # None = all sensors of the inventory, e.g. ["RGB"]
PROBES = None                       # None = all recordings, e.g. [47, 48]
TARGET_SIZE = (256, 256)            # resolution used for the small/lost instance counts
ARRAY_SAMPLES = 3                   # value range is checked on this many files per sensor and split
OUT_DIR = paths.RESULTS_ROOT / "data_check" / DATASET_ROOT.name

if __name__ == "__main__":
    report = check_dataset(DATASET_ROOT, sensors=SENSORS, probes=PROBES, target_size=TARGET_SIZE,
                           array_samples=ARRAY_SAMPLES)
    summary = write_report(report, OUT_DIR)
    print(summary.read_text(encoding="utf-8"))
    print(f"Report written to {OUT_DIR}")
