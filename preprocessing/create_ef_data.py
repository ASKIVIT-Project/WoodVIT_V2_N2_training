"""Build the early-fusion (EF) input from the four single-sensor datasets.

Every patch with all four sensors is resized to TARGET_SIZE (bilinear) and stacked to 428
channels in the order R, G, B, NIR (224), Thermo (1), THz (200), as float32. The ground truth is
the RGB-raster ground truth resized with nearest neighbour. Writes EF/patches_<split>/,
patch_inventory_EF.csv and ef_source.json to EF_ROOT; train on it with DATA_ROOT = EF_ROOT.

Disk space: about 4 * H * W * 428 bytes per patch, i.e. about 112 MB at 256 x 256 and about
290 GB for the whole dataset. The free space is checked before anything is written.
Existing patches are skipped unless OVERWRITE is True; if EF_ROOT holds data of a different
build (other dataset, target size or code version), the script stops unless OVERWRITE is True.
"""
import paths
from woodvit_rcnn.early_fusion import build_early_fusion

DATA_ROOT = paths.DATA_ROOT
EF_ROOT = paths.EF_ROOT
TARGET_SIZE = (256, 256)    # (height, width); must equal the target size used for training
SPLITS = None               # None = all splits, e.g. ["train", "val"]
PROBES = None               # None = all recordings, e.g. [47, 48]
OVERWRITE = False

if __name__ == "__main__":
    report = build_early_fusion(DATA_ROOT, EF_ROOT, target_size=TARGET_SIZE, splits=SPLITS, probes=PROBES,
                                overwrite=OVERWRITE)
    print(f"{report.patches} patches: {report.written} written, {report.skipped} already present")
    if report.incomplete:
        print(f"{len(report.incomplete)} patches skipped because a sensor is missing, e.g. {report.incomplete[:5]}")
    print(f"Inventory: {report.inventory}")
