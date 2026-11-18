"""Cut an N=1 dataset (48 cm patches) into the N=2 quarters (24 cm) used by the baselines.

Only needed if you have the N=1 patches only. Writes <Sensor>/patches_<split>/ and
patch_inventory_N2.csv to N2_ROOT. Needs about as much disk space as the N=1 dataset.
Existing children are skipped unless OVERWRITE is True. The script can be run in parts (some
recordings or sensors at a time): the inventory keeps the rows written by earlier runs.
"""
import paths
from woodvit_rcnn.subdivide import subdivide_dataset

N1_ROOT = paths.N1_ROOT
N2_ROOT = paths.DATA_ROOT
PROBES = None           # None = all recordings, e.g. [47, 48]
SENSORS = None          # None = all sensors; the RGB ground truth is always read for the file names
PREVIEWS = True         # also cut the 8-bit preview PNGs, if present
OVERWRITE = False

if __name__ == "__main__":
    report = subdivide_dataset(N1_ROOT, N2_ROOT, probes=PROBES, sensors=SENSORS, previews=PREVIEWS,
                               overwrite=OVERWRITE)
    print(f"{report.parents} parent patches, {report.children_written} children written, "
          f"{report.children_skipped} already present, {report.previews_missing} without preview")
    print(f"Inventory: {N2_ROOT / 'patch_inventory_N2.csv'}")
