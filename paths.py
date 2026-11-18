"""Locations of the dataset and of everything the scripts write.

Edit the four paths below for your machine; plain strings such as "/data/woodvit/patches_N2" work
as well. The defaults point into this repository (`data/` and `results/`, both ignored by git), so
the tracked version contains no machine-specific paths. To keep your local edits out of git, run
once:

    git update-index --skip-worktree paths.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# WoodVIT V2 patch dataset at level N=2 (24 cm quarters): <Sensor>/patches_<split>/ and
# patch_inventory_N2.csv. Read only.
DATA_ROOT = REPO_ROOT / "data" / "patches_N2"

# WoodVIT V2 patch dataset at level N=1 (48 cm patches). Only needed by
# preprocessing/subdivide_n1_to_n2.py, if you have N=1 patches only. Read only.
N1_ROOT = REPO_ROOT / "data" / "patches_N1"

# Trained models, predictions, metrics and late-fusion results.
RESULTS_ROOT = REPO_ROOT / "results"

# Early-fusion input (all four sensors resized to 256 x 256 and stacked, about 112 MB per patch).
EF_ROOT = Path(RESULTS_ROOT) / "early_fusion_256"

# The scripts expect Path objects.
DATA_ROOT, N1_ROOT, RESULTS_ROOT, EF_ROOT = (Path(p) for p in (DATA_ROOT, N1_ROOT, RESULTS_ROOT, EF_ROOT))
