# WoodVIT V2: Mask R-CNN instance segmentation baselines

Training and evaluation code of the instance segmentation baselines of the WoodVIT V2 dataset
(RGB, hyperspectral NIR, thermography and THz data of bulky waste with instance masks):

* single-sensor Mask R-CNN models for RGB, NIR, IR (called `Thermo` in the code and the data) and THz,
  with the classes `Wood` and `Non_wood` (spelled as in the dataset files),
* a THz "metal expert" (Metal versus Non_metal; the version used by the late fusion is
  `metal_expert_2/`),
* early fusion of all four sensors (428 input channels),
* a rule-based late fusion of four models (RGB, NIR, Thermo and the THz metal expert; the
  early-fusion model can be added as a fifth),
* the evaluation: COCO metrics, the object surface area (OSA), instance-count metrics and mIoU.

The code reproduces the baselines described in the WoodVIT V2 data descriptor (see "How to cite").

## Requirements

* Python 3.11
* An NVIDIA GPU with a driver for CUDA 12.8 (the pinned wheels support current GPUs, including
  the RTX 50 series). Training on the CPU works for tests, but is far too slow for real runs.
* Disk space: the N=2 dataset, plus about 310 GB for the early-fusion input, plus about 0.2 GB per
  trained model.

## Installation

```
python -m venv .venv
.venv\Scripts\activate            (Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions used for the baselines (PyTorch 2.9.1, torchvision 0.24.1,
pycocotools 2.0.10, albumentations 2.0.8, ...). `pip install -e .` makes the library `woodvit_rcnn`
and `paths.py` importable from every script folder.

## Data

Download WoodVIT V2 (DOI: to be added). The code reads the patches at level N=2 (24 cm quarters of
the 48 cm patches):

```
patches_N2/
  patch_inventory_N2.csv            one row per sensor file: split, recording, patch, x, y, path
  RGB/patches_train/  RGB/patches_val/  RGB/patches_test/
  NIR/...  Thermo/...  THz/...
      <stem>.npy                    sensor data, float32 in [0, 1], H x W x C (Thermo: H x W)
      <stem>_GT.png                 ground truth, uint16, 3 channels (class value, instance id, unused)
      <stem>.png                    8-bit preview (optional, only for viewing)
```

* The inventory is the only source of split, recording and patch position; file names are not parsed.
* RGB data are stored in the channel order B, G, R; the code flips them to R, G, B when loading.
* Training and evaluation accept N=2 patches only (inventory column `size` = `24.0cm`).
* If you only have the N=1 patches (48 cm), cut them into N=2 with
  `preprocessing/subdivide_n1_to_n2.py` (exact quarters, no interpolation; needs about as much disk
  space as the N=1 data). For the released dataset the result equals the released N=2 patches bit
  for bit (sensor data and ground truth); only the previews of NIR, Thermo and THz differ, because
  they keep the contrast stretch of the 48 cm patch.

## Configuration

All scripts are plain Python files with their settings as variables at the top. There is no command
line interface: edit the variables, then run the script (`python training/main_train.py` or from an IDE).

The data and output locations are set once in [paths.py](paths.py):

| Variable | Meaning |
|---|---|
| `DATA_ROOT` | the `patches_N2` folder (read only) |
| `N1_ROOT` | the `patches_N1` folder, only for the subdivision script |
| `RESULTS_ROOT` | trained models, predictions, metrics, late fusion |
| `EF_ROOT` | the early-fusion input built by `preprocessing/create_ef_data.py` |

To keep your local paths out of git: `git update-index --skip-worktree paths.py`.

## Quick start: one RGB model

The shortest complete pass trains and evaluates the RGB model. It needs the inventory and the `RGB/`
folder of the N=2 dataset and a GPU; the early-fusion input is not needed.

1. Set `DATA_ROOT` and `RESULTS_ROOT` in `paths.py`.
2. Keep `PRESET = "rgb"` in `training/config_train.py` and run `training/smoke_test.py`, then
   `training/main_train.py`. The run folder `RGB_N2_<date>_<time>_MAIN_256_lr5e-05_pat25` appears
   under `RESULTS_ROOT/Trained_Models/RGB/`.
3. Enter this folder name as `MODEL_FOLDER` in `inference/main_inference.py` and run the script.
4. Enter the same name in `metrics/main_metrics.py` and run it. The values of the paper table are in
   `metrics/test/operating_point.csv` of the run folder.

For a first look without a full training, set
`OVERRIDES = {"num_epochs": 1, "probes": (3, 4, 47)}` in `config_train.py`: one epoch on three
recordings. 

## Reproducing the baselines

1. **Check the data** with `preprocessing/check_data.py`: files against the inventory, shapes, value
   range, instances per class and split.
2. **Build the early-fusion input** (only for the `ef` model) with `preprocessing/create_ef_data.py`:
   all four sensors resized to 256 x 256 and stacked (428 channels, about 112 MB per patch).
3. **Train** each model: set `PRESET` in `training/config_train.py`, run `training/smoke_test.py`
   (one training and one validation step, writes nothing), then `training/main_train.py`.

   | Preset | Model | Target | Covered recordings in training |
   |---|---|---|---|
   | `rgb`, `nir`, `thermo` | single sensor | Wood / Non_wood | no |
   | `thz` | THz, channels 160 to 190 only (31 of 200) | Wood / Non_wood | yes |
   | `thz_metal` | THz metal expert, first version (not used by the paper) | Metal / Non_metal | yes |
   | `ef` | early fusion, batch size 2 | Wood / Non_wood | yes |
   | `rgb_sub` | RGB | 13 subclasses | no |

   All presets: input 256 x 256 px, COCO-pretrained `maskrcnn_resnet50_fpn_v2`, AdamW with a
   one-cycle schedule (peak learning rate 5e-5), batch size 4, up to 50 epochs, early stopping after
   25 epochs without improvement of the validation mask mAP. Single values can be changed with
   `OVERRIDES` in `config_train.py`, e.g. `{"num_epochs": 1, "probes": (3, 4, 47)}` for a quick run.
   The **metal expert of the late fusion** is not a preset: train and predict it with the scripts
   of `metal_expert_2/` (channel statistics, smoke test, training, inference; see
   [metal_expert_2/README.md](metal_expert_2/README.md)). Its run folder is evaluated and fused
   like any other.
4. **Predict** the `val` and `test` splits with `inference/main_inference.py` (`MODEL_FOLDER` = the
   run folder name). Predictions always cover the full split, covered recordings included.
5. **Evaluate** with `metrics/main_metrics.py` (one model) or `metrics/run_metrics_for_multiple.py`
   (all models).
6. **Late fusion**:
   1. Enter the four run folders (RGB, NIR, Thermo, THz metal expert) in
      `late_fusion/config_fusion.py` and leave the `EF` entry empty. This is the late fusion of the
      paper. With an early-fusion run entered under `EF`, five models are fused instead.
   2. Keep the parameters of `config_fusion.py`: the defaults are the paper values (input score 0.5,
      IoU 0.6, at least 3 models per pixel, single predictions from score 0.5, maximum score).
   3. Set `SPLIT = "test"` and run `late_fusion/late_fusion.py`. The results go to
      `Late_Fusion_noEF/` (four models) or `Late_Fusion/` (five models), so the two variants never
      overwrite each other.
   4. Optional: a grid search over the parameters on `val` with
      `late_fusion/main_late_fusion_hyper.py`, then `late_fusion/find_best_lf_params.py`, which
      writes top-3 lists for segm mAP, nMCC (classified), C-OSA and M-OSA.
7. **Collect the table** with `metrics/extract_wanted_metrics.py` (split `test`, IoU 0.5,
   confidence 0.8).
8. Optional: `metrics/main_occluded_cases.py` reports the detection of the covered-metal classes
   (class values 110, 120, 130) for a run folder or a fused prediction file.
9. Optional: `inference/visualisation.py` draws predictions next to the ground truth.

## Outputs

```
RESULTS_ROOT/
  Trained_Models/<Sensor>/<model folder>/
    training/   config.json, checkpoint.pth, training_metadata.json, train_log.csv, train.log, tensorboard/
    inference/  pred_<split>.json, gt_<split>.json
    metrics/<split>/  summary_metrics.csv, operating_point.csv, iou_metric.csv, conf_metric.csv,
                      iou_cm_metrics.csv, conf_cm_metrics.csv, instance_counts_subclasses.csv, ...
    visualisation/<split>/
  Late_Fusion_noEF/<split>/<parameter folder>/  pred_fused.json, config_fusion.json, metrics/
                      (late fusion of four models; Late_Fusion/ with the early-fusion model as fifth)
  combined_metrics/   one row per model at the operating point (the paper table)
  data_check/
```

`operating_point.csv` holds the values of the paper table at IoU 0.5 and confidence 0.8: C-OSA,
M-OSA, U-OSA, Coverage, Accuracy, Precision, Recall, F1, nMCC (undetected instances counted as
errors), mIoU and the COCO mAP values. Every run records the identity (hash) of the dataset
inventory it was trained on; evaluation and late fusion refuse to combine predictions and ground
truth from different inventories. The hash covers the inventory file only: a data or ground-truth
file replaced in place under an unchanged inventory is not detected. All CSV files use a comma as
separator and a dot as decimal mark.

## Repository layout

```
paths.py            data and output locations (edit for your machine)
preprocessing/      check_data, create_ef_data, subdivide_n1_to_n2
training/           config_train, main_train, smoke_test
inference/          main_inference, visualisation
metrics/            main_metrics, run_metrics_for_multiple, extract_wanted_metrics, main_occluded_cases
late_fusion/        config_fusion, late_fusion, main_late_fusion_hyper, find_best_lf_params
metal_expert_2/     THz metal expert, second version (own README)
woodvit_rcnn/       the library: classes, dataset, model, training loop, metrics, fusion
```

## How to cite

If you use this code, please cite the WoodVIT V2 data descriptor (reference to be added on
publication).
