# Metal_Expert_2: THz metal expert, second version

A Mask R-CNN for Metal versus Non_metal on the THz data that starts from the `thz_metal` preset and
changes six things (library module `woodvit_rcnn/metal_expert_2.py`, settings stored with every run in
`config.json` under `extra["metal_expert_2"]`):

| Change | Setting |
|---|---|
| Channel window | THz channels 160 to 190 inclusive (31 of 200) |
| Normalisation | per-channel mean and standard deviation of the training split instead of ImageNet values |
| Augmentation | the THz baseline list without CoarseDropout and GaussNoise |
| Backbone | all layers trainable; first convolution = repeated RGB weights scaled by 3 / 31 |
| Sampling | training images with covered metal (class values 110, 120, 130) drawn 3 times as often |
| Selection | checkpoint and early stopping by the segm AP at IoU 0.5 of the class Metal on `val` |

Everything else is the baseline. Runs are normal run folders under `Trained_Models/THz/`, so
`metrics/main_metrics.py` and the late fusion read them unchanged. The baseline inference cannot load
these checkpoints (31 input channels); use the scripts of this folder.

## Scripts

All settings are variables at the top of `config_metal_expert_2.py` (learning rates, channel window,
repeat factor, selection metric, overrides of the preset).

1. `compute_channel_stats.py`: per-channel statistics of the training split, written as JSON under
   `RESULTS_ROOT/metal_expert_2/`. Run once per dataset build; the training scripts refuse statistics
   whose inventory hash differs from the dataset.
2. `smoke_test_metal_expert_2.py`: one training and one validation step, writes nothing.
3. `main_train_metal_expert_2.py`: one run per learning rate, one after the other.
4. `main_inference_metal_expert_2.py`: predictions and ground truth for `val` and `test` in the format of
   the baseline inference.
5. `metrics/main_metrics.py` (baseline, unchanged): set `MODEL_FOLDER` to the run.
6. `analyse_covered_metal.py`: detection of open metal (70) and the covered-metal classes (110, 120, 130)
   as Metal, as Non_metal, or not at all, at IoU 0.5 and 0.3 with confidence 0.5, plus the share of their
   pixels under Metal predictions. Output `metrics/<split>/covered_metal_analysis.csv`.

## Results (dataset build of 2026-10-22, 50 recordings)

Three runs with seed 1234 and peak learning rates 5e-5, 1e-4 and 2e-4; the run with the best segm AP50
of Metal on `val` (2e-4, 0.238; the others 0.228 and 0.231) was evaluated once on `test`:

| test | segm mAP | segm mAP50 | segm AP Metal | mIoU | C-OSA | U-OSA | Precision | Recall | nMCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ME2, lr 2e-4 | 0.190 | 0.373 | 0.108 | 0.502 | 0.669 | 0.295 | 0.954 | 0.216 | 0.351 |

Covered metal (confidence 0.5, IoU 0.5, `test`): of 416 annotated covered-metal instances 1 is detected
as Metal, 16 are detected but called Non_metal and 399 are not detected at all; open metal: 32.5 % as
Metal, 9.8 % as Non_metal, 57.7 % not detected. Only 6 to 9 % of the covered-metal pixels lie under Metal
predictions. Covered metal is therefore missed, not misnamed: a separate Covered_metal class would not
help. The expert does not detect covered metal better than the first version.
