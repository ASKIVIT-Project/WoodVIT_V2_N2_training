"""Settings of the THz metal expert, second version (Metal_Expert_2), read by the scripts in this folder.

Every run starts from the ``thz_metal`` preset (THz, Metal versus Non_metal, covered recordings in
training, 256 px) and differs from the baseline only in the points implemented in
woodvit_rcnn/metal_expert_2.py: channel window, normalisation with the statistics of the training
split, augmentation without CoarseDropout and GaussNoise, fully trainable backbone with a scaled
first convolution, more frequent sampling of images with covered metal, and selection by the AP50
of the class Metal. One run per learning rate, same seed, trained one after the other.
"""
from woodvit_rcnn.metal_expert_2 import MetalExpert2Settings

LEARNING_RATES = (5e-5, 1e-4, 2e-4)     # peak learning rates of the one-cycle schedule, one run each
CHANNEL_START = 160                     # first THz channel of the window ...
CHANNEL_STOP = 191                      # ... and one past the last: channels 160 to 190 inclusive, 31 channels
COVERED_METAL_REPEAT = 3                # training images with covered metal are drawn this often
SELECTION_METRIC = "segm_AP50_Metal"    # checkpoint and early stopping: segm AP at IoU 0.5 of the class Metal
OVERRIDES: dict = {}                    # single values of the preset, e.g. {"num_epochs": 1, "probes": (3, 4, 47)}

# Per-channel statistics of the training split (compute_channel_stats.py), relative to paths.RESULTS_ROOT.
CHANNEL_STATS_FILE = f"metal_expert_2/channel_stats_THz_train_ch{CHANNEL_START}_{CHANNEL_STOP - 1}.json"

SETTINGS = MetalExpert2Settings(channel_start=CHANNEL_START, channel_stop=CHANNEL_STOP,
                                covered_metal_repeat=COVERED_METAL_REPEAT, selection_metric=SELECTION_METRIC)
