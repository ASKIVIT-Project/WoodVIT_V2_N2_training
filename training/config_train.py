"""Training configuration read by main_train.py and smoke_test.py.

Choose one preset per baseline of the paper:

    rgb, nir, thermo   single-sensor models, Wood vs Non_wood, trained without covered recordings
    thz                THz model on the THz channels 160 to 190, Wood vs Non_wood, trained with covered
                       recordings
    thz_metal          THz metal expert, Metal vs Non_metal, trained with covered recordings
    ef                 early fusion (428 channels), Wood vs Non_wood, with covered recordings, batch 2
    rgb_sub            RGB model on the 13 subclasses (Undefined instances are left out)

OVERRIDES changes single values of the preset, for example
{"num_epochs": 1, "probes": (3, 4, 47)} for a quick run on three recordings (together they
contain train, val and test patches).
All available keys are the fields of woodvit_rcnn.presets.TrainConfig.
"""
from woodvit_rcnn.presets import make_config

PRESET = "rgb"
OVERRIDES: dict = {}

# None: a new folder named after sensor, date, target and settings.
# A folder name: continue writing into an existing run folder (it must not contain a checkpoint).
MODEL_FOLDER = None

CONFIG = make_config(PRESET, **OVERRIDES)
