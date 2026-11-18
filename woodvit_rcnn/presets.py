"""Training configurations of the WoodVIT V2 baselines.

``make_config(preset)`` returns the configuration described in the paper for one baseline.
Any value can be overridden, e.g. ``make_config("rgb", num_epochs=1, probes=(47, 48))``.

Covered configurations (recordings 48, 49, 51, 54) are part of the training data only for the
models that see the THz data (``thz``, ``thz_metal``, ``ef``). Evaluation always uses the full
split, independent of the preset.

The ``thz`` preset feeds the model the THz channels 160 to 190 only (``channel_window``); all other
presets use every channel of their sensor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

from woodvit_rcnn.classes import get_target
from woodvit_rcnn.sensors import get_sensor


@dataclass(frozen=True)
class TrainConfig:
    preset: str
    sensor: str                         # RGB | NIR | Thermo | THz | EF
    target: str                         # main | metal | sub
    covered_in_training: bool           # use recordings 48, 49, 51, 54 for training and validation
    batch_size: int = 4

    # input
    target_size: tuple[int, int] = (256, 256)
    probes: tuple[int, ...] | None = None       # restrict to these recordings (None = all)
    channel_window: tuple[int, int] | None = None   # feed only the sensor channels [start, stop) (None = all)

    # optimisation: AdamW with a one-cycle schedule; the learning rate below is its peak value
    learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    warmup_fraction: float = 0.1
    num_epochs: int = 50
    patience: int = 25                  # stop after this many epochs without improvement ...
    min_delta: float = 0.001            # ... of the validation mask mAP by more than min_delta
    amp: bool = True                    # bfloat16 autocast on CUDA
    seed: int = 1234
    num_workers: int = 0

    # model
    pretrained: bool = True             # COCO weights of maskrcnn_resnet50_fpn_v2
    anchor_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    aspect_ratios: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)
    min_size: int = 800                 # torchvision rescales every input to this size internally
    max_size: int = 1333
    mask_threshold: float = 0.5         # binarisation of the predicted mask probabilities
    val_score_threshold: float = 0.05   # minimum score of predictions used for the validation mAP

    extra: dict = field(default_factory=dict)   # free notes, stored in config.json

    def __post_init__(self) -> None:
        get_sensor(self.sensor)
        get_target(self.target)
        if self.sensor == "EF" and self.target == "sub":
            raise ValueError("the early-fusion model is trained on the main or metal target only")
        if self.channel_window is not None:
            channels = get_sensor(self.sensor).channels
            if len(self.channel_window) != 2 or not 0 <= self.channel_window[0] < self.channel_window[1] <= channels:
                raise ValueError(f"channel window {self.channel_window} is not a range [start, stop) inside "
                                 f"0..{channels} ({self.sensor})")

    @property
    def input_channels(self) -> int:
        """Channels of the model input: the channel window, or all channels of the sensor."""
        if self.channel_window is None:
            return get_sensor(self.sensor).channels
        return self.channel_window[1] - self.channel_window[0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_size"] = list(self.target_size)
        d["probes"] = None if self.probes is None else list(self.probes)
        d["channel_window"] = None if self.channel_window is None else list(self.channel_window)
        d["anchor_sizes"] = list(self.anchor_sizes)
        d["aspect_ratios"] = list(self.aspect_ratios)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        d = dict(d)
        for key in ("target_size", "anchor_sizes", "aspect_ratios"):
            if key in d and d[key] is not None:
                d[key] = tuple(d[key])
        if d.get("probes") is not None:
            d["probes"] = tuple(int(p) for p in d["probes"])
        if d.get("channel_window") is None:
            # runs trained before the field existed stored the window as a note
            note = (d.get("extra") or {}).get(LEGACY_WINDOW_KEY)
            d["channel_window"] = None if note is None else (note["channel_start"], note["channel_stop"])
        if d["channel_window"] is not None:
            d["channel_window"] = tuple(int(v) for v in d["channel_window"])
        return cls(**d)


LEGACY_WINDOW_KEY = "thz_channel_window"     # key in ``extra`` of older runs: channel_start, channel_stop
THZ_WINDOW = (160, 191)                     # THz channels 160 to 190 inclusive, 31 of 200

PRESETS: dict[str, TrainConfig] = {
    "rgb": TrainConfig("rgb", "RGB", "main", covered_in_training=False),
    "nir": TrainConfig("nir", "NIR", "main", covered_in_training=False),
    "thermo": TrainConfig("thermo", "Thermo", "main", covered_in_training=False),
    "thz": TrainConfig("thz", "THz", "main", covered_in_training=True, channel_window=THZ_WINDOW),
    "thz_metal": TrainConfig("thz_metal", "THz", "metal", covered_in_training=True),
    "ef": TrainConfig("ef", "EF", "main", covered_in_training=True, batch_size=2),
    "rgb_sub": TrainConfig("rgb_sub", "RGB", "sub", covered_in_training=False),
}


def make_config(preset: str, **overrides) -> TrainConfig:
    """Configuration of a preset with optional overrides of single values."""
    try:
        base = PRESETS[preset]
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}, expected one of {sorted(PRESETS)}") from None
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    if overrides.get("probes") is not None:
        overrides["probes"] = tuple(int(p) for p in overrides["probes"])
    if overrides.get("target_size") is not None:
        overrides["target_size"] = tuple(overrides["target_size"])
    if overrides.get("channel_window") is not None:
        overrides["channel_window"] = tuple(int(v) for v in overrides["channel_window"])
    return replace(base, **overrides)
