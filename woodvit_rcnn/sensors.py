"""Sensors of WoodVIT V2 and the early-fusion stack.

``index`` is the leading digit of all image ids of a sensor (see ``run_io.image_id``).
``stored_bgr`` marks sensors whose ``.npy`` files hold the colour channels in the order
B, G, R (OpenCV convention). They are flipped to R, G, B when loaded.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    name: str               # folder name in the dataset
    index: int              # leading digit of the image ids
    channels: int           # channels of the .npy files (Thermo files are 2-D, one channel)
    stored_bgr: bool = False
    n2_raster: int | None = None    # edge length of an N=2 patch in pixels


RGB = Sensor("RGB", 1, 3, stored_bgr=True, n2_raster=1447)
NIR = Sensor("NIR", 2, 224, n2_raster=239)
THERMO = Sensor("Thermo", 3, 1, n2_raster=240)
THZ = Sensor("THz", 4, 200, n2_raster=225)

# Early fusion: the four sensors resized to a common raster and stacked along the channel axis
# in this order. The RGB part is stored as R, G, B (already flipped), so EF is never flipped again.
EF_ORDER: tuple[str, ...] = ("RGB", "NIR", "Thermo", "THz")
EF = Sensor("EF", 5, sum(s.channels for s in (RGB, NIR, THERMO, THZ)))

# Image-id index of late-fusion results (there are no late-fusion input files).
LF_INDEX = 6

SENSORS: dict[str, Sensor] = {s.name: s for s in (RGB, NIR, THERMO, THZ, EF)}
SINGLE_SENSORS: tuple[str, ...] = ("RGB", "NIR", "Thermo", "THz")


def get_sensor(name: str) -> Sensor:
    try:
        return SENSORS[name]
    except KeyError:
        raise ValueError(f"unknown sensor {name!r}, expected one of {sorted(SENSORS)}") from None


def ef_channel_ranges() -> dict[str, tuple[int, int]]:
    """Channel range [start, stop) of every sensor inside the early-fusion stack."""
    ranges, start = {}, 0
    for name in EF_ORDER:
        stop = start + SENSORS[name].channels
        ranges[name] = (start, stop)
        start = stop
    return ranges
