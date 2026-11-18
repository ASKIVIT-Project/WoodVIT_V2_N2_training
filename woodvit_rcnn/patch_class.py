"""Class label of a whole patch, as used in the dataset file names.

The label is computed from the class channel of the patch's RGB-raster ground truth, in this
order (the rule of the released WoodVIT V2 dataset):

1. the largest non-wood class (covered classes included) with more than 15 % of the pixels
   -> ``Non_wood_<class>``
2. otherwise the largest uncovered wood class with more than 15 % -> ``Wood_<class>``
3. otherwise background with more than 80 % -> ``Non_wood_Background``
4. otherwise ``Undefined_Undefined``

On equal shares the smaller class value wins; values outside the class table count as
Undefined. The label is for naming only and never used for training.
"""
from __future__ import annotations

import numpy as np

from woodvit_rcnn.classes import BY_VALUE, CLASSES, UNDEFINED_VALUE

NON_WOOD_MIN = 0.15
WOOD_MIN = 0.15
BACKGROUND_MIN = 0.80

_WOOD = frozenset(c.value for c in CLASSES if c.group == "wood")
_NON_WOOD = frozenset(c.value for c in CLASSES if c.group in ("non_wood", "covered_wood", "covered_metal"))


def class_fractions(class_map: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(np.asarray(class_map), return_counts=True)
    total = float(np.asarray(class_map).size)
    out: dict[int, float] = {}
    for v, c in zip(values.tolist(), counts.tolist()):
        key = v if v in BY_VALUE else UNDEFINED_VALUE
        out[key] = out.get(key, 0.0) + c / total
    return out


def _largest(fractions: dict[int, float], group: frozenset[int]) -> tuple[int, float] | None:
    candidates = sorted(v for v in fractions if v in group)
    if not candidates:
        return None
    best = max(candidates, key=lambda v: fractions[v])     # max keeps the first (smallest value) on ties
    return best, fractions[best]


def patch_label(class_map: np.ndarray) -> str:
    """File-name label of a patch, e.g. ``Non_wood_Metal`` or ``Wood_Chipboard``."""
    fractions = class_fractions(class_map)
    for main, group, minimum in (("Non_wood", _NON_WOOD, NON_WOOD_MIN), ("Wood", _WOOD, WOOD_MIN)):
        hit = _largest(fractions, group)
        if hit is not None and hit[1] > minimum:
            return f"{main}_{BY_VALUE[hit[0]].name}"
    if fractions.get(0, 0.0) > BACKGROUND_MIN:
        return "Non_wood_Background"
    return "Undefined_Undefined"


def split_label(label: str) -> tuple[str, str]:
    """``Non_wood_Metal`` -> (``Non_wood``, ``Metal``)."""
    for main in ("Non_wood", "Wood", "Undefined"):
        if label.startswith(main + "_"):
            return main, label[len(main) + 1:]
    return "", label
