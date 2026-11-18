"""Class table of the WoodVIT V2 ground truth and the three training targets.

A ground-truth PNG is a 16-bit, three-channel image. In OpenCV's channel order (B, G, R) the
red channel (index 2) holds the class value, the green channel (index 1) the instance id, and
the blue channel is unused. Instance id 0 means "no instance".

Training targets:

* ``main``:  Wood versus Non_wood. Wood are the five uncovered wood classes (values 10 to 50).
  Everything else, including Covered_solid_wood_under_plastic (60) and Undefined (140), is
  Non_wood.
* ``metal``: Metal versus Non_metal (the metal expert). Metal are Metal (70) and the three
  covered-metal classes (110 to 130). Everything else is Non_metal.
* ``sub``:   the 13 subclasses (values 10 to 130). Undefined (140) is not a category of this
  target: its instances are left out of training and ground truth (``Target.ignored_values``).
  The dataset has no Undefined instance in the training split.

Category ids are 1-based; 0 is the background of Mask R-CNN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAPPING_VERSION = 1


@dataclass(frozen=True)
class GTClass:
    name: str
    value: int      # value in the class channel of the ground-truth PNG
    group: str      # background | wood | covered_wood | non_wood | covered_metal | undefined


CLASSES: tuple[GTClass, ...] = (
    GTClass("Background", 0, "background"),
    GTClass("Chipboard", 10, "wood"),
    GTClass("Solid_wood", 20, "wood"),
    GTClass("Medium_density_fiberboard", 30, "wood"),
    GTClass("Oriented_strand_board", 40, "wood"),
    GTClass("Other_Wood", 50, "wood"),
    GTClass("Covered_solid_wood_under_plastic", 60, "covered_wood"),
    GTClass("Metal", 70, "non_wood"),
    GTClass("Plastic", 80, "non_wood"),
    GTClass("Ceramics", 90, "non_wood"),
    GTClass("Other_non_wood", 100, "non_wood"),
    GTClass("Covered_metal_under_chipboard", 110, "covered_metal"),
    GTClass("Covered_metal_under_solid_wood", 120, "covered_metal"),
    GTClass("Covered_metal_under_plastic", 130, "covered_metal"),
    GTClass("Undefined", 140, "undefined"),
)

BY_VALUE: dict[int, GTClass] = {c.value: c for c in CLASSES}
BY_NAME: dict[str, GTClass] = {c.name: c for c in CLASSES}
BACKGROUND_VALUE = 0
UNDEFINED_VALUE = 140

# All values an annotated instance can carry (everything except Background).
SUBCLASS_VALUES: tuple[int, ...] = tuple(c.value for c in CLASSES if c.value != BACKGROUND_VALUE)
COVERED_METAL_VALUES: frozenset[int] = frozenset(c.value for c in CLASSES if c.group == "covered_metal")


@dataclass(frozen=True)
class Target:
    """A training target: category names and the mapping from ground-truth class values."""

    name: str
    categories: tuple[str, ...]                 # category id = index + 1
    value_to_category: dict[int, int] = field(compare=False)
    ignored_values: frozenset[int] = frozenset()    # class values whose instances are left out

    @property
    def num_classes(self) -> int:
        """Number of Mask R-CNN classes including the background."""
        return len(self.categories) + 1

    def category_id(self, value: int) -> int:
        """Category id of a ground-truth class value. Raises KeyError for unknown values."""
        try:
            return self.value_to_category[int(value)]
        except KeyError:
            raise KeyError(f"class value {value} has no category in target {self.name!r}") from None

    def category_name(self, category_id: int) -> str:
        return self.categories[int(category_id) - 1]

    def coco_categories(self) -> list[dict]:
        return [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(self.categories)]


def _binary_target(name: str, positive: str, negative: str, positive_values: set[int]) -> Target:
    mapping = {v: (1 if v in positive_values else 2) for v in SUBCLASS_VALUES}
    return Target(name=name, categories=(positive, negative), value_to_category=mapping)


MAIN = _binary_target("main", "Wood", "Non_wood", {c.value for c in CLASSES if c.group == "wood"})
METAL = _binary_target("metal", "Metal", "Non_metal", {70} | set(COVERED_METAL_VALUES))
_SUB_VALUES = tuple(v for v in SUBCLASS_VALUES if v != UNDEFINED_VALUE)
SUB = Target(
    name="sub",
    categories=tuple(BY_VALUE[v].name for v in _SUB_VALUES),
    value_to_category={v: i + 1 for i, v in enumerate(_SUB_VALUES)},
    ignored_values=frozenset({UNDEFINED_VALUE}),
)

TARGETS: dict[str, Target] = {t.name: t for t in (MAIN, METAL, SUB)}


def get_target(name: str) -> Target:
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"unknown target {name!r}, expected one of {sorted(TARGETS)}") from None


def class_name(value: int) -> str:
    """Name of a ground-truth class value. Raises KeyError for values outside the table."""
    return BY_VALUE[int(value)].name
