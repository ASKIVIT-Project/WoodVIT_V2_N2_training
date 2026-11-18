"""Instance counts: predicted class x annotated subclass x size class.

With the assignment of ``matching``, every annotated instance is counted once: in the row of the
class predicted for it, or in the row ``undetected``. The columns give the annotated category of
the training target (``GT_<category>``, from the ``category_id`` of the ground truth) and the
annotated subclass (from the ``class_value`` of the ground truth). Predictions without reference
are counted in ``pred_without_gt`` of their predicted class, sized by their own area.

Layout: for every class a row with size ``total``, then the rows of the sizes small, medium,
large.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from woodvit_rcnn.classes import BY_VALUE, SUBCLASS_VALUES
from woodvit_rcnn.matching import SIZE_NAMES, ImageData, ImageMatch, size_class

UNDETECTED = "undetected"
SUBCLASS_NAMES: tuple[str, ...] = tuple(BY_VALUE[v].name for v in SUBCLASS_VALUES)
KEY_COLUMNS: tuple[str, ...] = ("training_class", "iou_threshold", "confidence_threshold", "size")


def count_columns(categories: list[dict]) -> list[str]:
    return [*KEY_COLUMNS, *(f"GT_{c['name']}" for c in categories), *SUBCLASS_NAMES, "pred_without_gt"]


def count_instances(images: list[ImageData], matches: list[ImageMatch], categories: list[dict],
                    iou_threshold: float, conf_threshold: float) -> pd.DataFrame:
    """Count table of one IoU and one confidence threshold.

    ``categories`` are the COCO categories of the ground truth (and of the predictions). Every
    ground-truth annotation must carry ``class_value``.
    """
    names = {int(c["id"]): c["name"] for c in categories}
    # counts[row class][size][(annotated category id, class value)]
    counts: dict[str, dict[str, dict[tuple[int, int], int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    without_gt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def pred_class(pred: dict) -> str:
        try:
            return names[int(pred["category_id"])]
        except KeyError:
            raise ValueError(f"prediction category {pred['category_id']} is not a category of the ground truth") from None

    n_gt = 0
    for image, match in zip(images, matches):
        for ann in image.gt:
            if "class_value" not in ann:
                raise ValueError("ground-truth annotation without class_value; build the ground truth with "
                                 "coco_gt.build_coco_gt")
        n_gt += len(image.gt)
        assigned_class = {}
        for p in range(match.n_kept):
            g = int(match.gt_for_pred[p])
            if g < 0:
                without_gt[pred_class(image.preds[p])][size_class(image.pred_areas[p])] += 1
            else:
                assigned_class[g] = pred_class(image.preds[p])
        for g, ann in enumerate(image.gt):
            row = assigned_class.get(g, UNDETECTED)
            counts[row][size_class(float(ann["area"]))][(int(ann["category_id"]), int(ann["class_value"]))] += 1

    counted = sum(n for by_size in counts.values() for cell in by_size.values() for n in cell.values())
    if counted != n_gt:
        raise AssertionError(f"{n_gt} annotated instances, but {counted} counted")

    row_classes = [names[k] for k in sorted(names)] + [UNDETECTED]
    columns = count_columns(categories)

    def make_row(cls: str, size: str, sizes: tuple[str, ...]) -> dict:
        row = {"training_class": cls, "iou_threshold": float(iou_threshold),
               "confidence_threshold": float(conf_threshold), "size": size}
        for cat_id, cat_name in names.items():
            row[f"GT_{cat_name}"] = sum(n for s in sizes for (c, _), n in counts[cls][s].items() if c == cat_id)
        for value, sub in zip(SUBCLASS_VALUES, SUBCLASS_NAMES):
            row[sub] = sum(n for s in sizes for (_, v), n in counts[cls][s].items() if v == value)
        row["pred_without_gt"] = sum(without_gt[cls][s] for s in sizes)
        return row

    rows = [make_row(cls, "total", SIZE_NAMES) for cls in row_classes]
    rows += [make_row(cls, size, (size,)) for size in SIZE_NAMES for cls in row_classes]
    return pd.DataFrame(rows, columns=columns)
