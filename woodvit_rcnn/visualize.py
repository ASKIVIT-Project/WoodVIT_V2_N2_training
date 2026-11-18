"""Predictions next to the ground truth, drawn on the sensor image.

The background image is the preview PNG of the patch if it exists, otherwise it is made from the
sensor data: RGB data directly (as R, G, B), the RGB part of early-fusion data, and for all other
sensors the maximum over the channels, stretched to the full grey range. It is resized to the
target size of the predictions. Masks are blended in (overlapping masks share one averaged
colour), boxes and labels are drawn on top. With two categories (Wood / Non_wood, Metal /
Non_metal) the first category is drawn in shades of green, the second in shades of blue.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from woodvit_rcnn.coco_utils import rle_to_mask
from woodvit_rcnn.dataset import load_image
from woodvit_rcnn.inventory import PatchRecord, find_inventory, read_inventory, select_records
from woodvit_rcnn.run_io import RunPaths, dataset_identity, read_json, read_predictions, read_run_config
from woodvit_rcnn.sensors import ef_channel_ranges, get_sensor

SCORE_THRESHOLD = 0.5
ALPHA = 0.6
GREENS = ((144, 238, 144), (34, 139, 34), (0, 200, 0), (85, 170, 85), (0, 128, 0), (154, 205, 50), (60, 179, 113),
          (0, 100, 0))
BLUES = ((135, 206, 250), (65, 105, 225), (0, 150, 255), (100, 149, 237), (0, 0, 205), (138, 43, 226), (70, 130, 180),
         (25, 25, 112))
PALETTE = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
           (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
           (152, 223, 138), (255, 152, 150))


def colour(category_id: int, index: int, num_categories: int) -> tuple[int, int, int]:
    """Colour of the index-th annotation of an image with the given category."""
    if num_categories == 2 and category_id in (1, 2):
        shades = GREENS if category_id == 1 else BLUES
        return shades[index % len(shades)]
    return PALETTE[(int(category_id) - 1) % len(PALETTE)]


def base_image(record: PatchRecord, data_root: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Background image of a patch as uint8 (H, W, 3) R, G, B array of the given (H, W) size."""
    preview = record.preview_path(data_root)
    if preview.is_file():
        image = np.asarray(Image.open(preview).convert("RGB"))
    else:
        sensor = get_sensor(record.sensor)
        data = load_image(record.data_path(data_root), sensor.channels, flip_bgr=sensor.stored_bgr)
        if sensor.name == "RGB":
            image = data
        elif sensor.name == "EF":
            start, stop = ef_channel_ranges()["RGB"]
            image = data[..., start:stop]
        else:
            grey = data.max(axis=2)
            peak = float(grey.max())
            grey = grey / peak if peak > 0 else grey
            image = np.repeat(grey[..., None], 3, axis=2)
        image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = size
    if image.shape[:2] != (h, w):
        shrink = image.shape[0] > h or image.shape[1] > w
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return np.ascontiguousarray(image)


def draw_annotations(image: np.ndarray, annotations: list[dict], category_names: dict[int, str],
                     show_scores: bool = True, alpha: float = ALPHA) -> np.ndarray:
    """Masks, boxes and labels of COCO annotations or results drawn on a uint8 (H, W, 3) image."""
    if not annotations:
        return image.copy()
    num_categories = len(category_names)
    colours = [colour(a["category_id"], i, num_categories) for i, a in enumerate(annotations)]

    out = image.astype(np.float32)
    accum = np.zeros(out.shape, dtype=np.float32)
    count = np.zeros(out.shape[:2], dtype=np.float32)
    for ann, c in zip(annotations, colours):
        mask = rle_to_mask(ann["segmentation"]).astype(bool)
        accum[mask] += c
        count[mask] += 1
    covered = count > 0
    out[covered] = (1 - alpha) * out[covered] + alpha * (accum[covered] / count[covered, None])

    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in annotations)], dtype=torch.float32)
    labels = []
    for a in annotations:
        name = category_names.get(int(a["category_id"]), str(a["category_id"]))
        labels.append(f"{name}:{a['score']:.2f}" if show_scores and "score" in a else name)
    drawn = draw_bounding_boxes(torch.from_numpy(out.astype(np.uint8)).permute(2, 0, 1).contiguous(), boxes,
                                labels=labels, colors=colours, width=1)
    return drawn.permute(1, 2, 0).contiguous().numpy()


def side_by_side(left: np.ndarray, right: np.ndarray, titles=("Prediction", "Ground truth"),
                 gap: int = 10, title_height: int = 20) -> Image.Image:
    """Two images of equal size next to each other with a title above each."""
    h, w = left.shape[:2]
    canvas = Image.new("RGB", (2 * w + gap, h + title_height), color=(255, 255, 255))
    canvas.paste(Image.fromarray(left), (0, title_height))
    canvas.paste(Image.fromarray(right), (w + gap, title_height))
    draw = ImageDraw.Draw(canvas)
    for x0, title in ((0, titles[0]), (w + gap, titles[1])):
        draw.text((x0 + 4, 4), title, fill=(0, 0, 0))
    return canvas


def visualize_split(run: RunPaths, data_root: str | Path, split: str, out_dir: str | Path | None = None,
                    score_threshold: float = SCORE_THRESHOLD, image_ids=None, max_images: int | None = None,
                    progress: bool = True) -> list[Path]:
    """One PNG per image: predictions with a score of at least ``score_threshold`` next to the
    ground truth. Reads the prediction and ground-truth files of the run; writes to
    ``out_dir`` (default ``<run>/visualisation/<split>``). Returns the written files."""
    config, _ = read_run_config(run)
    predictions = read_predictions(run.predictions(split), expected_dataset=dataset_identity(find_inventory(data_root)))
    gt = read_json(run.ground_truth(split))
    names = {int(c["id"]): c["name"] for c in gt["categories"]}
    size = tuple(predictions["target_size"])

    pred_by_image, gt_by_image = defaultdict(list), defaultdict(list)
    for r in predictions["results"]:
        if r["score"] >= score_threshold:
            pred_by_image[int(r["image_id"])].append(r)
    for a in gt["annotations"]:
        gt_by_image[int(a["image_id"])].append(a)
    records = {r.image_id: r for r in select_records(read_inventory(data_root), sensor=config.sensor, split=split)}

    ids = [int(i) for i in (predictions["image_ids"] if image_ids is None else image_ids)]
    if max_images is not None:
        ids = ids[:max_images]
    out_dir = Path(out_dir) if out_dir is not None else run.root / "visualisation" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for img_id in tqdm(ids, desc=f"Visualise {split}", disable=not progress):
        if img_id not in records:
            raise KeyError(f"image {img_id} is not in the {config.sensor} {split} split of {data_root}")
        base = base_image(records[img_id], data_root, size)
        left = draw_annotations(base, pred_by_image[img_id], names, show_scores=True)
        right = draw_annotations(base, gt_by_image[img_id], names, show_scores=False)
        path = out_dir / f"{img_id}_comparison.png"
        side_by_side(left, right).save(path)
        written.append(path)
    return written
