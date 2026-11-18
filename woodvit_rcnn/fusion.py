"""Rule-based late fusion of four or five Mask R-CNN models.

The fused models have fixed roles: the RGB, NIR and Thermo models (target ``main``), the THz metal
expert (target ``metal``) and, optionally, the early-fusion model (EF, target ``main``). The EF role
may be left out (``OPTIONAL_ROLES``): the fusion then merges the four other models with the same
rules, and its results go to their own folder (``fusion_root``). Per image (patch):

1. Predictions with a confidence below ``score_threshold`` are dropped.
2. Two predictions of different models are linked if their masks overlap with an IoU of at least
   ``iou_threshold``. The connected components of these links form the groups; a prediction
   without any link stays single.
3. In a group, only the prediction with the highest score of each model counts (one RGB basis,
   one vote per model); further predictions of the same model in the group are dropped.
4. Mask: the RGB mask is the basis; pixels of the other masks are added where at least
   ``min_models_per_pixel`` of the other models agree. Without an RGB mask, the pixels where at
   least ``min_models_per_pixel`` models agree form the mask (their union if there are none).
5. Class: majority vote of the RGB, NIR and Thermo models. The metal expert votes only when it
   predicts Metal, which counts as Non_wood. The EF model contributes mask and score but does not
   vote. A tie goes to the class with the higher summed score (then to the lower category id).
   Without any vote the class is Non_wood.
6. Score: the maximum score of the group (``score_fusion="max"``) or the mean.
7. Single predictions are kept unchanged if their score reaches ``single_score_threshold``, unless
   they stem from the metal expert, whose single predictions are always dropped.

Boxes of all fused results are the inclusive extent of their masks (``coco_utils.mask_bbox``).
Results carry the late-fusion image id (sensor digit ``sensors.LF_INDEX``) and the roles of the
contributing models (``models``). The ground truth is the RGB ground truth with the same image ids.
"""
from __future__ import annotations

import datetime as _dt
import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from woodvit_rcnn import __version__
from woodvit_rcnn.classes import MAIN, METAL
from woodvit_rcnn.coco_gt import build_coco_gt
from woodvit_rcnn.coco_utils import mask_bbox, mask_to_rle, rle_to_mask
from woodvit_rcnn.early_fusion import SOURCE_FILE as EF_SOURCE_FILE
from woodvit_rcnn.early_fusion import SOURCE_INVENTORY_KEY as EF_SOURCE_KEY
from woodvit_rcnn.evaluate import (FILES, LATE_FUSION, LATE_FUSION_NO_EF, OPERATING_CONF, OPERATING_IOU, evaluate_run,
                                   metrics_at)
from woodvit_rcnn.inventory import find_inventory
from woodvit_rcnn.run_io import (check_dataset_identity, core_id, dataset_identity, find_run, read_json,
                                 read_predictions, with_sensor, write_json, write_predictions)
from woodvit_rcnn.sensors import LF_INDEX

# role -> (sensor, target) of the model that fills it
ROLES: dict[str, tuple[str, str]] = {
    "RGB": ("RGB", "main"),
    "NIR": ("NIR", "main"),
    "Thermo": ("Thermo", "main"),
    "THz_metal": ("THz", "metal"),
    "EF": ("EF", "main"),
}
BASE_ROLE = "RGB"
METAL_ROLE = "THz_metal"
SILENT_ROLES: tuple[str, ...] = ("EF",)
OPTIONAL_ROLES: tuple[str, ...] = ("EF",)       # roles that may be left out of the fusion
WOOD = MAIN.categories.index("Wood") + 1
NON_WOOD = MAIN.categories.index("Non_wood") + 1
METAL_CATEGORY = METAL.categories.index("Metal") + 1
SCORE_FUSIONS: tuple[str, ...] = ("max", "mean")
FUSION_FORMAT = "woodvit_rcnn.late_fusion/1"
PREDICTION_FILE = "pred_fused.json"
CONFIG_FILE = "config_fusion.json"


@dataclass(frozen=True)
class FusionParams:
    """Parameters of the late fusion; the defaults are the values of the paper."""

    score_threshold: float = 0.5            # minimum confidence of a prediction to enter the fusion
    iou_threshold: float = 0.6              # mask IoU that links predictions of different models
    min_models_per_pixel: int = 3           # other models that must agree on a pixel to add it
    single_score_threshold: float = 0.5     # minimum confidence of a prediction without partner
    score_fusion: str = "max"               # max | mean

    def __post_init__(self) -> None:
        if self.score_fusion not in SCORE_FUSIONS:
            raise ValueError(f"score_fusion must be one of {SCORE_FUSIONS}, got {self.score_fusion!r}")
        if self.min_models_per_pixel < 1:
            raise ValueError("min_models_per_pixel must be at least 1")

    def folder_name(self, target_size) -> str:
        """Name of the output folder, e.g. ``max_256_pred0.5_iou0.6_conf1s0.5_3s_per_px``."""
        return (f"{self.score_fusion}_{int(target_size[0])}_pred{self.score_threshold}_iou{self.iou_threshold}_"
                f"conf1s{self.single_score_threshold}_{self.min_models_per_pixel}s_per_px")


Member = tuple[str, dict]       # (role, prediction)


def fusion_roles(given) -> tuple[str, ...]:
    """The roles of a fusion, in the order of ``ROLES``. ``given`` maps roles to model folders or
    to prediction files; an optional role (EF) that is absent or empty is left out, any other
    absent or empty role and any unknown role is an error."""
    unknown = [role for role in given if role not in ROLES]
    if unknown:
        raise ValueError(f"unknown roles {unknown}; the roles are {list(ROLES)}")
    unset = [role for role in ROLES if not given.get(role) and role not in OPTIONAL_ROLES]
    if unset:
        raise ValueError(f"nothing set for the roles {unset}")
    return tuple(role for role in ROLES if given.get(role))


def fusion_root(roles) -> str:
    """Name of the results folder of a fusion of ``roles``: ``Late_Fusion`` for all five models,
    ``Late_Fusion_noEF`` without the early-fusion model. The two never share a folder, so runs with
    the same parameters cannot overwrite each other or be mixed in the parameter search."""
    return LATE_FUSION if "EF" in roles else LATE_FUSION_NO_EF


# ----------------------------------------------------------------------------- fusion rules
def group_members(members: list[Member], iou_threshold: float) -> list[list[int]]:
    """Connected components of the members; two members of different models are linked if their
    mask IoU reaches ``iou_threshold``. Components are lists of member indices in input order."""
    n = len(members)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        segs = [m[1]["segmentation"] for m in members]
        ious = np.asarray(mask_utils.iou(segs, segs, [0] * n), dtype=float).reshape(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                if members[i][0] != members[j][0] and ious[i, j] >= iou_threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)
    return sorted(components.values(), key=lambda c: c[0])


def best_per_model(members: list[Member], component: list[int]) -> list[int]:
    """The member with the highest score of each model in a component (first one on equal score)."""
    best: dict[str, int] = {}
    for i in component:
        role, pred = members[i]
        if role not in best or float(pred["score"]) > float(members[best[role]][1]["score"]):
            best[role] = i
    return sorted(best.values())


def vote(members: list[Member]) -> int:
    """Fused main category of a group (one member per model)."""
    votes: dict[int, int] = defaultdict(int)
    score_sums: dict[int, float] = defaultdict(float)
    for role, pred in members:
        if role in SILENT_ROLES:
            continue
        if role == METAL_ROLE:
            if int(pred["category_id"]) != METAL_CATEGORY:
                continue
            category = NON_WOOD
        else:
            category = int(pred["category_id"])
        votes[category] += 1
        score_sums[category] += float(pred["score"])
    if not votes:
        return NON_WOOD
    most = max(votes.values())
    candidates = [c for c in votes if votes[c] == most]
    if len(candidates) > 1:
        top = max(score_sums[c] for c in candidates)
        candidates = [c for c in candidates if score_sums[c] == top]
    return min(candidates)


def fuse_group(members: list[Member], params: FusionParams) -> dict:
    """Fused result of a group with one member per model."""
    roles = [role for role, _ in members]
    if len(set(roles)) != len(roles):
        raise ValueError(f"fuse_group expects one prediction per model, got {roles}")
    masks = {role: rle_to_mask(pred["segmentation"]).astype(bool) for role, pred in members}
    others = [mask for role, mask in masks.items() if role != BASE_ROLE]
    if BASE_ROLE in masks:
        fused = masks[BASE_ROLE].copy()
        if others:
            fused |= np.sum(others, axis=0) >= params.min_models_per_pixel
    else:
        stack = np.stack(list(masks.values()))
        fused = stack.sum(axis=0) >= params.min_models_per_pixel
        if not fused.any():
            fused = stack.any(axis=0)
    scores = [float(pred["score"]) for _, pred in members]
    score = max(scores) if params.score_fusion == "max" else float(np.mean(scores))
    return {"category_id": vote(members), "score": score, "bbox": mask_bbox(fused),
            "segmentation": mask_to_rle(fused), "models": roles}


def keep_single(member: Member, params: FusionParams) -> dict | None:
    """A prediction without partner: kept unchanged unless it stems from the metal expert."""
    role, pred = member
    if role == METAL_ROLE or float(pred["score"]) < params.single_score_threshold:
        return None
    return {"category_id": int(pred["category_id"]), "score": float(pred["score"]),
            "bbox": mask_bbox(rle_to_mask(pred["segmentation"])), "segmentation": pred["segmentation"],
            "models": [role]}


def fuse_image(preds_by_role: dict[str, list[dict]], params: FusionParams) -> list[dict]:
    """Fused results of one image (without image id); roles missing from the dict have no predictions."""
    members = [(role, pred) for role in ROLES for pred in preds_by_role.get(role, [])
               if float(pred["score"]) >= params.score_threshold]
    fused = []
    for component in group_members(members, params.iou_threshold):
        if len(component) == 1:
            single = keep_single(members[component[0]], params)
            if single is not None:
                fused.append(single)
        else:
            fused.append(fuse_group([members[i] for i in best_per_model(members, component)], params))
    return fused


def fuse_predictions(predictions: dict[str, dict], params: FusionParams, progress: bool = False
                     ) -> tuple[list[dict], list[int]]:
    """Late fusion of the prediction files of the given roles (all of ``ROLES``, or without the
    optional ones): (results, late-fusion image ids)."""
    roles = fusion_roles(predictions)
    by_core: dict[str, dict[int, list[dict]]] = {role: defaultdict(list) for role in roles}
    for role in roles:
        for res in predictions[role]["results"]:
            by_core[role][core_id(res["image_id"])].append(res)
    cores = sorted({core_id(i) for i in predictions[BASE_ROLE]["image_ids"]})
    results = []
    for core in tqdm(cores, desc="Late fusion", disable=not progress):
        lf_id = with_sensor(core, LF_INDEX)
        for res in fuse_image({role: by_core[role].get(core, []) for role in roles}, params):
            results.append({"image_id": lf_id, **res})
    return results, [with_sensor(c, LF_INDEX) for c in cores]


# ----------------------------------------------------------------------------- inputs
def check_fusion_inputs(predictions: dict[str, dict], ef_root: str | Path | None = None) -> None:
    """The prediction files fit together: roles, split, raster, images and dataset build. The
    early-fusion role may be absent (fusion of four models).

    The early-fusion predictions are computed on the early-fusion folder, which has its own
    inventory. They are accepted if they carry the same dataset identity as the others, or if they
    carry the identity of the inventory in ``ef_root`` and the fingerprint of that folder names the
    dataset of the other predictions.
    """
    roles = fusion_roles(predictions)
    for role in roles:
        sensor, target = ROLES[role]
        data = predictions[role]
        if data.get("sensor") != sensor or data.get("target") != target:
            raise ValueError(f"role {role} needs a {sensor} model with target {target!r}, got "
                             f"{data.get('sensor')} / {data.get('target')!r}")
    for key in ("split", "target_size"):
        values = {role: predictions[role].get(key) for role in roles}
        if len({str(v) for v in values.values()}) != 1:
            raise ValueError(f"the prediction files differ in {key}: {values}")
    reference = predictions[BASE_ROLE].get("dataset")
    for role in roles:
        if role == "EF":
            continue
        check_dataset_identity(reference, predictions[role].get("dataset"), f"{role} predictions")
    ef = predictions["EF"].get("dataset") if "EF" in roles else None
    if "EF" in roles and (not ef or ef.get("sha1") != reference.get("sha1")):
        if ef_root is None:
            raise ValueError("the EF predictions carry another dataset identity than the RGB predictions; "
                             "pass the early-fusion folder (ef_root) to check its source")
        check_dataset_identity(dataset_identity(find_inventory(ef_root)), ef, "EF predictions")
        source = Path(ef_root) / EF_SOURCE_FILE
        found = read_json(source).get(EF_SOURCE_KEY) if source.is_file() else None
        check_dataset_identity(reference, found, f"source of the early-fusion folder ({source})")
    cores = {role: {core_id(i) for i in predictions[role]["image_ids"]} for role in roles}
    for role in roles:
        if cores[role] != cores[BASE_ROLE]:
            raise ValueError(f"{role} predictions cover other images than the RGB predictions "
                             f"({len(cores[role] ^ cores[BASE_ROLE])} differ)")


@dataclass
class FusionInputs:
    """Everything a late-fusion run needs, loaded once (e.g. for a parameter grid)."""

    model_folders: dict[str, str]       # fused roles only
    predictions: dict[str, dict]
    gt: dict
    split: str
    target_size: tuple[int, int]

    @property
    def roles(self) -> tuple[str, ...]:
        """The fused roles, in the order of ``ROLES``."""
        return fusion_roles(self.predictions)


def load_fusion_inputs(results_root: str | Path, model_folders: dict[str, str], split: str,
                       data_root: str | Path, ef_root: str | Path | None = None,
                       progress: bool = False) -> FusionInputs:
    """Prediction files of the fused models (``model_folders``: role -> run folder name) and the RGB
    ground truth of the split with late-fusion image ids. An empty or absent ``EF`` entry gives the
    fusion of four models; ``ef_root`` is not used then."""
    roles = fusion_roles(model_folders)
    predictions = {role: read_predictions(find_run(results_root, model_folders[role]).predictions(split))
                   for role in roles}
    check_fusion_inputs(predictions, ef_root)
    target_size = tuple(int(v) for v in predictions[BASE_ROLE]["target_size"])
    gt, _ = build_coco_gt(data_root, "RGB", "main", split, target_size=target_size, include_covered=True,
                          image_sensor_index=LF_INDEX, progress=progress)
    return FusionInputs({role: model_folders[role] for role in roles}, predictions, gt, split, target_size)


def fusion_folder(results_root: str | Path, split: str, params: FusionParams, target_size,
                  roles=tuple(ROLES)) -> Path:
    """Output folder of a fusion of ``roles``: ``<results>/<fusion_root(roles)>/<split>/<parameter folder>``."""
    return Path(results_root) / fusion_root(roles) / split / params.folder_name(target_size)


def run_late_fusion(inputs: FusionInputs, params: FusionParams, results_root: str | Path, evaluate: bool = True,
                    plots: bool = True, progress: bool = False) -> Path:
    """Fuse, write ``pred_fused.json`` and ``config_fusion.json`` to
    ``<results>/Late_Fusion/<split>/<parameter folder>/`` (``Late_Fusion_noEF`` without the
    early-fusion model) and evaluate into its ``metrics/``."""
    roles = inputs.roles
    folder = fusion_folder(results_root, inputs.split, params, inputs.target_size, roles)
    results, image_ids = fuse_predictions(inputs.predictions, params, progress=progress)
    rgb = inputs.predictions[BASE_ROLE]
    write_predictions(folder / PREDICTION_FILE, results, image_ids=image_ids, sensor="LF", target="main",
                      split=inputs.split, target_size=inputs.target_size, dataset=rgb["dataset"],
                      source=f"{fusion_root(roles).lower()}/{inputs.split}/{folder.name}")
    write_json(folder / CONFIG_FILE, {
        "format": FUSION_FORMAT, "code_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "split": inputs.split, "target_size": list(inputs.target_size), "dataset": rgb["dataset"],
        "roles": list(roles), "model_folders": {role: inputs.model_folders[role] for role in roles},
        "params": asdict(params),
    })
    if evaluate:
        evaluate_run(inputs.gt, read_predictions(folder / PREDICTION_FILE), folder / "metrics", plots=plots)
    return folder


def parameter_grid(score_thresholds, iou_thresholds, min_models, single_score_thresholds,
                   score_fusions=("max",)) -> list[FusionParams]:
    """All parameter sets of a grid with a single-prediction threshold not below the input threshold."""
    return [FusionParams(score_threshold=s, iou_threshold=i, min_models_per_pixel=m, single_score_threshold=c,
                         score_fusion=f)
            for f, i, s, m, c in itertools.product(score_fusions, iou_thresholds, score_thresholds, min_models,
                                                   single_score_thresholds)
            if c >= s]


# ----------------------------------------------------------------------------- parameter selection
# Selection criteria of the parameter search: metric -> ascending (True = smaller is better).
SELECTION: dict[str, bool] = {"segm_mAP": False, "nMCC_classified": False, "C_OSA": False, "M_OSA": True}


def parameter_table(results_root: str | Path, split: str = "val", iou: float = OPERATING_IOU,
                    conf: float = OPERATING_CONF, roles=tuple(ROLES)) -> pd.DataFrame:
    """One row per late-fusion folder of a split (fusion of ``roles``): parameters and metrics at
    (iou, conf). The fusion parameters keep their names (``iou_threshold`` is the grouping IoU); the
    thresholds of the evaluation are the columns ``eval_iou_threshold`` and ``eval_conf_threshold``."""
    rows = []
    for folder in sorted((Path(results_root) / fusion_root(roles) / split).glob("*")):
        metrics_dir = folder / "metrics"
        if not (folder / CONFIG_FILE).is_file() or not (metrics_dir / FILES["summary"]).is_file():
            continue
        params = read_json(folder / CONFIG_FILE)["params"]
        metrics = metrics_at(metrics_dir, iou, conf)
        for key in ("iou_threshold", "conf_threshold"):     # do not overwrite the fusion parameter of the same name
            metrics[f"eval_{key}"] = metrics.pop(key)
        rows.append({"folder": folder.name, **params, **metrics})
    return pd.DataFrame(rows)


def top_lists(table: pd.DataFrame, n: int = 3) -> dict[str, pd.DataFrame]:
    """The ``n`` best folders per selection criterion (ties ordered by folder name)."""
    out = {}
    for metric, ascending in SELECTION.items():
        valid = table.dropna(subset=[metric])
        out[metric] = valid.sort_values([metric, "folder"], ascending=[ascending, True], kind="stable").head(n)
    return out


def write_top_lists(results_root: str | Path, split: str = "val", iou: float = OPERATING_IOU,
                    conf: float = OPERATING_CONF, n: int = 3, roles=tuple(ROLES)
                    ) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Write ``all_metrics_per_folder.csv``, ``top<n>_<metric>.csv`` and ``combined_top<n>_per_folder.csv``
    to ``<results>/<fusion_root(roles)>/<split>/top_metrics/``; returns the folder and the top lists."""
    base = Path(results_root) / fusion_root(roles) / split
    table = parameter_table(results_root, split, iou, conf, roles)
    if table.empty:
        raise FileNotFoundError(f"no evaluated late-fusion folders under {base}")
    out_dir = base / "top_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "all_metrics_per_folder.csv", index=False)
    tops = top_lists(table, n)
    for metric, df in tops.items():
        df.to_csv(out_dir / f"top{n}_{metric}.csv", index=False)
    combined = pd.concat(tops.values()).drop_duplicates(subset="folder")
    combined.to_csv(out_dir / f"combined_top{n}_per_folder.csv", index=False)
    return out_dir, tops
