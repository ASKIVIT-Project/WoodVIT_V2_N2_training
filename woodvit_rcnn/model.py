"""Mask R-CNN (ResNet-50 FPN v2) of the WoodVIT V2 baselines.

``build_model`` starts from torchvision's ``maskrcnn_resnet50_fpn_v2`` and adapts it to a sensor
and a training target:

* COCO weights when ``config.pretrained`` and no checkpoint is given. torchvision then freezes
  ``conv1``, ``bn1`` and ``layer1`` of the backbone (three trainable stages). Their BatchNorm
  layers are ordinary ``BatchNorm2d`` layers, so their running statistics still follow the
  training data although their weights are frozen.
* Inputs are rescaled internally to ``config.min_size`` / ``config.max_size`` (800 / 1333 by
  default, i.e. a 256 px patch is processed at 800 px).
* Anchor generator from ``config.anchor_sizes`` (one size per FPN level) and
  ``config.aspect_ratios``, with a new, randomly initialised RPN head of depth 1 (one 3 x 3
  convolution) that matches the number of anchors per location.
* Normalisation with the ImageNet mean and standard deviation, repeated cyclically over the
  channels (R, G, B, R, G, B, ...). No dataset statistics are used.
* For inputs with C != 3 channels the first convolution is replaced; its weights are the
  pretrained RGB weights repeated cyclically over the channels and cut to C. The new
  convolution is trainable.
* New box and mask predictors for ``num_classes`` classes (background included).

The model is wrapped in ``MaskRCNNWithEvalLoss`` so the validation loss can be computed in
evaluation mode without changing the BatchNorm statistics. Checkpoints hold the state dict of the
wrapper (keys start with ``model.``); the prefix is optional when loading. Checkpoints with three
anchors per location (torchvision's default anchor generator and RPN head) are recognised and
loaded with that architecture.

There is no class-agnostic NMS: the outputs are those of torchvision's Mask R-CNN (per-class NMS,
score threshold 0.05, at most 100 detections per image).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from woodvit_rcnn.presets import TrainConfig

IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)
COCO_WEIGHTS = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
TRAINABLE_BACKBONE_LAYERS = 3       # layer2, layer3, layer4; conv1, bn1 and layer1 stay frozen
FPN_LEVELS = 5                      # feature maps '0', '1', '2', '3', 'pool'
CHECKPOINT_PREFIX = "model."
LEGACY_NUM_ANCHORS = 3              # torchvision's default anchor generator (and RPN head of depth 2)
_RPN_CONV = re.compile(r"^rpn\.head\.conv\.(\d+)\.0\.weight$")


def resolve_device(device=None) -> torch.device:
    """``device`` as torch.device; None means CUDA if available, else CPU."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cyclic(values, n: int) -> list[float]:
    """``values`` repeated cyclically to length n."""
    return [float(values[i % len(values)]) for i in range(n)]


class MaskRCNNWithEvalLoss(nn.Module):
    """Mask R-CNN that also returns losses in evaluation mode.

    torchvision computes losses only in training mode, where BatchNorm layers update their
    running statistics. Called in evaluation mode with targets, this wrapper switches the model
    to training mode with statistics tracking disabled (BatchNorm then normalises with the batch
    statistics, as in training, but stores nothing), computes the losses without gradients and
    restores the evaluation mode. Without targets, or in training mode, it behaves like the
    wrapped model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        if self.training or targets is None:
            return self.model(images, targets)
        return self._loss_without_bn_update(images, targets)

    def _loss_without_bn_update(self, images, targets):
        bns = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        tracking = [m.track_running_stats for m in bns]
        was_training = self.model.training
        for m in bns:
            m.track_running_stats = False
        self.model.train()
        try:
            with torch.no_grad():
                return self.model(images, targets)
        finally:
            self.model.train(was_training)
            for m, t in zip(bns, tracking):
                m.track_running_stats = t


# ----------------------------------------------------------------------------- checkpoints
def read_checkpoint(path: str | Path) -> dict:
    """State dict of a checkpoint with the wrapper prefix ``model.`` removed."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(state).__name__}")
    return {k.removeprefix(CHECKPOINT_PREFIX): v for k, v in state.items()}


def checkpoint_architecture(state: dict) -> tuple[int, int]:
    """(anchors per location, depth of the RPN head) stored in a state dict."""
    key = "rpn.head.cls_logits.weight"
    if key not in state:
        raise ValueError(f"not a Mask R-CNN checkpoint: {key} missing")
    depth = sum(1 for k in state if _RPN_CONV.match(k))
    return int(state[key].shape[0]), depth


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save the state dict of the (wrapped) model atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- model
def _anchor_generator(config: TrainConfig) -> AnchorGenerator:
    if len(config.anchor_sizes) != FPN_LEVELS:
        raise ValueError(f"anchor_sizes needs one size per FPN level ({FPN_LEVELS}), got {config.anchor_sizes}")
    sizes = tuple((int(s),) for s in config.anchor_sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=(tuple(float(r) for r in config.aspect_ratios),) * len(sizes))


def _replace_first_conv(model: nn.Module, channels: int) -> None:
    old = model.backbone.body.conv1
    new = nn.Conv2d(channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    repeats = -(-channels // old.in_channels)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, repeats, 1, 1)[:, :channels])
    model.backbone.body.conv1 = new


def build_model(config: TrainConfig, num_classes: int, channels: int, checkpoint: str | Path | None = None,
                device=None) -> MaskRCNNWithEvalLoss:
    """Mask R-CNN for ``channels`` input channels and ``num_classes`` classes (background included).

    With ``checkpoint`` the weights are loaded from that file (no download); otherwise the COCO
    weights are used if ``config.pretrained``, else the model is randomly initialised.
    """
    state = read_checkpoint(checkpoint) if checkpoint is not None else None
    use_coco = state is None and config.pretrained
    kwargs = {
        "min_size": config.min_size, "max_size": config.max_size,
        "image_mean": cyclic(IMAGENET_MEAN, channels), "image_std": cyclic(IMAGENET_STD, channels),
    }
    if use_coco:
        model = maskrcnn_resnet50_fpn_v2(weights=COCO_WEIGHTS, trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
                                         **kwargs)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)

    num_anchors = len(config.aspect_ratios)
    if state is not None:
        stored_anchors, stored_depth = checkpoint_architecture(state)
        if stored_anchors == LEGACY_NUM_ANCHORS and stored_anchors != num_anchors:
            num_anchors = None          # keep torchvision's default anchor generator and RPN head
        elif stored_anchors != num_anchors or stored_depth != 1:
            raise ValueError(f"{checkpoint}: {stored_anchors} anchors and RPN depth {stored_depth} do not match "
                             f"the configuration ({num_anchors} aspect ratios, depth 1)")
    if num_anchors is not None:
        model.rpn.anchor_generator = _anchor_generator(config)
        model.rpn.head = RPNHead(in_channels=model.backbone.out_channels,
                                 num_anchors=model.rpn.anchor_generator.num_anchors_per_location()[0])

    if channels != 3:
        _replace_first_conv(model, channels)

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)
    mask_conv = model.roi_heads.mask_predictor.conv5_mask
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_conv.in_channels, mask_conv.out_channels, num_classes)

    if state is not None:
        model.load_state_dict(state)

    wrapped = MaskRCNNWithEvalLoss(model)
    return wrapped.to(resolve_device(device))


def frozen_parameters(model: nn.Module) -> list[str]:
    """Names of all parameters that are not trained (``requires_grad`` is False)."""
    return [name for name, p in model.named_parameters() if not p.requires_grad]
