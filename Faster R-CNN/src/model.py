"""Faster R-CNN (the production descendant of R-CNN / Fast R-CNN) with a replaced, properly initialised box predictor."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models import detection as det
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ARCHS = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": dict(
        fn=det.fasterrcnn_mobilenet_v3_large_320_fpn, coco=det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
        backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, min_size=320, max_size=640),
    "fasterrcnn_mobilenet_v3_large_fpn": dict(
        fn=det.fasterrcnn_mobilenet_v3_large_fpn, coco=det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, min_size=800, max_size=1333),
    "fasterrcnn_resnet50_fpn": dict(
        fn=det.fasterrcnn_resnet50_fpn, coco=det.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        backbone=ResNet50_Weights.IMAGENET1K_V1, min_size=800, max_size=1333),
    "fasterrcnn_resnet50_fpn_v2": dict(
        fn=det.fasterrcnn_resnet50_fpn_v2, coco=det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        backbone=ResNet50_Weights.IMAGENET1K_V1, min_size=800, max_size=1333),
}


def init_predictor(pred: FastRCNNPredictor) -> None:
    """Detectron-style init: tiny weights so the initial classification loss is ~ln(K) and boxes start as 'no change'."""
    nn.init.normal_(pred.cls_score.weight, std=0.01)
    nn.init.normal_(pred.bbox_pred.weight, std=0.001)
    nn.init.zeros_(pred.cls_score.bias)
    nn.init.zeros_(pred.bbox_pred.bias)


def build_model(model_cfg: dict, num_foreground: int, pretrained: str | None = None) -> nn.Module:
    """`pretrained` overrides model_cfg['pretrained']; inference code passes 'none' so no weights are downloaded."""
    arch = model_cfg["arch"]
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {sorted(ARCHS)}")
    spec = ARCHS[arch]
    mode = pretrained or model_cfg.get("pretrained", "coco")
    k = num_foreground + 1  # + background
    min_size = model_cfg.get("min_size") or [spec["min_size"]]
    kw = dict(min_size=tuple(min_size) if isinstance(min_size, (list, tuple)) else (int(min_size),),
              max_size=int(model_cfg.get("max_size") or spec["max_size"]),
              box_score_thresh=float(model_cfg.get("score_floor", 0.05)),
              box_nms_thresh=float(model_cfg.get("nms_thresh", 0.5)),
              box_detections_per_img=int(model_cfg.get("detections_per_img", 100)))
    if model_cfg.get("box_batch_size_per_image"):
        kw["box_batch_size_per_image"] = int(model_cfg["box_batch_size_per_image"])
    tbl = model_cfg.get("trainable_backbone_layers")
    if mode == "coco":
        if tbl is not None:
            kw["trainable_backbone_layers"] = int(tbl)
        model = spec["fn"](weights=spec["coco"], **kw)  # 91 COCO classes; predictor replaced below
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, k)
    elif mode == "imagenet":
        if tbl is not None:
            kw["trainable_backbone_layers"] = int(tbl)
        model = spec["fn"](weights=None, weights_backbone=spec["backbone"], num_classes=k, **kw)
    elif mode == "none":
        model = spec["fn"](weights=None, weights_backbone=None, num_classes=k, **kw)
    else:
        raise ValueError("pretrained must be coco | imagenet | none")
    init_predictor(model.roi_heads.box_predictor)
    p = float(model_cfg.get("box_head_dropout", 0.0))
    if p > 0:  # dropout on the box-head representation (Deep Learning book 7.12); off by default
        model.roi_heads.box_head = nn.Sequential(model.roi_heads.box_head, nn.Dropout(p))
    return model


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def set_loss_mode(model: nn.Module) -> nn.Module:
    """Train mode (so torchvision returns losses) but with BatchNorm and Dropout in eval: no stat updates, no noise."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
            m.eval()
    return model
