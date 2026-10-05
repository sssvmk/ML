"""Inference helpers: image preprocessing, post-processing (no NMS) and a checkpoint predictor."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from detr_voc.model import build_model
from detr_voc.ops import box_cxcywh_to_xyxy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def aspect_size(w: int, h: int, size: int, max_size: int | None) -> tuple[int, int]:
    """(height, width) after resizing the shorter side to `size`, the longer side capped at max_size (official DETR rule)."""
    if max_size is not None:
        lo, hi = float(min(w, h)), float(max(w, h))
        if hi / lo * size > max_size:
            size = int(round(max_size * lo / hi))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def to_normalised_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - m) / s


def denormalize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)


@torch.no_grad()
def postprocess(outputs: dict, sizes: torch.Tensor) -> list[dict]:
    """Paper sec. 4: every query yields one detection. Its class is the best REAL class (the 'no object' slot is overridden
    by the second-highest class, which the paper reports as +2 AP) and its score is that class's probability.
    sizes: [B, 2] original (width, height). Boxes are xyxy in original pixels; labels are 1..K."""
    prob = F.softmax(outputs["pred_logits"].float(), -1)[..., :-1]
    scores, labels = prob.max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"].float())
    w, h = sizes[:, 0].float(), sizes[:, 1].float()
    boxes = boxes * torch.stack([w, h, w, h], dim=1)[:, None, :].to(boxes.device)
    return [{"scores": s, "labels": lb + 1, "boxes": b} for s, lb, b in zip(scores, labels, boxes, strict=True)]


def load_checkpoint(path, device="cpu"):
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(st["state_dict"])
    return model.to(device).eval(), cfg, st["classes"]


class Predictor:
    """Run a trained checkpoint on images (paths, PIL images or HxWx3 uint8 arrays). No MLflow, no downloads."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.classes = load_checkpoint(checkpoint, self.device)

    @staticmethod
    def _pil(item) -> Image.Image:
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("RGB")
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        arr = np.asarray(item)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("array input must be uint8 with shape (H, W, 3)")
        return Image.fromarray(arr)

    @torch.inference_mode()
    def predict(self, images, score_threshold=0.5, max_detections=100) -> list[dict]:
        aug = self.cfg["aug"]
        out = []
        for item in images:
            img = self._pil(item)
            w, h = img.size
            oh, ow = aspect_size(w, h, aug["test_size"], aug["max_size"])
            x = to_normalised_tensor(img.resize((ow, oh), Image.BILINEAR))[None].to(self.device)
            res = postprocess(self.model(x, torch.zeros(1, oh, ow, dtype=torch.bool, device=self.device)),
                              torch.tensor([[w, h]]))[0]
            order = res["scores"].argsort(descending=True)[:max_detections]
            dets = [{"box": [round(float(v), 2) for v in res["boxes"][i]], "score": round(float(res["scores"][i]), 5),
                     "label_id": int(res["labels"][i]), "label": self.classes[int(res["labels"][i]) - 1]}
                    for i in order.tolist() if float(res["scores"][i]) >= score_threshold]
            out.append({"width": w, "height": h, "detections": dets})
        return out
