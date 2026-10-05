"""SSD box machinery: default boxes, matching, MultiBox loss with hard negative mining, decoding + NMS (paper sec. 2)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], dim=-1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2]], dim=-1)


def box_iou_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [B,G,4] xyxy, b [P,4] xyxy -> IoU [B,G,P]."""
    lt = torch.max(a[:, :, None, :2], b[None, None, :, :2])
    rb = torch.min(a[:, :, None, 2:], b[None, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, :, None] + area_b[None, None, :] - inter).clamp(min=1e-9)


def prior_scales(num_maps: int, first_scale: float, min_scale: float, max_scale: float) -> list[float]:
    """Paper eq. (4): s_k = s_min + (s_max - s_min)/(m - 1) * (k - 1) for the later maps; the first map gets its own
    smaller scale; one extra scale (same step) serves the sqrt(s_k * s_k+1) box of the last map."""
    later = num_maps - 1
    step = (max_scale - min_scale) / max(later - 1, 1)
    return [first_scale] + [min_scale + step * k for k in range(later)] + [max_scale + step]


def generate_priors(feature_sizes, boxes_per_loc, first_scale=0.1, min_scale=0.2, max_scale=0.9, clip=True):
    """Default boxes as (cx, cy, w, h) in [0, 1]. 6 boxes per cell use aspect ratios {1, 1', 2, 1/2, 3, 1/3}; 4 omit 3, 1/3."""
    scales = prior_scales(len(feature_sizes), first_scale, min_scale, max_scale)
    out = []
    for k, (f, nb) in enumerate(zip(feature_sizes, boxes_per_loc, strict=True)):
        s, s_next = scales[k], scales[k + 1]
        r2, r3 = math.sqrt(2), math.sqrt(3)
        shapes = [(s, s), (math.sqrt(s * s_next),) * 2, (s * r2, s / r2), (s / r2, s * r2)]
        if nb == 6:
            shapes += [(s * r3, s / r3), (s / r3, s * r3)]
        elif nb != 4:
            raise ValueError("boxes_per_loc entries must be 4 or 6")
        for i in range(f):
            for j in range(f):
                for w, h in shapes:
                    out.append([(j + 0.5) / f, (i + 0.5) / f, w, h])
    t = torch.tensor(out, dtype=torch.float32)
    return t.clamp_(0.0, 1.0) if clip else t


def encode(matched_xyxy: torch.Tensor, priors: torch.Tensor, variances) -> torch.Tensor:
    m = xyxy_to_cxcywh(matched_xyxy)
    d_cxcy = (m[..., :2] - priors[..., :2]) / (variances[0] * priors[..., 2:])
    d_wh = torch.log(m[..., 2:].clamp(min=1e-6) / priors[..., 2:]) / variances[1]
    return torch.cat([d_cxcy, d_wh], dim=-1)


def decode(loc: torch.Tensor, priors: torch.Tensor, variances) -> torch.Tensor:
    cxcy = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
    wh = priors[..., 2:] * torch.exp((loc[..., 2:] * variances[1]).clamp(max=4.0))
    return cxcywh_to_xyxy(torch.cat([cxcy, wh], dim=-1))


def pad_targets(targets: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """list of {boxes [n,4] normalised xyxy, labels [n]} -> padded tensors and a validity mask."""
    g = max((len(t["labels"]) for t in targets), default=0)
    b = len(targets)
    boxes = torch.zeros(b, g, 4, device=device)
    labels = torch.zeros(b, g, dtype=torch.long, device=device)
    valid = torch.zeros(b, g, dtype=torch.bool, device=device)
    for i, t in enumerate(targets):
        n = len(t["labels"])
        if n:
            boxes[i, :n], labels[i, :n], valid[i, :n] = t["boxes"].to(device), t["labels"].to(device), True
    return boxes, labels, valid


def match_priors(gt_boxes, gt_labels, valid, priors_xyxy, threshold=0.5):
    """Paper sec. 2.2: every ground truth takes its best-overlapping default box, then every default box with
    IoU >= threshold to some ground truth is positive. Returns labels [B,P] (0 = background) and boxes [B,P,4]."""
    b, g, _ = gt_boxes.shape
    p = priors_xyxy.shape[0]
    if g == 0:
        return torch.zeros(b, p, dtype=torch.long, device=priors_xyxy.device), torch.zeros(b, p, 4,
                                                                                             device=priors_xyxy.device)
    iou = box_iou_batched(gt_boxes, priors_xyxy).masked_fill(~valid[:, :, None], -1.0)
    best_prior_idx = iou.argmax(dim=2)                      # [B,G]
    best_gt_iou, best_gt_idx = iou.max(dim=1)               # [B,P]
    bi, gi = torch.nonzero(valid, as_tuple=True)
    pi = best_prior_idx[bi, gi]
    best_gt_idx[bi, pi] = gi                                 # force each ground truth to keep its best prior
    best_gt_iou[bi, pi] = 2.0
    labels = gt_labels.gather(1, best_gt_idx)
    labels = torch.where(best_gt_iou >= threshold, labels, torch.zeros_like(labels))
    boxes = gt_boxes.gather(1, best_gt_idx[:, :, None].expand(-1, -1, 4))
    return labels, boxes


class MultiBoxLoss(nn.Module):
    """L = (L_conf + alpha * L_loc) / N, softmax confidence with 3:1 hard negative mining, smooth-L1 localisation."""

    def __init__(self, priors_cxcywh: torch.Tensor, iou_threshold=0.5, neg_pos_ratio=3, alpha=1.0,
                 variances=(0.1, 0.2)):
        super().__init__()
        self.register_buffer("priors", priors_cxcywh.clone(), persistent=False)
        self.register_buffer("priors_xyxy", cxcywh_to_xyxy(priors_cxcywh), persistent=False)
        self.iou_threshold, self.neg_pos_ratio, self.alpha, self.variances = iou_threshold, neg_pos_ratio, alpha, variances

    def forward(self, loc_p: torch.Tensor, conf_p: torch.Tensor, targets: list[dict]) -> dict:
        bsz, n_priors, n_cls = conf_p.shape
        gt_boxes, gt_labels, valid = pad_targets(targets, conf_p.device)
        labels, matched = match_priors(gt_boxes, gt_labels, valid, self.priors_xyxy, self.iou_threshold)
        pos = labels > 0
        loc_t = encode(matched, self.priors, self.variances)
        loc_loss = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction="sum")
        ce = F.cross_entropy(conf_p.reshape(-1, n_cls), labels.reshape(-1), reduction="none").view(bsz, n_priors)
        with torch.no_grad():
            neg_score = ce.detach().clone()
            neg_score[pos] = 0.0
            rank = neg_score.argsort(dim=1, descending=True).argsort(dim=1)
            num_pos = pos.sum(dim=1, keepdim=True)
            num_neg = (self.neg_pos_ratio * num_pos).clamp(max=n_priors - 1)
            neg = (rank < num_neg) & ~pos
        conf_loss = ce[pos | neg].sum()
        n = num_pos.sum().clamp(min=1).float()
        return {"total": (conf_loss + self.alpha * loc_loss) / n, "conf": conf_loss / n, "loc": loc_loss / n,
                "num_pos": num_pos.sum()}


@torch.no_grad()
def postprocess(loc, conf, priors_cxcywh, variances=(0.1, 0.2), score_thresh=0.01, nms_iou=0.45, max_det=200,
                pre_nms_top_k=2000) -> list[dict]:
    """Decode, drop scores below the threshold, per-class NMS (IoU 0.45), keep the best `max_det` (paper sec. 3.7).
    Boxes are normalised xyxy; labels are 1..C-1."""
    probs = conf.float().softmax(-1)[..., 1:]
    boxes_all = decode(loc.float(), priors_cxcywh, variances).clamp(0.0, 1.0)
    results = []
    for b in range(probs.shape[0]):
        sc = probs[b]
        pi, ci = (sc > score_thresh).nonzero(as_tuple=True)
        s = sc[pi, ci]
        if s.numel() > pre_nms_top_k:
            top = s.topk(pre_nms_top_k).indices
            pi, ci, s = pi[top], ci[top], s[top]
        bx = boxes_all[b][pi]
        keep = batched_nms(bx, s, ci, nms_iou)[:max_det]
        results.append({"boxes": bx[keep], "scores": s[keep], "labels": ci[keep] + 1})
    return results
