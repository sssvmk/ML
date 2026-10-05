"""Hungarian matching and the set-prediction loss (paper section 3.1): L = class CE + 5 * L1 + 2 * GIoU per matched pair."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from detr_voc.ops import box_cxcywh_to_xyxy, giou_matrix, solve_assignment


class HungarianMatcher(nn.Module):
    """One-to-one assignment of predictions to ground truth minimising  w_c * (-p(class)) + w_b * L1 + w_g * (-GIoU)."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        bs, nq = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        if sum(sizes) == 0:
            e = torch.zeros(0, dtype=torch.int64)
            return [(e, e)] * bs
        prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)
        boxes = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([t["labels"] for t in targets]).to(prob.device)
        tgt_boxes = torch.cat([t["boxes"] for t in targets]).to(boxes.device).float()
        cost = (self.cost_bbox * torch.cdist(boxes, tgt_boxes, p=1) - self.cost_class * prob[:, tgt_ids]
                - self.cost_giou * giou_matrix(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        cost = cost.view(bs, nq, -1).cpu()
        out = []
        for i, c in enumerate(cost.split(sizes, -1)):
            r, col = solve_assignment(c[i].numpy())
            out.append((torch.as_tensor(r, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return out


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weights: dict, eos_coef: float):
        super().__init__()
        self.num_classes, self.matcher, self.weights = num_classes, matcher, weights
        w = torch.ones(num_classes + 1)
        w[-1] = eos_coef                                   # paper: "no object" down-weighted by a factor 10
        self.register_buffer("empty_weight", w, persistent=False)

    def _single(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"].float()
        idx = (torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)]), torch.cat([s for s, _ in indices]))
        classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
        classes[idx] = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices, strict=True)]).to(logits.device)
        loss_ce = F.cross_entropy(logits.transpose(1, 2), classes, self.empty_weight)
        src = outputs["pred_boxes"].float()[idx]
        tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices, strict=True)]).to(src.device).float()
        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / num_boxes
        giou = torch.diag(giou_matrix(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt))) if len(src) else src.sum() * 0
        loss_giou = (1 - giou).sum() / num_boxes if len(src) else src.sum() * 0
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        """Returns the weighted total plus every component (and the per-decoder-layer auxiliary losses)."""
        num_boxes = max(float(sum(len(t["labels"]) for t in targets)), 1.0)
        main = self._single(outputs, targets, self.matcher(outputs, targets), num_boxes)
        parts = dict(main)
        total = sum(self.weights[k] * v for k, v in main.items())
        for n, aux in enumerate(outputs.get("aux_outputs", [])):
            a = self._single(aux, targets, self.matcher(aux, targets), num_boxes)
            parts.update({f"{k}_{n}": v for k, v in a.items()})
            total = total + sum(self.weights[k] * v for k, v in a.items())
        return {"total": total, **{k: v.detach() for k, v in parts.items()}}
