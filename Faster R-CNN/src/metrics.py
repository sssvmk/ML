"""Detection metrics: VOC-protocol AP (11-point VOC07 and all-point), COCO-style mAP@[.5:.95], AR@100, error breakdown.

Matching follows the Pascal VOC devkit: detections are sorted by score; each takes the ground-truth box of its class
with the highest IoU; if IoU >= threshold the detection is a true positive unless that GT was already matched
(duplicate -> false positive); 'difficult' GTs are ignored (neither TP nor FP, and not counted as positives).
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for XYXY boxes (continuous coordinates, no +1)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def ap_all_points(rec: np.ndarray, prec: np.ndarray) -> float:
    """Area under the monotone precision envelope (VOC2010+ / COCO-like)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_11_point(rec: np.ndarray, prec: np.ndarray) -> float:
    """Official Pascal VOC2007 metric: mean of max precision at recall >= 0, 0.1, ..., 1.0."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return float(ap)


class DetectionEvaluator:
    def __init__(self, class_names: list[str]):
        self.class_names = list(class_names)
        self.k = len(class_names)
        self.preds: list[dict] = []
        self.gts: list[dict] = []

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        """preds: dicts with boxes[N,4], scores[N], labels[N] (1..K); targets: boxes, labels, optional difficult."""
        for p, t in zip(preds, targets, strict=True):
            self.preds.append({"boxes": np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4),
                               "scores": np.asarray(p["scores"], dtype=np.float64).reshape(-1),
                               "labels": np.asarray(p["labels"], dtype=np.int64).reshape(-1)})
            n = len(np.asarray(t["labels"]))
            diff = np.asarray(t["difficult"], dtype=bool) if "difficult" in t else np.zeros(n, dtype=bool)
            self.gts.append({"boxes": np.asarray(t["boxes"], dtype=np.float64).reshape(-1, 4),
                             "labels": np.asarray(t["labels"], dtype=np.int64).reshape(-1), "difficult": diff})

    def _class_curves(self, c: int, thresholds):
        """For class c: per threshold -> (recall, precision) arrays, plus npos."""
        npos = sum(int(np.sum((g["labels"] == c) & ~g["difficult"])) for g in self.gts)
        dets = []  # (score, img, det_idx_in_class)
        per_img = []
        for i, (p, g) in enumerate(zip(self.preds, self.gts, strict=True)):
            m = p["labels"] == c
            gm = g["labels"] == c
            boxes_d, scores_d = p["boxes"][m], p["scores"][m]
            iou = box_iou_np(boxes_d, g["boxes"][gm])
            per_img.append((iou, g["difficult"][gm]))
            for j, s in enumerate(scores_d):
                dets.append((s, i, j))
        dets.sort(key=lambda x: -x[0])
        out = {}
        for t in thresholds:
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))
            matched = [np.zeros(iou.shape[1], dtype=bool) for iou, _ in per_img]
            for n, (_, i, j) in enumerate(dets):
                iou, diff = per_img[i]
                if iou.shape[1] == 0:
                    fp[n] = 1
                    continue
                row = iou[j]
                k = int(np.argmax(row))
                if row[k] >= t:
                    if diff[k]:
                        continue  # ignored
                    if not matched[i][k]:
                        tp[n] = 1
                        matched[i][k] = True
                    else:
                        fp[n] = 1
                else:
                    fp[n] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            rec = ctp / max(npos, 1)
            prec = ctp / np.maximum(ctp + cfp, 1e-12)
            out[t] = (rec, prec)
        return out, npos

    def compute(self) -> dict:
        ap_all = np.full((self.k, len(IOU_THRESHOLDS)), np.nan)
        ap07 = np.full(self.k, np.nan)
        recall_final = np.full((self.k, len(IOU_THRESHOLDS)), np.nan)
        for ci in range(self.k):
            curves, npos = self._class_curves(ci + 1, IOU_THRESHOLDS)
            if npos == 0:
                continue
            for ti, t in enumerate(IOU_THRESHOLDS):
                rec, prec = curves[t]
                if len(rec) == 0:
                    ap_all[ci, ti], recall_final[ci, ti] = 0.0, 0.0
                    if t == 0.5:
                        ap07[ci] = 0.0
                    continue
                ap_all[ci, ti] = ap_all_points(rec, prec)
                recall_final[ci, ti] = rec[-1]
                if t == 0.5:
                    ap07[ci] = ap_11_point(rec, prec)

        def nm(x):
            return float(np.nanmean(x)) if np.any(~np.isnan(x)) else 0.0
        i50, i75 = int(np.flatnonzero(IOU_THRESHOLDS == 0.5)[0]), int(np.flatnonzero(IOU_THRESHOLDS == 0.75)[0])
        return {
            "map50_voc07": nm(ap07), "map50": nm(ap_all[:, i50]), "map75": nm(ap_all[:, i75]), "map": nm(ap_all),
            "mar100": nm(recall_final),
            "ap50_per_class": {n: (None if np.isnan(ap_all[i, i50]) else float(ap_all[i, i50]))
                               for i, n in enumerate(self.class_names)},
            "n_images": len(self.gts), "n_gt": int(sum(len(g["labels"]) for g in self.gts)),
        }


def error_breakdown(preds, targets, score_thr=0.5, iou_hi=0.5, iou_lo=0.1) -> dict:
    """Hoiem-style error analysis for detections with score >= score_thr (difficult GTs are ignored).

    correct | duplicate | localization (right class, IoU in [lo,hi)) | confusion (other class, IoU >= hi) | background,
    plus 'missed' = non-difficult GTs never matched.
    """
    cnt = dict(correct=0, duplicate=0, localization=0, confusion=0, background=0, missed=0, n_gt=0, n_det=0)
    for p, t in zip(preds, targets, strict=True):
        pb, ps, pl = np.asarray(p["boxes"]).reshape(-1, 4), np.asarray(p["scores"]), np.asarray(p["labels"])
        gb, gl = np.asarray(t["boxes"]).reshape(-1, 4), np.asarray(t["labels"])
        gd = np.asarray(t["difficult"], dtype=bool) if "difficult" in t else np.zeros(len(gl), dtype=bool)
        keep = ps >= score_thr
        order = np.argsort(-ps[keep])
        pb, pl = pb[keep][order], pl[keep][order]
        iou = box_iou_np(pb, gb)
        matched = np.zeros(len(gl), dtype=bool)
        cnt["n_gt"] += int(np.sum(~gd))
        cnt["n_det"] += len(pb)
        for i in range(len(pb)):
            same = np.flatnonzero(gl == pl[i])
            other = np.flatnonzero(gl != pl[i])
            best_same = (same[np.argmax(iou[i, same])], iou[i, same].max()) if len(same) else (None, 0.0)
            best_other = iou[i, other].max() if len(other) else 0.0
            if best_same[1] >= iou_hi:
                k = best_same[0]
                if gd[k]:
                    continue
                if matched[k]:
                    cnt["duplicate"] += 1
                else:
                    matched[k] = True
                    cnt["correct"] += 1
            elif best_same[1] >= iou_lo:
                cnt["localization"] += 1
            elif best_other >= iou_hi:
                cnt["confusion"] += 1
            else:
                cnt["background"] += 1
        cnt["missed"] += int(np.sum(~matched & ~gd))
    n_det = max(cnt["n_det"], 1)
    cnt["precision_at_thr"] = cnt["correct"] / n_det
    cnt["recall_at_thr"] = cnt["correct"] / max(cnt["n_gt"], 1)
    return cnt
