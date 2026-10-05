"""Input and prediction drift against the reference statistics saved at training time."""
from __future__ import annotations

import json

import numpy as np


def psi(ref_frac, cur_frac, eps=1e-4) -> float:
    r, c = np.clip(np.asarray(ref_frac, float), eps, None), np.clip(np.asarray(cur_frac, float), eps, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference_path, channel_means: np.ndarray, results: list[dict], serve_cfg: dict) -> dict:
    """channel_means: [N,3] per-image mean RGB in [0,1]; results: Predictor.predict outputs for the same images."""
    ref = json.load(open(reference_path))
    mon = serve_cfg["monitoring"]
    n = len(channel_means)
    se = np.maximum(np.array(ref["channel_mean_std"]), 1e-6) / np.sqrt(n)
    z = ((channel_means.mean(0) - np.array(ref["channel_mean"])) / se).tolist()
    thr = ref["score_threshold"]
    scores = [d["score"] for r in results for d in r["detections"]]
    hist = np.histogram(scores, bins=ref["score_bins"])[0] / max(len(scores), 1)
    p_score = psi(ref["score_hist"], hist) if scores else float("nan")
    per_img = float(np.mean([sum(d["score"] >= thr for d in r["detections"]) for r in results]))
    ratio = (per_img + 1e-6) / (ref["detections_per_image"] + 1e-6)
    ref_cls = ref["predicted_class_counts"]
    classes = sorted(set(ref_cls) | {str(d["label_id"]) for r in results for d in r["detections"] if d["score"] >= thr})
    cur_cls = {c: 0 for c in classes}
    for r in results:
        for d in r["detections"]:
            if d["score"] >= thr:
                cur_cls[str(d["label_id"])] += 1
    p_cls = psi([ref_cls.get(c, 0) + 1 for c in classes], [cur_cls[c] + 1 for c in classes])
    flags = []
    if max(abs(v) for v in z) > mon["channel_mean_z_alert"]:
        flags.append("input channel means shifted")
    if not np.isnan(p_score) and p_score > mon["psi_alert"]:
        flags.append("detection-score distribution shifted (ALERT)")
    elif not np.isnan(p_score) and p_score > mon["psi_warn"]:
        flags.append("detection-score distribution shifted (warn)")
    if ratio > mon["detections_ratio_alert"] or ratio < 1 / mon["detections_ratio_alert"]:
        flags.append("detections per image changed sharply")
    if p_cls > mon["psi_alert"]:
        flags.append("predicted-class mix shifted (ALERT)")
    return {"n_images": n, "channel_mean_z": z, "score_psi": p_score, "detections_per_image": per_img,
            "detections_ratio_vs_reference": ratio, "class_mix_psi": p_cls, "flags": flags}
