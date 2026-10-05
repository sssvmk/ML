"""Input and prediction drift against the reference statistics saved at training time."""
from __future__ import annotations

import json

import numpy as np


def psi(ref_frac, cur_frac, eps=1e-4) -> float:
    r, c = np.clip(np.asarray(ref_frac, float), eps, None), np.clip(np.asarray(cur_frac, float), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference_path, images_uint8: np.ndarray, probs: np.ndarray, serve_cfg: dict) -> dict:
    ref = json.load(open(reference_path))
    mon = serve_cfg["monitoring"]
    x = images_uint8.astype(np.float32) / 255.0
    ch_mean = x.mean((0, 1, 2))
    se = np.array(ref["channel_std"]) / np.sqrt(len(x))
    z = ((ch_mean - np.array(ref["channel_mean"])) / se).tolist()
    conf_hist = np.histogram(probs.max(1), bins=ref["confidence_bins"])[0] / len(probs)
    p = psi(ref["confidence_hist"], conf_hist)
    flags = []
    if max(abs(v) for v in z) > mon["channel_mean_z_alert"]:
        flags.append("input channel means shifted")
    if p > mon["psi_alert"]:
        flags.append("prediction-confidence distribution shifted (ALERT)")
    elif p > mon["psi_warn"]:
        flags.append("prediction-confidence distribution shifted (warn)")
    return {"n_images": int(len(x)), "channel_mean_z": z, "confidence_psi": p, "flags": flags}
