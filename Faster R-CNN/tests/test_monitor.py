import json

import numpy as np

from src.monitor import drift_report, psi
from src.utils import ROOT, load_yaml

REF = {"channel_mean": [0.5, 0.5, 0.5], "channel_mean_std": [0.1, 0.1, 0.1], "detections_per_image": 2.0,
       "score_hist": [0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], "score_bins": list(np.linspace(0, 1, 11)),
       "predicted_class_counts": {"1": 50, "2": 50}, "score_threshold": 0.5}


def _res(scores, label=1):
    return {"detections": [{"score": s, "label_id": label} for s in scores]}


def test_psi_zero_for_identical_and_positive_for_shift():
    a = [0.1] * 10
    assert psi(a, a) < 1e-9 and psi(a, [0.0] * 9 + [1.0]) > 0.25


def test_drift_report_quiet_when_unchanged_and_flags_when_shifted(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(REF))
    cfg = load_yaml(ROOT / "config" / "serve.yaml")
    rng = np.random.default_rng(0)
    same_means = 0.5 + 0.1 * rng.standard_normal((200, 3))
    rng_scores = [_res([0.95, 0.9, 0.3, 0.1]) for _ in range(200)]
    rep = drift_report(p, same_means, rng_scores, cfg)
    assert rep["detections_per_image"] == 2.0 and not any("input" in f or "per image" in f for f in rep["flags"])
    dark = np.full((200, 3), 0.05)
    shifted = [_res([0.99] * 12, label=7) for _ in range(200)]
    bad = drift_report(p, dark, shifted, cfg)
    assert "input channel means shifted" in bad["flags"] and "detections per image changed sharply" in bad["flags"]
    assert any("class mix" in f for f in bad["flags"])
