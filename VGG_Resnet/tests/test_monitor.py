import json

import numpy as np

from src.monitor import drift_report, psi
from src.utils import ROOT, load_yaml


def test_psi_zero_for_identical_and_positive_for_shift():
    a = [0.1] * 10
    assert psi(a, a) == 0
    assert psi(a, [0.0] * 9 + [1.0]) > 0.25


def test_drift_report_flags_shifted_images(tmp_path):
    ref = {"channel_mean": [0.5] * 3, "channel_std": [0.25] * 3, "confidence_hist": [0.0] * 9 + [1.0],
           "confidence_bins": list(np.linspace(0, 1, 11))}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    cfg = load_yaml(ROOT / "config" / "serve.yaml")
    dark = np.zeros((200, 32, 32, 3), np.uint8)
    probs = np.full((200, 100), 0.01)
    rep = drift_report(p, dark, probs, cfg)
    assert rep["flags"]
