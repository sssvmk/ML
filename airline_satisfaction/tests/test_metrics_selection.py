import numpy as np
from sklearn.metrics import roc_auc_score

from airsat.metrics import choose_threshold, fast_weighted_auc


def test_fast_weighted_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.round(rng.random(2000) * 0.5 + 0.4 * y, 1)          # many ties on purpose
    uniq, inv = np.unique(p, return_inverse=True)
    assert abs(fast_weighted_auc(y, p, inv, len(uniq), np.ones(2000)) - roc_auc_score(y, p)) < 1e-12


def test_weighted_auc_equals_resampled_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) + 0.3 * y
    idx = rng.integers(0, 500, 500)
    w = np.bincount(idx, minlength=500).astype(float)
    uniq, inv = np.unique(p, return_inverse=True)
    assert abs(fast_weighted_auc(y, p, inv, len(uniq), w) - roc_auc_score(y[idx], p[idx])) < 1e-9


def test_threshold_chosen_on_given_data_improves_f1():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 3000), 0.01, 0.99)
    t = choose_threshold(y, p, "max_f1")
    from sklearn.metrics import f1_score
    assert f1_score(y, p >= t) >= f1_score(y, p >= 0.5)
