"""Unit tests for metrics.py (numpy + scikit-learn only)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics  # noqa: E402


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ------------------------------- regression ------------------------------- #
def test_mse_and_se_hand_computed():
    # errors [1,-1,2,0] -> squared [1,1,4,0]: mean 1.5, sample var 3, SE = sqrt(3)/2
    m = metrics.regression_metrics(np.array([1.0, -1.0, 2.0, 0.0]), np.zeros(4))
    assert m["mse"] == pytest.approx(1.5)
    assert m["mse_se"] == pytest.approx(math.sqrt(3) / 2)
    assert m["rmse"] == pytest.approx(math.sqrt(1.5))
    assert m["rmse_se"] == pytest.approx(m["mse_se"] / (2 * math.sqrt(1.5)))
    assert m["mae"] == pytest.approx(1.0)
    assert m["mse_ci95_low"] == pytest.approx(1.5 - 1.96 * m["mse_se"])


def test_mse_se_matches_true_sampling_spread():
    """Mean reported SE should match the empirical std of MSE over repeated test sets."""
    rng = np.random.default_rng(0)
    n, reps = 400, 1500
    mses, ses = [], []
    for _ in range(reps):
        e = rng.normal(size=n)  # errors ~ N(0,1) -> true SD of MSE is sqrt(2/n)
        m = metrics.regression_metrics(e, np.zeros(n))
        mses.append(m["mse"])
        ses.append(m["mse_se"])
    assert np.mean(ses) == pytest.approx(np.std(mses, ddof=1), rel=0.06)
    assert np.mean(ses) == pytest.approx(math.sqrt(2 / n), rel=0.06)
    # ~95% of intervals mse +- 1.96 SE should contain the true MSE (= 1)
    cover = np.mean([abs(m - 1.0) <= 1.96 * s for m, s in zip(mses, ses)])
    assert 0.92 <= cover <= 0.98


def test_r2_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.regression_metrics(y, y)["r2"] == pytest.approx(1.0)
    assert metrics.regression_metrics(np.full(4, y.mean()), y)["r2"] == pytest.approx(0.0)


def test_paired_difference_sign_and_se():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    good = y + 0.1 * rng.normal(size=2000)
    bad = y + 1.0 * rng.normal(size=2000)
    d, se = metrics.paired_mse_difference(good, bad, y)
    assert d < 0 and d + 1.96 * se < 0  # clearly better
    d0, _ = metrics.paired_mse_difference(good, good, y)
    assert d0 == 0.0


# ------------------------------ classification ----------------------------- #
def test_auc_perfect_and_chance():
    y = np.repeat(np.arange(3), 50)
    perfect = np.eye(3)[y]
    assert metrics.macro_auc(perfect, y)[0] == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    yy = rng.integers(0, 4, 8000)
    assert metrics.macro_auc(softmax(rng.normal(size=(8000, 4))), yy)[0] == pytest.approx(0.5, abs=0.03)


def test_auc_matches_sklearn_multiclass_ovr():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 1000)
    p = softmax(rng.normal(size=(1000, 5)) + 2.0 * np.eye(5)[y])
    ref = roc_auc_score(y, p, multi_class="ovr", average="macro")
    assert metrics.macro_auc(p, y)[0] == pytest.approx(ref, abs=1e-12)


def test_binary_auc_matches_sklearn():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    p1 = np.clip(0.5 * y + 0.5 * rng.random(500), 0.001, 0.999)
    probs = np.stack([1 - p1, p1], axis=1)
    assert metrics.macro_auc(probs, y)[0] == pytest.approx(roc_auc_score(y, p1), abs=1e-12)


def test_absent_class_is_skipped_not_crash():
    y = np.array([0, 0, 1, 1])  # class 2 never occurs
    probs = np.array([[.8, .1, .1], [.7, .2, .1], [.1, .8, .1], [.2, .7, .1]])
    auc, per = metrics.macro_auc(probs, y)
    assert set(per) == {0, 1} and auc == pytest.approx(1.0)


def test_classification_metrics_content_and_bootstrap():
    rng = np.random.default_rng(5)
    n = 600
    y = rng.integers(0, 3, n)
    p = softmax(rng.normal(size=(n, 3)) + 1.5 * np.eye(3)[y])
    m = metrics.classification_metrics(p, y, n_boot=150, seed=7)
    assert m["auc_ci95_low"] <= m["auc_macro_ovr"] <= m["auc_ci95_high"]
    assert m["auc_se"] > 0
    acc = (p.argmax(1) == y).mean()
    assert m["accuracy"] == pytest.approx(acc)
    assert m["accuracy_se"] == pytest.approx(math.sqrt(acc * (1 - acc) / (n - 1)), rel=1e-6)
    assert m["log_loss"] > 0 and 0 <= m["macro_f1"] <= 1
    assert all(f"auc_class_{c}" in m for c in range(3))
    # deterministic given the seed
    assert metrics.classification_metrics(p, y, n_boot=150, seed=7) == m


def test_bootstrap_se_is_sane_against_resampling_of_data():
    """Bootstrap SE of AUC should be close to the spread of AUC over fresh draws."""
    rng = np.random.default_rng(6)
    aucs = []
    for _ in range(200):
        y = rng.integers(0, 2, 300)
        p1 = 1 / (1 + np.exp(-(1.0 * y - 0.5 + rng.normal(size=300))))
        aucs.append(roc_auc_score(y, p1))
    y = rng.integers(0, 2, 300)
    p1 = 1 / (1 + np.exp(-(1.0 * y - 0.5 + rng.normal(size=300))))
    m = metrics.classification_metrics(np.stack([1 - p1, p1], 1), y, n_boot=300, seed=0)
    assert m["auc_se"] == pytest.approx(np.std(aucs, ddof=1), rel=0.35)
