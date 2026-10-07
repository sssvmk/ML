"""Metrics, thresholds and calibration helpers shared by every algorithm module."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

EPS = 1e-12


def expected_calibration_error(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def choose_threshold(y, p, rule: str = "max_f1", fixed: float = 0.5) -> float:
    """Pick the decision threshold on VALIDATION data only."""
    if rule == "fixed":
        return float(fixed)
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    if len(grid) < 3:
        return float(fixed)
    fn = f1_score if rule == "max_f1" else accuracy_score
    scores = [fn(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if pred.min() != pred.max() else 0.0,
        "threshold": float(threshold),
    }


def primary_value(metric: str, y, p, threshold: float = 0.5) -> float:
    """Higher is better for every returned value (log loss is negated)."""
    m = compute_metrics(y, p, threshold)
    return -m["log_loss"] if metric == "neg_log_loss" else m[metric]


def fast_weighted_auc(y: np.ndarray, p: np.ndarray, inv: np.ndarray, n_groups: int, w: np.ndarray) -> float:
    """AUC with integer bootstrap weights in O(n); `inv` maps each row to its tie-group of sorted unique scores."""
    wp = np.bincount(inv, weights=w * y, minlength=n_groups)
    wn = np.bincount(inv, weights=w * (1 - y), minlength=n_groups)
    tp, tn = wp.sum(), wn.sum()
    if tp == 0 or tn == 0:
        return float("nan")
    cum_neg_before = np.cumsum(wn) - wn
    return float((wp * (cum_neg_before + 0.5 * wn)).sum() / (tp * tn))


def slice_metrics(X_df, y, p, threshold: float, slice_cols: list[str], min_rows: int = 100) -> dict:
    """Accuracy / AUC / positive rate per level of each slice column, plus the accuracy gap between levels."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    out = {}
    for col in slice_cols:
        if col not in X_df.columns:
            continue
        levels = {}
        for lv, idx in X_df.reset_index(drop=True).groupby(col, observed=True).indices.items():
            if len(idx) < min_rows:
                continue
            yy = y[idx]
            levels[str(lv)] = {"n": int(len(idx)), "accuracy": float((pred[idx] == yy).mean()),
                               "positive_rate": float(yy.mean()),
                               "auc": float(roc_auc_score(yy, p[idx])) if len(np.unique(yy)) == 2 else None}
        if levels:
            accs = [v["accuracy"] for v in levels.values()]
            out[col] = {"levels": levels, "accuracy_gap": float(max(accs) - min(accs))}
    return out
