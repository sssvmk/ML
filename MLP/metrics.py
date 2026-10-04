"""
Evaluation metrics with uncertainty (numpy + scikit-learn only; no torch / MLflow).

classification (headline: ROC-AUC)
    auc_macro_ovr          macro-averaged one-vs-rest ROC-AUC (binary: the usual AUC)
    auc_se, auc_ci95_*     bootstrap standard error / 95% percentile interval of that AUC
    accuracy (+ _se)       share correct; SE = std/sqrt(n)
    log_loss (+ _se)       mean negative log-likelihood; SE = std/sqrt(n)
    macro_f1               unweighted mean of per-class F1
    auc_class_<i>          per-class one-vs-rest AUC

regression (headline: MSE +- SE)
    mse, mse_se            mean of squared errors; SE = std(e^2)/sqrt(n)
    mse_ci95_low/high      mse -/+ 1.96 SE
    rmse, rmse_se          sqrt(mse); SE by the delta method: mse_se / (2 rmse)
    mae, mae_se            mean absolute error; SE = std(|e|)/sqrt(n)
    r2                     1 - SSE/SST

The standard errors describe uncertainty caused by the FINITE SIZE of the evaluated set
(sampling uncertainty). They do not include run-to-run training variance (different seeds).
"""
import math

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _se(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    return float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def macro_auc(probs, y):
    """Macro one-vs-rest ROC-AUC. Returns (macro_auc, {class: auc}).

    Classes that are absent (or are the only class present) in `y` are skipped, since
    their AUC is undefined; the macro average uses the remaining classes.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    per_class = {}
    for c in range(probs.shape[1]):
        pos = y == c
        if 0 < pos.sum() < len(y):
            per_class[c] = float(roc_auc_score(pos.astype(int), probs[:, c]))
    if not per_class:
        return float("nan"), per_class
    return float(np.mean(list(per_class.values()))), per_class


def classification_metrics(probs, y, n_boot=200, seed=0):
    """Metrics for predicted class probabilities `probs` [n, k] against integer labels `y`."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n, k = probs.shape
    pred = probs.argmax(axis=1)

    correct = (pred == y).astype(np.float64)
    nll = -np.log(np.clip(probs[np.arange(n), y], 1e-12, None))
    auc, per_class = macro_auc(probs, y)

    boots = []
    if n_boot and n_boot > 1:
        rng = np.random.default_rng(seed)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            a, _ = macro_auc(probs[idx], y[idx])
            if not math.isnan(a):
                boots.append(a)
    if len(boots) > 1:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        auc_se = float(np.std(boots, ddof=1))
    else:
        lo = hi = auc_se = float("nan")

    out = {
        "auc_macro_ovr": auc, "auc_se": auc_se,
        "auc_ci95_low": float(lo), "auc_ci95_high": float(hi),
        "accuracy": float(correct.mean()), "accuracy_se": _se(correct),
        "log_loss": float(nll.mean()), "log_loss_se": _se(nll),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0)),
    }
    out.update({f"auc_class_{c}": a for c, a in per_class.items()})
    return out


def regression_metrics(pred, y):
    """Metrics for predictions `pred` against targets `y`, both 1-D, in original units."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    err = pred - y
    sq = err ** 2
    mse = float(sq.mean())
    mse_se = _se(sq)
    rmse = math.sqrt(mse)
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "mse": mse, "mse_se": mse_se,
        "mse_ci95_low": mse - 1.96 * mse_se, "mse_ci95_high": mse + 1.96 * mse_se,
        "rmse": rmse, "rmse_se": (mse_se / (2 * rmse)) if rmse > 0 else 0.0,
        "mae": float(np.abs(err).mean()), "mae_se": _se(np.abs(err)),
        "r2": (1.0 - float(sq.sum()) / sst) if sst > 0 else float("nan"),
    }


def paired_mse_difference(pred, baseline_pred, y):
    """MSE(model) - MSE(baseline) on the same rows, with a paired standard error.

    Negative = the model is better. Pairing removes the shared difficulty of each row, so
    this is a much sharper comparison than eyeballing two separate MSE +- SE values.
    Returns (difference, se).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    d = (np.asarray(pred, dtype=np.float64).ravel() - y) ** 2 \
        - (np.asarray(baseline_pred, dtype=np.float64).ravel() - y) ** 2
    return float(d.mean()), _se(d)
