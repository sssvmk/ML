"""Plot helpers: every figure is saved to disk (headless-safe) for human inspection."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def plot_roc_pr(y, p, path, title="") -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, p, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y, p, ax=ax[1])
    ax[0].set_title(f"ROC {title}")
    ax[1].set_title(f"Precision-Recall {title}")
    return _save(fig, path)


def plot_calibration(y, p, path, title="") -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    try:
        frac, mean = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax[0].plot(mean, frac, "o-", label="model")
    except ValueError:
        pass
    ax[0].plot([0, 1], [0, 1], "k--", label="perfect")
    ax[0].set(xlabel="mean predicted probability", ylabel="observed frequency", title=f"Reliability {title}")
    ax[0].legend()
    ax[1].hist(p, bins=40)
    ax[1].set(title="Predicted probability distribution", xlabel="p(satisfied)")
    return _save(fig, path)


def plot_confusion(y, pred, path, title="") -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix {title}")
    return _save(fig, path)


def plot_curve(df: pd.DataFrame, x: str, cols: list[str], path, title="", ylabel="", xlog=False, hline=None) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cols:
        if c in df:
            ax.plot(df[x], df[c], marker="o" if len(df) < 30 else None, label=c)
    if hline is not None:
        ax.axhline(hline, color="r", ls="--", label="target")
    if xlog:
        ax.set_xscale("log")
    ax.set(xlabel=x, ylabel=ylabel, title=title)
    ax.legend()
    return _save(fig, path)


def plot_bar(series: pd.Series, path, title="", xlabel="") -> str:
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(series))))
    series.sort_values().plot.barh(ax=ax)
    ax.set(title=title, xlabel=xlabel)
    return _save(fig, path)


def plot_optuna_history(trials: pd.DataFrame, path, title="") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(trials):
        ax.scatter(trials["number"], trials["value"], s=14, label="trial")
        ax.plot(trials["number"], trials["value"].cummax(), "r-", label="best so far")
    ax.set(xlabel="trial", ylabel="validation primary metric", title=title)
    ax.legend()
    return _save(fig, path)
