"""
hd_lib.py - shared glue for the Chapter 18 (p >> N) study. All modelling is done by scikit-learn (+ cvxpy for the fused lasso); this file only holds
  * preprocessing transformers that live INSIDE every CV fold (variance filter, winsorising, scaling, within-class scaling, hierarchical gene ordering)
  * the repeated-CV tuning with a one-SE rule that uses the Nadeau-Bengio corrected standard error (folds overlap), complexity-ordered
  * metrics (regression / classification), per-sample losses, plots, logging, the Progress helper and the multiple-testing helpers (Holm, McNemar, paired tests)
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))   # Windows: stops joblib/loky from calling the removed `wmic` tool
try:                                                         # loky asks `wmic` for the physical core count on Windows (removed from Windows 11): answer it ourselves, silently
    from joblib.externals.loky.backend import context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda _n=(_os.cpu_count() or 1): (_n, None)
except Exception:
    pass

import json
import logging
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Cfg:
    seed: int = 0
    cv_folds: int = 5
    cv_repeats: int = 3
    n_jobs: int = 1
    alpha: float = 0.05            # significance level of the comparison tests
    stability_runs: int = 50       # sub-samples of the selection-stability analysis
    winsorize: str = "auto"        # auto | on | off  (auto: from the EDA)


class Progress:
    def __init__(self, name, cb=None, total=5):
        self.name, self.cb, self.total, self.done = name, cb, total, 0

    def set_total(self, total):
        self.total = max(int(total), 1)

    def tick(self, msg="", n=1):
        self.done = min(self.done + n, self.total)
        if self.cb:
            self.cb(self.name, self.done, self.total, msg)


def get_logger(path, name):
    lg = logging.getLogger(f"{name}:{path}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    h = logging.FileHandler(path, mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    return lg


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# ---------------------------------------------------------------------------------------------------- transformers (all fitted inside the CV folds)
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every gene at its q / 1-q quantiles of the TRAINING fold (EDA-driven; protects the scale estimates from extreme expression values)."""
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.quantile(X, self.q, axis=0), np.quantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


class WithinClassScaler(BaseEstimator, TransformerMixin):
    """Centre by the overall mean and divide each gene by its POOLED WITHIN-CLASS standard deviation (ML, divisor n): LDA with an identity covariance on these features is diagonal LDA (18.2)."""
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.mean_ = X.mean(0)
        resid = X.copy()
        for c in np.unique(y):
            resid[y == c] -= X[y == c].mean(0)
        self.scale_ = np.sqrt((resid ** 2).sum(0) / len(y))                      # maximum-likelihood pooled sd = scikit-learn's LDA covariance normalisation (divisor n), so identity covariance <=> trace/p = 1
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_


class GeneOrderer(BaseEstimator, TransformerMixin):
    """Order the genes by the leaf order of an average-linkage hierarchical clustering on 1 - correlation (fitted on the training fold only; label-free). Neighbouring genes are then similar,
    which is what the fused-lasso difference penalty assumes."""
    def fit(self, X, y=None):
        from scipy.cluster.hierarchy import linkage, leaves_list
        Z = linkage(np.asarray(X, float).T, method="average", metric="correlation")
        self.order_ = leaves_list(Z)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.order_]


def prefix_steps(winsorize, scale=True):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    steps = [("var", VarianceThreshold(1e-10))]
    if winsorize:
        steps.append(("win", Winsorizer(0.01)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_pipeline(steps, out_dir):
    from sklearn.pipeline import Pipeline
    from joblib import Memory
    return Pipeline(steps, memory=Memory(location=str(Path(out_dir) / "cache"), verbose=0))


def gene_mask(pipe, p):
    """Boolean mask over the ORIGINAL genes of the genes that survive the variance filter and every selection step of the fitted pipeline."""
    mask = np.ones(p, bool)
    idx = np.arange(p)
    for _, st in pipe.steps[:-1]:
        if hasattr(st, "get_support"):
            s = st.get_support()
            idx = idx[s]
    mask[:] = False
    mask[idx] = True
    return mask


# ---------------------------------------------------------------------------------------------------- tuning
def loss_from_scores(task, S):
    return -S if task == "regression" else 1.0 - S


def tune(est, grid, X, y, task, cfg, complexity, scoring):
    """Repeated (stratified) K-fold CV over a grid; one-SE rule with the Nadeau-Bengio corrected SE, picking the least complex candidate within one SE of the best."""
    from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
    cv = (RepeatedStratifiedKFold if task == "classification" else RepeatedKFold)(n_splits=cfg.cv_folds, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
    gs = GridSearchCV(est, grid, cv=cv, scoring=scoring, refit=False, error_score="raise", n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    r = pd.DataFrame(gs.cv_results_)
    S = r[[c for c in r.columns if c.startswith("split") and c.endswith("_test_score")]].values
    L = loss_from_scores(task, S)
    J, ntr_frac = L.shape[1], 1.0 / (cfg.cv_folds - 1)
    r["cv_loss"] = L.mean(1)
    r["cv_se"] = np.sqrt((1.0 / J + ntr_frac) * L.var(1, ddof=1))                       # Nadeau-Bengio: folds overlap, the plain SE is far too small
    r["complexity"] = [complexity(p) for p in r["params"]]
    best = int(np.argmin(r["cv_loss"].values))
    cand = np.where(r["cv_loss"].values <= r["cv_loss"].values[best] + r["cv_se"].values[best])[0]
    sel = int(min(cand, key=lambda i: (r["complexity"].values[i], r["cv_loss"].values[i])))
    r["selected_one_se"], r["cv_minimum"] = False, False
    r.loc[sel, "selected_one_se"], r.loc[best, "cv_minimum"] = True, True
    return r, dict(r.loc[sel, "params"]), dict(r.loc[best, "params"])


# ---------------------------------------------------------------------------------------------------- metrics
def reg_metrics(y, p, y_ref_mean):
    e2 = (y - p) ** 2
    mse_mean = float(np.mean((y - y_ref_mean) ** 2))
    return {"mse": float(e2.mean()), "se": float(e2.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else float("nan"), "rmse": float(np.sqrt(e2.mean())), "mae": float(np.mean(np.abs(y - p))),
            "r2_vs_train_mean": float(1 - e2.mean() / mse_mean) if mse_mean > 0 else float("nan"), "n": int(len(y))}


def ovr_auc(y, score, classes):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score((y == c).astype(int), score[:, k]) for k, c in enumerate(classes)]))


def clf_metrics(y, pred, proba, score, classes):
    err = float(np.mean(pred != y))
    out = {"error": err, "se_error": float(np.sqrt(err * (1 - err) / len(y))), "n": int(len(y)), "n_errors": int(np.sum(pred != y))}
    if proba is not None:
        pc = np.clip(proba, 1e-6, 1 - 1e-6)
        pc = pc / pc.sum(1, keepdims=True)
        ll = -np.log(pc[np.arange(len(y)), np.searchsorted(classes, y)])
        out.update(log_loss=float(ll.mean()), se_log_loss=float(ll.std(ddof=1) / np.sqrt(len(y))))
    sc = proba if proba is not None else score
    if sc is not None:
        try:
            out["macro_ovr_auc"] = ovr_auc(y, sc, classes)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------------------------------- plots
def plot_cv(r, out, title, xparam=None):
    nums = [k for k in r["params"].iloc[0] if isinstance(r["params"].iloc[0][k], (int, float, np.integer, np.floating)) and not isinstance(r["params"].iloc[0][k], bool)]
    P = pd.DataFrame(list(r["params"]))
    if xparam is None and nums:
        xparam = max(nums, key=lambda k: P[k].nunique())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if xparam is None or P[xparam].nunique() < 2:
        ax.bar(range(len(r)), r["cv_loss"]); ax.set_xticks(range(len(r)))
    else:
        others = [k for k in P.columns if k != xparam and P[k].nunique() > 1]
        grp = P[others].apply(lambda r: " ".join(str(v) for v in r), axis=1) if others else pd.Series([""] * len(P))      # robust to missing parameters (e.g. gamma for the linear kernel)
        for g in grp.unique():
            m = (grp == g).values
            o = np.argsort(P.loc[m, xparam].values.astype(float))
            ax.errorbar(P.loc[m, xparam].values.astype(float)[o], r.loc[m, "cv_loss"].values[o], yerr=r.loc[m, "cv_se"].values[o], marker="o", ms=3, capsize=2, label=g or None, lw=1)
        if P[xparam].astype(float).min() > 0 and P[xparam].astype(float).max() / P[xparam].astype(float).min() > 50:
            ax.set_xscale("log")
        if others and len(grp.unique()) <= 12:
            ax.legend(fontsize=7)
    s = r.index[r["selected_one_se"]][0]
    ax.axhline(r.loc[s, "cv_loss"], color="tab:red", ls=":", lw=1)
    ax.set_xlabel(xparam or "configuration"); ax.set_ylabel("CV loss (MSE or error)"); ax.set_title(title + "  (red: one-SE choice)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_heatmap(r, out, title):
    P = pd.DataFrame(list(r["params"]))
    nums = [k for k in P.columns if P[k].nunique() > 1 and pd.api.types.is_numeric_dtype(P[k])]
    if len(nums) < 2:
        return
    a, b = nums[:2]
    T = r.assign(**{a: P[a].values, b: P[b].values}).groupby([a, b])["cv_loss"].min().unstack(b)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(T.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(T.shape[1])); ax.set_xticklabels([f"{v:g}" for v in T.columns], rotation=45); ax.set_yticks(range(T.shape[0])); ax.set_yticklabels([f"{v:g}" for v in T.index])
    ax.set_xlabel(b); ax.set_ylabel(a); ax.set_title(title); fig.colorbar(im, label="CV loss"); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_confusion(y, pred, classes, out, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return cm


def stability_selection(make_fitted, X, y, support_fn, runs, seed, frac=0.8, stratify=False):
    """Selection frequency of every gene over sub-samples (80 % without replacement) refitted at the chosen hyper-parameters."""
    rng = np.random.RandomState(seed)
    n, p = X.shape
    freq = np.zeros(p)
    done = 0
    for _ in range(runs):
        if stratify:
            idx = np.concatenate([rng.choice(np.where(y == c)[0], max(2, int(round(frac * (y == c).sum()))), replace=False) for c in np.unique(y)])
        else:
            idx = rng.choice(n, int(round(frac * n)), replace=False)
        try:
            freq += support_fn(make_fitted(X[idx], y[idx]), p)
            done += 1
        except Exception:
            continue
    return freq / max(done, 1)


# ---------------------------------------------------------------------------------------------------- tests
def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mcnemar_exact(correct_a, correct_b):
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_t(loss_a, loss_b):
    from scipy.stats import ttest_rel
    d = np.asarray(loss_a) - np.asarray(loss_b)
    return 1.0 if np.allclose(d, 0) else float(ttest_rel(loss_a, loss_b).pvalue)


def dm_test(loss_a, loss_b):
    """Diebold-Mariano for 1-step forecasts with the Harvey-Leybourne-Newbold small-sample correction (t with n-1 df). Rows are exchangeable here, so no autocorrelation term."""
    from scipy.stats import t as tdist
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if np.allclose(d, 0) or n < 3:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * tdist.sf(abs(stat), n - 1))
