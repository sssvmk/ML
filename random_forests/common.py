"""
common.py - shared infrastructure for the ESLII Ch. 6 kernel-method study (regression on Zillow, classification on Santander).

Protocol (identical for every method of a task)
  * regression  : temporal split train <= 2017-02-28 | val Mar-Jul 2017 | test >= Aug 2017; fit target = logerror clipped at the train 1st/99th percentile; all metrics on RAW logerror
  * classification : stratified 80/10/10 split on Target; probabilities clipped to [1e-4, 1-1e-4]; threshold tuned on VALIDATION (Youden's J)
  * tuning      : stratified (classification) / time-ordered systematic (regression) SUBSAMPLE of `tune_rows` training rows; 10-fold CV on that subsample
                  (regression: contiguous time blocks; classification: stratified folds); one-SE rule (paired SE of fold differences by default)
  * final model : memory = ALL training rows at the CV-chosen setting (span is a fraction, so it transfers); evaluated on the FULL validation and test sets
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))   # Windows: stops joblib/loky from calling the removed `wmic` tool
try:                                                         # loky asks `wmic` for the physical core count on Windows (removed from Windows 11): answer it ourselves, silently
    from joblib.externals.loky.backend import context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda _n=(_os.cpu_count() or 1): (_n, None)
except Exception:
    pass

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as t_dist

ROOT = Path(__file__).resolve().parent
PROB_EPS = 1e-4


@dataclass
class RunConfig:
    seed: int = 42
    cv_folds: int = 10
    tune_rows: int = 30000
    one_se_mode: str = "paired"          # "paired" | "plain"
    rank_trees: int = 60                 # ExtraTrees used in the variable ranking
    rank_rows: int = 20000
    hme_restarts: int = 2
    n_jobs: int = 1                      # threads for LightGBM / XGBoost (set per worker by run_all)
    n_configs: int = 16                  # random-search budget: LightGBM / XGBoost
    n_configs_rf: int = 12               # random-search budget per forest entry
    rf_trees_cv: int = 200               # trees per forest inside CV (the book: ~200 trees are enough)
    rf_trees_final: int = 300            # trees of the final forest (all training rows)
    prox_rows: int = 3000                # rows of the proximity plot
    prox_trees: int = 200                # trees used for the proximity matrix
    perm_rows: int = 2000                # validation rows used by the permutation importance
    perm_repeats: int = 2                # shuffles per variable in the permutation importance
    n_configs_nn: int = 16               # random-search budget for the iterative prototype methods (LVQ, Gaussian mixtures, DANN); K-means / k-NN grids are searched fully
    n_configs_svm: int = 8               # random-search budget for the SVM-family methods (kernel SVMs are expensive)
    svm_final_rows: int = 50000          # cap on the rows of the FINAL fit of kernel SVM / SVR / FDA (linear SVC / SVR, PDA, MDA use all training rows)
    n_configs_sk: int = 6                # random-search budget: scikit-learn boosting (exact greedy trees, slower)
    sk_final_rows: int = 60000           # cap on the rows of the FINAL fit of the exact-greedy scikit-learn methods (runtime guard; LightGBM / XGBoost use all rows)
    leak_auc: float = 0.9                # Santander: exclude any single feature whose univariate train AUC is >= this (or <= 1 - this): target leakage; 0 = off
    max_iter: int = 0                    # 0 = no cap; otherwise only boosting iterations M <= max_iter are searched (runtime control)
    diag_rows: int = 8000                # rows used by the diagnostic refits (train-vs-test curves, shrinkage curves, AdaBoost curves)
    alpha: float = 0.05

    def to_json(self, p):
        Path(p).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, p):
        return cls(**json.loads(Path(p).read_text()))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def get_logger(out_dir, name, console=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"kern.{name}.{out_dir}")
    lg.setLevel(logging.INFO); lg.propagate = False
    for h in list(lg.handlers):
        lg.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w"); fh.setFormatter(fmt); lg.addHandler(fh)
    if console:
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
    return lg


class Progress:
    def __init__(self, name, cb=None, total=1):
        self.name, self.cb, self.total, self.done = name, cb, total, 0

    def set_total(self, total):
        self.total = max(int(total), 1)

    def tick(self, msg="", n=1):
        self.done = min(self.done + n, self.total)
        if self.cb:
            self.cb(self.name, self.done, self.total, msg)


def print_progress(name, done, total, msg):
    print(f"[{name}] {done}/{total} {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------
class Prepared:
    """Arrays written by prep_reg.py / prep_clf.py: X_{train,val,test} (float32), y_raw_*, y_fit_train (regression), feature names."""

    def __init__(self, d):
        d = self.dir = Path(d)
        self.meta = json.loads((d / "prepared_meta.json").read_text())
        self.task = self.meta["task"]
        self.feature_names = self.meta["feature_names"]
        self.p = len(self.feature_names)
        self._X = {s: np.load(d / f"X_{s}.npy", mmap_mode="r") for s in ("train", "val", "test")}
        self._y = {s: np.load(d / f"y_raw_{s}.npy") for s in ("train", "val", "test")}
        if self.task == "regression":
            self.y_fit_train = np.load(d / "y_fit_train.npy")
            self.dates_test = np.load(d / "dates_test.npy")
        self._ranks = None

    def X(self, split):
        return self._X[split]

    def y(self, split):
        return self._y[split]

    def rankings(self):
        if self._ranks is None:
            r = np.load(self.dir / "rankings.npz")
            self._ranks = {"folds": r["folds"], "sub": r["sub"]}
        return self._ranks


def tuning_subsample(prep, cfg):
    y = prep.y("train")
    n = len(y)
    if n <= cfg.tune_rows:
        return np.arange(n)
    if prep.task == "regression":                       # systematic sample keeps the time order
        return np.unique(np.linspace(0, n - 1, cfg.tune_rows).astype(int))
    rng = np.random.RandomState(cfg.seed)
    parts = []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        parts.append(rng.choice(idx, int(round(cfg.tune_rows * len(idx) / n)), replace=False))
    return np.sort(np.concatenate(parts))


def make_folds(y, k, task, seed):
    n = len(y)
    if task == "regression":
        e = np.linspace(0, n, k + 1).astype(int)
        return [np.arange(e[i], e[i + 1]) for i in range(k)]
    rng = np.random.RandomState(seed)
    parts = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        for i, a in enumerate(np.array_split(idx, k)):
            parts[i].append(a)
    return [np.sort(np.concatenate(p)) for p in parts]


# ------------------------------------------------------------------------------------------------------------
# metrics / tests
# ------------------------------------------------------------------------------------------------------------
def reg_metrics(y, yhat, base_mean):
    e2 = (y - yhat) ** 2
    n = len(y)
    base = float(np.mean((y - base_mean) ** 2))
    return {"n": int(n), "mse": float(e2.mean()), "mse_se": float(e2.std(ddof=1) / np.sqrt(n)), "rmse": float(np.sqrt(e2.mean())),
            "mae": float(np.abs(y - yhat).mean()), "baseline_train_mean_mse": base, "rel_mse_vs_mean_baseline": float(e2.mean() / base)}


def auc_score(y, s):
    pos = int((y == 1).sum()); neg = len(y) - pos
    return float((rankdata(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def delong_var(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = rankdata(pos), rankdata(neg), rankdata(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    return auc, float(np.var((tz[:m] - tx) / n, ddof=1) / m + np.var(1 - (tz[m:] - ty) / m, ddof=1) / n)


def delong_parts(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = rankdata(pos), rankdata(neg), rankdata(np.concatenate([pos, neg]))
    return (tz[:m].sum() / m - (m + 1) / 2) / n, (tz[:m] - tx) / n, 1 - (tz[m:] - ty) / m


def youden_threshold(y, s):
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    J = np.where(np.r_[ss[1:] != ss[:-1], True], tp / tp[-1] - fp / fp[-1], -np.inf)
    i = int(np.argmax(J))
    return float((ss[i] + ss[i + 1]) / 2) if i + 1 < len(ss) else float(ss[i] - 1e-9)


def clip_prob(p):
    return np.clip(p, PROB_EPS, 1 - PROB_EPS)


def logloss_vec(y, p):
    p = clip_prob(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_table(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    return pd.DataFrame([{"bin": i, "n": int((b == i).sum()), "mean_pred": float(p[b == i].mean()), "frac_pos": float(y[b == i].mean())}
                         for i in range(len(edges) - 1) if (b == i).any()])


def clf_metrics(y, p, thr):
    p = clip_prob(p)
    n = len(y)
    ll = logloss_vec(y, p)
    yhat = (p > thr).astype(int)
    err = float((yhat != y).mean())
    auc, var = delong_var(y, p)
    cal = calibration_table(y, p)
    tn, fp = int(((y == 0) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 1)).sum())
    fn, tp = int(((y == 1) & (yhat == 0)).sum()), int(((y == 1) & (yhat == 1)).sum())
    null = logloss_vec(y, np.full(n, y.mean()))
    return {"n": int(n), "log_loss": float(ll.mean()), "log_loss_se": float(ll.std(ddof=1) / np.sqrt(n)), "null_log_loss": float(null.mean()),
            "brier": float(np.mean((p - y) ** 2)), "auc": auc, "auc_se": float(np.sqrt(var)), "threshold": float(thr), "error": err,
            "error_se": float(np.sqrt(err * (1 - err) / n)), "majority_class_error": float(min(y.mean(), 1 - y.mean())),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ece": float(np.sum(cal["n"] * np.abs(cal["mean_pred"] - cal["frac_pos"])) / n), "confusion_matrix": [[tn, fp], [fn, tp]]}


def one_se(mean, se, comp):
    mean = np.asarray(mean, float)
    imin = int(np.nanargmin(mean))
    ok = np.where(mean <= mean[imin] + se[imin])[0]
    return int(ok[np.argmin(np.asarray(comp)[ok])]), imin


def one_se_paired(fm, comp):
    """One-SE rule on PAIRED fold differences to the CV-best setting (the plain SE is inflated by fold-level shifts shared by all settings)."""
    K = fm.shape[0]
    imin = int(np.argmin(fm.mean(0)))
    d = fm - fm[:, [imin]]
    ok = np.where(d.mean(0) <= d.std(0, ddof=1) / np.sqrt(K))[0]
    return int(ok[np.argmin(np.asarray(comp)[ok])]), imin


def dm_test(a, b):
    """Diebold-Mariano on per-point losses a, b (rows in time order; Newey-West long-run variance; Harvey correction)."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    dc = d - d.mean()
    L = max(int(n ** (1 / 3)), 1)
    lr = float(dc @ dc) / n
    for l in range(1, L + 1):
        lr += 2 * (1 - l / (L + 1)) * float(dc[l:] @ dc[:-l]) / n
    if lr <= 0:
        return float(d.mean()), 0.0, 1.0
    stat = d.mean() / np.sqrt(lr / n) * np.sqrt((n - 1) / n)
    return float(d.mean()), float(stat), float(2 * (1 - t_dist.cdf(abs(stat), n - 1)))


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    s = d.mean() / se if se > 0 else 0.0
    return float(d.mean()), float(s), float(2 * (1 - t_dist.cdf(abs(s), n - 1)))


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o



# ------------------------------------------------------------------------------------------------------------
def _safe_extra(extra, log, summary, ctx):
    """Method-specific extra outputs (plots, importances, ...) must never destroy the trained model's predictions and metrics: log the failure, record it in summary.json, carry on."""
    try:
        extra(ctx)
    except Exception as e:
        import traceback
        log.error("extra outputs FAILED (%s: %s) - model, test predictions and metrics are unaffected\n%s", type(e).__name__, e, traceback.format_exc())
        summary["extra_error"] = f"{type(e).__name__}: {e}"


def run_method(name, prep_dir, out_dir, cfg, progress, *, make_grid, predict_path, select_by=None, complexity=None, diagnostics=None, extra=None,
               platt=False, scores_are_probs=True, notes="", console=False):
    """
    make_grid(cfg, prep, Xs, ys)                         -> list of hyper-parameter dicts; every dict has "k" = number of top-ranked variables used
    predict_path(Xref, yref, grid, Xq, cfg, ctx)         -> array (H, len(Xq)).  ctx["order"] = variable ranking computed from the reference rows only
                                                            (per CV fold -> no leakage); ctx["final"], ctx["fold"], ctx["yq"] (query targets: used ONLY for held-out
                                                            log-likelihood style diagnostics), ctx["metrics"] (method may add per-grid arrays that are averaged over folds)
    select_by                                            -> "mse" (regression) | "auc" | "log_loss" (classification)
    platt                                                -> classification only: calibrate raw scores to probabilities by Platt scaling fitted on the VALIDATION set
    scores_are_probs                                     -> False if raw scores can leave [0, 1] (MARS): CV log-loss is then not computed
    """
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out, name, console=console)
    prog = progress or Progress(name)
    K = cfg.cv_folds
    prog.set_total(K + 5)
    try:
        prep = Prepared(prep_dir)
        task = prep.task
        reg = task == "regression"
        select_by = select_by or ("mse" if reg else "auc")
        X_tr, X_va, X_te = prep.X("train"), prep.X("val"), prep.X("test")
        y_tr, y_va, y_te = prep.y("train"), prep.y("val"), prep.y("test")
        y_fit_tr = prep.y_fit_train if reg else y_tr.astype(float)
        ranks = prep.rankings()
        sub = tuning_subsample(prep, cfg)
        Xs, ys_fit, ys_ev = np.asarray(X_tr[sub]), y_fit_tr[sub], y_tr[sub]
        folds = make_folds(ys_ev, K, task, cfg.seed)
        grid = make_grid(cfg, prep, Xs, ys_fit)
        H = len(grid)
        log.info("%s | task=%s | %d candidate variables | train=%d (tuning subsample %d) val=%d test=%d | grid %d | folds %d | selection metric %s", name, task, prep.p,
                 len(y_tr), len(sub), len(y_va), len(y_te), H, K, select_by)
        diag = diagnostics(Xs, ys_fit, grid, cfg, ranks["sub"]) if diagnostics else {}
        comp = np.asarray(diag["complexity"], float) if "complexity" in diag else np.array([complexity(h) for h in grid], float) if complexity else np.arange(H, dtype=float)
        prog.tick("diagnostics")

        fm_mse, fm_auc, fm_ll = np.zeros((K, H)), np.zeros((K, H)), np.full((K, H), np.nan)
        oof = np.zeros((H, len(ys_ev)))
        acc = {}
        for k, idx in enumerate(folds):
            tr = np.ones(len(ys_ev), bool); tr[idx] = False
            ctx = {"final": False, "fold": k, "order": ranks["folds"][k], "task": task, "yref_raw": ys_ev[tr], "yq": ys_ev[idx], "metrics": {}}
            P = predict_path(Xs[tr], ys_fit[tr], grid, Xs[idx], cfg, ctx)
            for h in range(H):
                oof[h, idx] = P[h]
                if reg:
                    fm_mse[k, h] = float(np.mean((ys_ev[idx] - P[h]) ** 2))
                else:
                    fm_auc[k, h] = auc_score(ys_ev[idx], P[h])
                    if scores_are_probs:
                        fm_ll[k, h] = float(logloss_vec(ys_ev[idx], P[h]).mean())
            for mk, mv in ctx["metrics"].items():
                acc.setdefault(mk, []).append(np.asarray(mv, float))
            prog.tick(f"CV fold {k + 1}/{K}")
        if reg:
            fm, lname = fm_mse, "mse"
        elif select_by == "log_loss":
            fm, lname = fm_ll, "log_loss"
        else:
            fm, lname = -fm_auc, "neg_auc"
        cv_mean, cv_se = fm.mean(0), fm.std(0, ddof=1) / np.sqrt(K)
        i_plain, i_min = one_se(cv_mean, cv_se, comp)
        i_sel = one_se_paired(fm, comp)[0] if cfg.one_se_mode == "paired" else i_plain

        Pv = predict_path(Xs, ys_fit, grid, X_va, cfg, {"final": False, "fold": -1, "order": ranks["sub"], "task": task, "yref_raw": ys_ev, "yq": y_va, "metrics": {}})
        curve = pd.DataFrame([{**g} for g in grid])
        curve["complexity"] = comp
        curve["cv_" + lname], curve["cv_se"] = cv_mean, cv_se
        if reg:
            curve["val_mse_tuning_fit"] = [float(np.mean((y_va - Pv[h]) ** 2)) for h in range(H)]
        else:
            curve["cv_auc"] = fm_auc.mean(0)
            if scores_are_probs:
                curve["cv_log_loss"] = fm_ll.mean(0)
            curve["val_auc_tuning_fit"] = [auc_score(y_va, Pv[h]) for h in range(H)]
            thr_cv = [youden_threshold(ys_ev, oof[h]) for h in range(H)]
            curve["cv_error_pooled_youden"] = [float(((oof[h] > thr_cv[h]).astype(int) != ys_ev).mean()) for h in range(H)]
        for mk, v in acc.items():
            curve["cv_" + mk] = np.mean(v, axis=0)
        for k_, v in diag.items():
            if k_ != "complexity":
                curve[k_] = v
        curve["is_cv_min"], curve["is_one_se"], curve["is_one_se_plain"] = False, False, False
        curve.loc[i_min, "is_cv_min"] = True; curve.loc[i_sel, "is_one_se"] = True; curve.loc[i_plain, "is_one_se_plain"] = True
        curve.to_csv(out / "cv_curve.csv", index=False)
        log.info("CV minimum %s | one-SE choice %s (%s rule)", grid[i_min], grid[i_sel], cfg.one_se_mode)
        prog.tick("selection")

        # ---- final model: ALL training rows, chosen setting; ranking from the tuning subsample ----
        ctxf = {"final": True, "order": ranks["sub"], "store": {}, "task": task, "yref_raw": y_tr, "yq": np.r_[y_va, y_te], "n_val": len(y_va), "metrics": {}}
        Pf = predict_path(X_tr, y_fit_tr, [grid[i_sel]], np.vstack([X_va, X_te]), cfg, ctxf)[0]
        p_va, p_te = Pf[:len(y_va)], Pf[len(y_va):]
        prog.tick("final fit")
        platt_params = None
        if reg:
            base = float(y_tr.mean())
            val, test = reg_metrics(y_va, p_va, base), reg_metrics(y_te, p_te, base)
        else:
            if platt:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1e6, max_iter=500).fit(p_va[:, None], y_va)
                platt_params = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
                p_va, p_te = lr.predict_proba(p_va[:, None])[:, 1], lr.predict_proba(p_te[:, None])[:, 1]
            p_va, p_te = clip_prob(p_va), clip_prob(p_te)
            thr = youden_threshold(y_va, p_va)
            val, test = clf_metrics(y_va, p_va, thr), clf_metrics(y_te, p_te, thr)
        summary = {"method": name, "task": task, "selected": grid[i_sel], "k_variables": grid[i_sel]["k"], "selected_complexity": float(comp[i_sel]), "cv_min": grid[i_min],
                   "one_se_mode": cfg.one_se_mode, "selected_plain_one_se": grid[i_plain], "selection_metric": select_by,
                   "cv": {lname: float(cv_mean[i_sel]), "se": float(cv_se[i_sel])}, "validation": val, "test": test, "store": ctxf["store"].get("summary", {}),
                   "platt_scaling": platt_params, "tuning_rows": int(len(sub)), "final_reference_rows": int(len(y_tr)), "notes": notes, "config": asdict(cfg),
                   "selected_variables": [prep.feature_names[i] for i in ranks["sub"][:grid[i_sel]["k"]]], "runtime_sec": round(time.time() - t0, 1)}
        np.save(out / "test_pred.npy", p_te.astype(np.float64)); np.save(out / "val_pred.npy", p_va.astype(np.float64))

        o = np.argsort(comp)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.errorbar(comp[o], cv_mean[o], yerr=cv_se[o], fmt="o-", ms=3, lw=1, capsize=2, label=f"10-fold CV ({lname}) +- SE")
        ax.axvline(comp[i_min], color="tab:red", ls=":", label="CV minimum"); ax.axvline(comp[i_sel], color="tab:purple", ls="--", label="one-SE choice")
        ax.set_xlabel("model complexity"); ax.set_ylabel(lname); ax.set_title(name); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "cv_curve.png", dpi=120); plt.close(fig)
        if "k" in curve:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            g = curve.groupby("k")["cv_" + lname].min()
            ax.plot(g.index, g.values, "o-"); ax.set_xlabel("number of top-ranked variables k"); ax.set_ylabel(f"best CV {lname}"); ax.set_title(f"{name}: variable-selection curve")
            fig.tight_layout(); fig.savefig(out / "variable_selection_curve.png", dpi=120); plt.close(fig)
        if not reg:
            cal = calibration_table(y_te, p_te); cal.to_csv(out / "calibration_test.csv", index=False)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
            order = np.argsort(-p_te, kind="stable"); ys_ = y_te[order]
            ax[0].plot(np.cumsum(1 - ys_) / (1 - ys_).sum(), np.cumsum(ys_) / ys_.sum()); ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title(f"{name}: test ROC (AUC {test['auc']:.4f})")
            mx = max(cal["mean_pred"].max(), cal["frac_pos"].max()) * 1.05
            ax[1].plot(cal["mean_pred"], cal["frac_pos"], "o-"); ax[1].plot([0, mx], [0, mx], "k:"); ax[1].set_title("calibration (test, quantile bins)")
            fig.tight_layout(); fig.savefig(out / "roc_calibration_test.png", dpi=120); plt.close(fig)
        if extra:
            _safe_extra(extra, log, summary, {"prep": prep, "out": out, "curve": curve, "summary": summary, "grid": grid, "i_sel": i_sel, "Xs": Xs, "ys_fit": ys_fit, "ys_ev": ys_ev, "folds": folds,
                   "X_tr": X_tr, "y_tr": y_tr, "y_fit_tr": y_fit_tr, "X_va": X_va, "y_va": y_va, "X_te": X_te, "y_te": y_te, "p_te": p_te, "p_va": p_va, "cfg": cfg,
                   "predict_path": predict_path, "log": log, "store": ctxf["store"], "order": ranks["sub"], "metrics": ctxf["metrics"]})
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
        (out / "test_metrics.json").write_text(json.dumps(_jsonable(test), indent=2))
        (out / "validation_metrics.json").write_text(json.dumps(_jsonable(val), indent=2))
        if reg:
            log.info("VALIDATION MSE %.6f | TEST MSE %.6f +- %.6f (mean baseline %.6f) | k=%d | %.1fs", val["mse"], test["mse"], test["mse_se"], test["baseline_train_mean_mse"],
                     grid[i_sel]["k"], time.time() - t0)
        else:
            log.info("TEST AUC %.4f +- %.4f | error %.4f +- %.4f | log-loss %.5f +- %.5f (base rate %.5f) | k=%d | %.1fs", test["auc"], test["auc_se"], test["error"], test["error_se"],
                     test["log_loss"], test["log_loss_se"], test["null_log_loss"], grid[i_sel]["k"], time.time() - t0)
        prog.tick("done")
        return summary
    except Exception:
        log.error("FAILED\n%s", traceback.format_exc())
        raise


def method_main(name, run_fn):
    ap = argparse.ArgumentParser(description=f"Standalone run of '{name}'")
    ap.add_argument("--prepared", default="results/prepared")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--tune-rows", type=int, default=None)
    a = ap.parse_args()
    prep_dir = Path(a.prepared)
    cfg = RunConfig.from_json(prep_dir / "run_config.json") if (prep_dir / "run_config.json").exists() else RunConfig()
    for k, v in (("cv_folds", a.cv_folds), ("tune_rows", a.tune_rows)):
        if v:
            setattr(cfg, k, v)
    if not (prep_dir / "rankings.npz").exists():
        raise SystemExit("prepared data / variable rankings not found - run run_all.py first")
    run_fn(prep_dir, Path(a.results) / name, cfg, Progress(name, print_progress), console=True)
