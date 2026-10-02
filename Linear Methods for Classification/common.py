"""
common.py - shared infrastructure for the Santander linear-classification study (target: binary `Target`).

Contents
  RunConfig                  run settings
  Prepared                   memory-mapped access to one engineered feature set (train/val/test arrays)
  ClassStats / get_bundle    per-class sufficient statistics (n, sum x, sum x x') and stratified 10-fold CV folds,
                             so the Gaussian methods (LDA/QDA/RDA/reduced-rank LDA/indicator regression) can be
                             trained AND cross-validated without touching the big X matrix again
  metrics                    error +- SE, AUC (+ DeLong variance), log-loss, Brier, ECE, confusion matrix, Youden threshold
  run_classifier             the common train -> 10-fold CV (one-SE rule) -> validation threshold -> test -> logging flow.
                             Each method file only supplies its own `fit` (and grid / hooks).

Protocol (identical for every method)
  * stratified 80/10/10 train/validation/test split (features.py), same rows for both feature sets
  * hyper-parameters chosen by stratified 10-fold CV on TRAIN only (one-SE rule on CV log-loss for probabilistic methods,
    on CV AUC for the others - error at a fixed threshold is degenerate under ~10 % positives)
  * final model fit on TRAIN; decision threshold tuned on VALIDATION (Youden's J); everything reported on TEST
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    seed: int = 42
    cv_folds: int = 10
    perceptron_max_epochs: int = 50
    perceptron_pocket: bool = False          # keep the best-training-error epoch instead of the last iterate
    svm_C: float = 100.0                     # "large C" soft-margin separating hyperplane
    svm_max_iter: int = 300
    rda_alphas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)   # 0 = LDA covariances, 1 = QDA covariances
    rda_gammas: tuple = (0.5, 0.75, 1.0)               # 1 = no shrinkage toward a scaled identity
    l1_nlambda: int = 30
    l1_ratio: float = 1e-3                   # lambda_min / lambda_max
    l1_max_outer: int = 4
    cv_logistic: bool = True
    winner_metric: str = "auc"               # "auc" | "error"

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        for k in ("rda_alphas", "rda_gammas"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# ---------------------------------------------------------------------------------------------------------------
def get_logger(out_dir, name, console=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"clf.{name}.{out_dir}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(sh)
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


# ---------------------------------------------------------------------------------------------------------------
class Prepared:
    """One engineered feature set (directory with X_*.npy / y_*.npy / feature_names.json)."""

    def __init__(self, fs_dir):
        d = self.dir = Path(fs_dir)
        self.fs = d.name
        self.feature_names = json.loads((d / "feature_names.json").read_text())
        self.p = len(self.feature_names)
        for s in ("train", "val", "test"):
            setattr(self, f"X_{s}", np.load(d / f"X_{s}.npy", mmap_mode="r"))
            setattr(self, f"y_{s}", np.load(d / f"y_{s}.npy").astype(int))
        self.n_train, self.n_val, self.n_test = len(self.y_train), len(self.y_val), len(self.y_test)


def _chunks(rows, n_total, chunk):
    if rows is None:
        rows = slice(0, n_total)
    if isinstance(rows, slice):
        for s in range(rows.start, rows.stop, chunk):
            yield slice(s, min(s + chunk, rows.stop))
    else:
        for part in np.array_split(rows, max(1, int(np.ceil(len(rows) / chunk)))):
            yield part


class ClassStats:
    """Per-class sufficient statistics: n (2,), s = sum x (2,p), S = sum x x' (2,p,p)."""
    __slots__ = ("n", "s", "S")

    def __init__(self, n, s, S):
        self.n, self.s, self.S = n, s, S

    def __sub__(self, o):
        return ClassStats(self.n - o.n, self.s - o.s, self.S - o.S)

    def __add__(self, o):
        return ClassStats(self.n + o.n, self.s + o.s, self.S + o.S)

    def moments(self):
        """class means (2,p) and within-class scatter matrices W_k (2,p,p)."""
        mu = self.s / self.n[:, None]
        W = self.S - self.n[:, None, None] * mu[:, :, None] * mu[:, None, :]
        return mu, W


def build_class_stats(X, y, rows=None, chunk=20000):
    p = X.shape[1]
    n, s, S = np.zeros(2), np.zeros((2, p)), np.zeros((2, p, p))
    for sl in _chunks(rows, X.shape[0], chunk):
        xb = np.asarray(X[sl], dtype=np.float64)
        yb = y[sl]
        for k in (0, 1):
            m = yb == k
            if m.any():
                xk = xb[m]
                n[k] += len(xk)
                s[k] += xk.sum(0)
                S[k] += xk.T @ xk
    return ClassStats(n, s, S)


def make_folds(y, k, seed):
    """Stratified K-fold index sets (each class split evenly)."""
    rng = np.random.RandomState(seed)
    parts = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        for i, a in enumerate(np.array_split(idx, k)):
            parts[i].append(a)
    return [np.sort(np.concatenate(p)) for p in parts]


class TrainView:
    """What a method's `fit` sees: full X/y, a 0/1 row-weight vector `w` (0 = held out) and class statistics `cs`."""

    def __init__(self, prep, w, cs):
        self.X, self.y, self.w, self.cs = prep.X_train, prep.y_train, w, cs
        self.n = int(w.sum())
        self.p = prep.p


class Bundle:
    def view(self, prep, k):
        w = np.ones(prep.n_train)
        w[self.folds[k]] = 0.0
        return TrainView(prep, w, self.cs_total - self.cs_folds[k])

    def full_view(self, prep):
        return TrainView(prep, np.ones(prep.n_train), self.cs_total)


def get_bundle(prep: Prepared, cfg: RunConfig) -> Bundle:
    path = prep.dir / f"bundle_{cfg.cv_folds}_{cfg.seed}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    b = Bundle()
    b.folds = make_folds(prep.y_train, cfg.cv_folds, cfg.seed)
    b.cs_folds = [build_class_stats(prep.X_train, prep.y_train, rows) for rows in b.folds]
    b.cs_total = b.cs_folds[0]
    for c in b.cs_folds[1:]:
        b.cs_total = b.cs_total + c
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(b, f, protocol=4)
    tmp.replace(path)
    return b


def full_view_nocv(prep):
    return TrainView(prep, np.ones(prep.n_train), None)


# ---------------------------------------------------------------------------------------------------------------
class Model:
    """score(X) -> decision score (higher = class 1; log-odds for probabilistic methods)."""

    def __init__(self, score_fn, coef=None, intercept=0.0, info=None):
        self._score, self.coef, self.intercept, self.info = score_fn, coef, intercept, info or {}

    def score(self, X, chunk=20000):
        out = [self._score(np.asarray(X[a:a + chunk], dtype=np.float64)) for a in range(0, X.shape[0], chunk)]
        return np.concatenate(out)


def ols_solve(Gc, cc, rel_tol=1e-10):
    w, V = np.linalg.eigh(Gc)
    keep = w > w.max() * rel_tol * len(w)
    return V[:, keep] @ ((V[:, keep].T @ cc) / w[keep]), int(keep.sum())


def gaussian_model(mu, Sig, priors, jitter=1e-8):
    """Two-class Gaussian discriminant. Sig = one matrix (LDA, linear) or a pair (QDA, quadratic). Score = log posterior odds."""
    lp = np.log(priors[1] / priors[0])
    p = len(mu[0])
    if not isinstance(Sig, (tuple, list)):
        S = Sig + jitter * np.trace(Sig) / p * np.eye(p)
        beta = np.linalg.solve(S, mu[1] - mu[0])
        b = -0.5 * (mu[1] + mu[0]) @ beta + lp
        return Model(lambda X: X @ beta + b, coef=beta, intercept=b, info={"type": "linear"})
    P, ld = [], []
    for k in (0, 1):
        S = Sig[k] + jitter * np.trace(Sig[k]) / p * np.eye(p)
        sign, logdet = np.linalg.slogdet(S)
        P.append(np.linalg.inv(S)); ld.append(logdet)

    def score(X):
        d = []
        for k in (0, 1):
            Z = X - mu[k]
            d.append(-0.5 * ld[k] - 0.5 * np.einsum("ij,ij->i", Z @ P[k], Z))
        return d[1] - d[0] + lp
    return Model(score, info={"type": "quadratic", "logdet": ld})


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------
def auc_score(y, s):
    pos = int((y == 1).sum()); neg = len(y) - pos
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def delong_parts(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = rankdata(pos), rankdata(neg), rankdata(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    return auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m


def delong_var(y, s):
    auc, v01, v10 = delong_parts(y, s)
    return auc, float(np.var(v01, ddof=1) / len(v01) + np.var(v10, ddof=1) / len(v10))


def youden_threshold(y, s):
    """Threshold maximising TPR - FPR; predict class 1 when score > threshold."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    J = tp / tp[-1] - fp / fp[-1]
    distinct = np.r_[ss[1:] != ss[:-1], True]
    J = np.where(distinct, J, -np.inf)
    i = int(np.argmax(J))
    return float((ss[i] + ss[i + 1]) / 2) if i + 1 < len(ss) else float(ss[i] - 1e-9)


def log_loss_vec(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_table(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        m = b == i
        if m.any():
            rows.append({"bin": i, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)


def eval_scores(y, s, thr, probabilistic):
    n = len(y)
    yhat = (s > thr).astype(int)
    tn, fp = int(((y == 0) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 1)).sum())
    fn, tp = int(((y == 1) & (yhat == 0)).sum()), int(((y == 1) & (yhat == 1)).sum())
    err = (fp + fn) / n
    err0 = float(((s > 0).astype(int) != y).mean())
    auc, var = delong_var(y, s)
    out = {"n": int(n), "threshold": float(thr), "error": err, "error_se": float(np.sqrt(err * (1 - err) / n)),
           "error_at_default_rule": err0, "majority_class_error": float(min(y.mean(), 1 - y.mean())),
           "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
           "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
           "auc": auc, "auc_se": float(np.sqrt(var)), "confusion_matrix": [[tn, fp], [fn, tp]]}
    if probabilistic:
        p = sigmoid(s)
        ll = log_loss_vec(y, p)
        cal = calibration_table(y, p)
        out.update({"log_loss": float(ll.mean()), "log_loss_se": float(ll.std(ddof=1) / np.sqrt(n)),
                    "brier": float(np.mean((p - y) ** 2)),
                    "ece": float(np.sum(cal["n"] * np.abs(cal["mean_pred"] - cal["frac_pos"])) / n),
                    "null_log_loss": float(log_loss_vec(y, np.full(n, y.mean())).mean())})
    return out


def one_se(loss_mean, loss_se, complexity):
    """Least complex setting whose CV loss <= min + SE(min). Returns (i_1se, i_min)."""
    m = np.asarray(loss_mean, float)
    imin = int(np.nanargmin(m))
    ok = np.where(m <= m[imin] + loss_se[imin])[0]
    return int(ok[np.argmin(np.asarray(complexity)[ok])]), imin


# ---------------------------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------------------------
def plot_roc(y, s, thr, path, title=""):
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tpr, fpr = np.cumsum(ys) / ys.sum(), np.cumsum(1 - ys) / (1 - ys).sum()
    k = int(np.searchsorted(-s[order], -thr))
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot(fpr, tpr, lw=1.5)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.scatter([fpr[min(k, len(fpr) - 1)]], [tpr[min(k, len(tpr) - 1)]], color="red", zorder=5, label="validation Youden threshold")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title(f"{title} - test ROC"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_calibration(cal, path, title=""):
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    ax.plot(cal["mean_pred"], cal["frac_pos"], "o-")
    m = max(cal["mean_pred"].max(), cal["frac_pos"].max()) * 1.05
    ax.plot([0, m], [0, m], "k:")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed positive rate"); ax.set_title(f"{title} - calibration (test, quantile bins)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm, path, title=""):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{title} - test confusion")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


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


# ---------------------------------------------------------------------------------------------------------------
# the common method flow
# ---------------------------------------------------------------------------------------------------------------
def run_classifier(name, fs_dir, out_dir, cfg, progress, *, make_grid, fit=None, fit_path=None, probabilistic,
                   complexity=None, do_cv=True, extra_plot=None, notes="", console=False):
    """
    make_grid(prep, cfg)        -> list of hyper-parameter dicts ([{}] if none)
    fit(view, hp)               -> Model            (one setting)
    fit_path(view, grid)        -> list[Model]      (whole grid at once, e.g. warm-started lambda path)
    complexity(hp)              -> number, larger = more complex (used by the one-SE rule); default: grid order
    do_cv                       -> run stratified 10-fold CV on the training set
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out, name, console=console)
    prog = progress or Progress(name)
    try:
        prep = Prepared(fs_dir)
        K = cfg.cv_folds
        grid = make_grid(prep, cfg)
        H = len(grid)
        comp = np.array([complexity(h) for h in grid], float) if complexity else np.arange(H, dtype=float)
        ytr = prep.y_train
        log.info("%s [%s] | train=%d val=%d test=%d p=%d | positives: train %.3f | grid size %d | cv=%s", name, prep.fs,
                 prep.n_train, prep.n_val, prep.n_test, prep.p, ytr.mean(), H, do_cv)
        cv_df, i_sel, i_min, sel_metric = None, 0, 0, None
        bundle = None
        if do_cv:
            bundle = get_bundle(prep, cfg)
            prog.set_total((K if fit_path else H * K) + 4)
            oof = np.zeros((H, prep.n_train))
            for k in range(K):
                view = bundle.view(prep, k)
                idx = bundle.folds[k]
                Xk = np.asarray(prep.X_train[idx])
                if fit_path:
                    models = fit_path(view, grid)
                    prog.tick(f"CV fold {k + 1}/{K}")
                else:
                    models = []
                    for h in range(H):
                        models.append(fit(view, grid[h]))
                        prog.tick(f"CV fold {k + 1}/{K} setting {h + 1}/{H}")
                for h, m in enumerate(models):
                    oof[h, idx] = m.score(Xk)
            rows = []
            for h in range(H):
                s = oof[h]
                thr = youden_threshold(ytr, s)
                fe, fa, fl = [], [], []
                for idx in bundle.folds:
                    yk, sk = ytr[idx], s[idx]
                    fe.append(float(((sk > thr).astype(int) != yk).mean()))
                    fa.append(auc_score(yk, sk))
                    if probabilistic:
                        fl.append(float(log_loss_vec(yk, sigmoid(sk)).mean()))
                r = {**grid[h], "complexity": comp[h], "cv_threshold": thr,
                     "cv_error": np.mean(fe), "cv_error_se": np.std(fe, ddof=1) / np.sqrt(K),
                     "cv_auc": np.mean(fa), "cv_auc_se": np.std(fa, ddof=1) / np.sqrt(K)}
                if probabilistic:
                    r.update({"cv_logloss": np.mean(fl), "cv_logloss_se": np.std(fl, ddof=1) / np.sqrt(K)})
                rows.append(r)
            cv_df = pd.DataFrame(rows)
            if probabilistic:
                sel_metric, lm, ls = "cv_logloss", cv_df["cv_logloss"].values, cv_df["cv_logloss_se"].values
            else:
                sel_metric, lm, ls = "cv_auc", -cv_df["cv_auc"].values, cv_df["cv_auc_se"].values
            i_sel, i_min = one_se(lm, ls, comp)
            cv_df["is_cv_best"] = False; cv_df["is_one_se_choice"] = False
            cv_df.loc[i_min, "is_cv_best"] = True; cv_df.loc[i_sel, "is_one_se_choice"] = True
            cv_df.to_csv(out / "cv_results.csv", index=False)
            log.info("CV (%s): best setting %s ; one-SE choice %s", sel_metric, grid[i_min], grid[i_sel])
            if H > 1:
                fig, ax = plt.subplots(figsize=(7, 4.3))
                x = np.arange(H)
                ax.errorbar(x, cv_df[sel_metric], yerr=cv_df[sel_metric + "_se"], fmt="o-", ms=3, capsize=2)
                ax.axvline(i_min, color="red", ls=":", label="CV best"); ax.axvline(i_sel, color="purple", ls="--", label="one-SE choice")
                ax.set_xlabel("grid index (increasing complexity where ordered)"); ax.set_ylabel(sel_metric); ax.legend(); ax.set_title(f"{name}: {sel_metric}")
                fig.tight_layout(); fig.savefig(out / "cv_curve.png", dpi=120); plt.close(fig)
        else:
            prog.set_total(3)

        # ---- final fit on the full training set ----
        view = bundle.full_view(prep) if bundle is not None else full_view_nocv(prep)
        if fit_path:
            model = fit_path(view, grid)[i_sel]
        else:
            model = fit(view, grid[i_sel])
        prog.tick("final fit")

        # ---- validation threshold, test evaluation ----
        sv = model.score(prep.X_val)
        thr = youden_threshold(prep.y_val, sv)
        st = model.score(prep.X_test)
        rng = np.random.RandomState(0)
        sub = np.sort(rng.choice(prep.n_train, min(30000, prep.n_train), replace=False))
        s_tr = model.score(np.asarray(prep.X_train[sub]))
        train = eval_scores(prep.y_train[sub], s_tr, thr, probabilistic)
        val = eval_scores(prep.y_val, sv, thr, probabilistic)
        test = eval_scores(prep.y_test, st, thr, probabilistic)
        prog.tick("validation / test")

        summary = {"method": name, "feature_set": prep.fs, "probabilistic": probabilistic,
                   "selected_hyperparameters": grid[i_sel], "cv_best_hyperparameters": grid[i_min],
                   "selection_metric": sel_metric,
                   "cv": None if cv_df is None else {c: cv_df.loc[i_sel, c] for c in cv_df.columns if c.startswith("cv_")},
                   "train_subsample_30k": train, "validation": val, "test": test, "model_info": model.info,
                   "n_nonzero_coefficients": None if model.coef is None else int((np.abs(model.coef) > 1e-12).sum()),
                   "notes": notes, "config": asdict(cfg), "runtime_sec": round(time.time() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
        (out / "test_metrics.json").write_text(json.dumps(_jsonable(test), indent=2))
        (out / "validation_metrics.json").write_text(json.dumps(_jsonable(val), indent=2))
        np.save(out / "test_scores.npy", st.astype(np.float64))
        np.save(out / "test_pred.npy", (st > thr).astype(np.int8))
        np.save(out / "val_scores.npy", sv.astype(np.float64))
        if model.coef is not None:
            pd.DataFrame({"feature": prep.feature_names, "coef": model.coef}).to_csv(out / "coefficients.csv", index=False)
        plot_roc(prep.y_test, st, thr, out / "roc_test.png", f"{name} [{prep.fs}]")
        plot_confusion(test["confusion_matrix"], out / "confusion_test.png", f"{name} [{prep.fs}]")
        if probabilistic:
            cal = calibration_table(prep.y_test, sigmoid(st))
            cal.to_csv(out / "calibration_test.csv", index=False)
            plot_calibration(cal, out / "calibration_test.png", f"{name} [{prep.fs}]")
        if extra_plot:
            extra_plot(out, cv_df, summary)
        log.info("TEST: error %.4f +- %.4f (majority %.4f) | AUC %.4f +- %.4f%s | threshold %.4f | %.1fs", test["error"],
                 test["error_se"], test["majority_class_error"], test["auc"], test["auc_se"],
                 f" | log-loss {test['log_loss']:.4f} +- {test['log_loss_se']:.4f}" if probabilistic else "", thr, time.time() - t0)
        prog.tick("done")
        return summary
    except Exception:
        log.error("FAILED\n%s", traceback.format_exc())
        raise


def method_main(name, run_fn):
    ap = argparse.ArgumentParser(description=f"Standalone run of '{name}'")
    ap.add_argument("--data", default=None, help="santander_train.csv (only needed if --prepared does not exist yet)")
    ap.add_argument("--prepared", default="results/prepared")
    ap.add_argument("--results", default="results")
    ap.add_argument("--feature-set", choices=["lean", "rich"], default="lean")
    ap.add_argument("--cv-folds", type=int, default=None)
    a = ap.parse_args()
    prep_root = Path(a.prepared)
    cfg_path = prep_root / "run_config.json"
    cfg = RunConfig.from_json(cfg_path) if cfg_path.exists() else RunConfig()
    if a.cv_folds:
        cfg.cv_folds = a.cv_folds
    if not (prep_root / a.feature_set / "feature_names.json").exists():
        if not a.data:
            raise SystemExit("prepared data not found - pass --data santander_train.csv to build it")
        import features
        features.build_prepared(features.load_table(a.data), prep_root, cfg)
    run_fn(prep_root / a.feature_set, Path(a.results) / a.feature_set / name, cfg, Progress(name, print_progress), console=True)
