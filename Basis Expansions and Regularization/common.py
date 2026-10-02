"""
common.py - shared infrastructure for the Zillow basis-expansion study (ESLII Ch. 5).

Protocol (identical for every method)
  * temporal split: train <= 2017-02-28, validation 2017-03-01..07-31, test >= 2017-08-01 (prepare.py)
  * fitting target = logerror clipped at the train 1st/99th percentile; EVERY metric uses the RAW logerror
  * smoothing parameter chosen by 10-fold CV on TRAIN only (contiguous time blocks of the date-sorted train set, so no future
    leakage into past folds); one-SE rule on the effective df (least complex model within 1 SE of the CV minimum). Default = PAIRED SE of the
    fold-wise differences to the best model (time-blocked folds differ in level, which inflates the plain SE); the plain ESLII choice is logged too
  * validation curve from the train-only fits; final model refit on train+validation at the CV-chosen setting; test evaluated once
  * smoothers see ONE predictor (chosen on train in prepare.py); thin-plate splines see (latitude, longitude)
"""
from __future__ import annotations

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
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import rankdata, t as t_dist

ROOT = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    seed: int = 42
    cv_folds: int = 10
    cv_scheme: str = "blocked"          # "blocked" (contiguous time blocks) | "random"
    refit: str = "train_val"            # "train" | "train_val"
    predictor: str = "auto"             # column used by the 1-D smoothers ("auto" = best on train CV)
    smooth_knots: int = 150             # smoothing spline: number of interior B-spline knots (R smooth.spline style)
    tps_knots: int = 200                # thin-plate: number of k-means knots
    rkhs_landmarks: int = 300           # RKHS (Nystrom): number of landmarks
    logit_knots: int = 25               # nonparametric logistic: interior knots
    wavelet: str = "sym8"               # needs PyWavelets, otherwise Haar is used
    wavelet_sim_reps: int = 50
    wavelet_sim_n: int = 1024
    wavelet_sim_snr: float = 7.0
    alpha: float = 0.05
    one_se_mode: str = "paired"         # "paired" (SE of fold-wise differences to the CV minimum) | "plain" (ESLII SE of the CV mean)

    def to_json(self, p):
        Path(p).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, p):
        return cls(**json.loads(Path(p).read_text()))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# ---------------------------------------------------------------------------------------------------------------
def get_logger(out_dir, name, console=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"basis.{name}.{out_dir}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    for h in list(lg.handlers):
        lg.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt); lg.addHandler(fh)
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


# ---------------------------------------------------------------------------------------------------------------
class Prepared:
    def __init__(self, d):
        d = self.dir = Path(d)
        self.meta = json.loads((d / "prepared_meta.json").read_text())
        for s in ("train", "val", "test"):
            setattr(self, f"x_{s}", np.load(d / f"x_{s}.npy"))
            setattr(self, f"xy_{s}", np.load(d / f"xy_{s}.npy"))
            setattr(self, f"y_{s}", np.load(d / f"y_raw_{s}.npy"))
            setattr(self, f"b_{s}", np.load(d / f"b_{s}.npy").astype(float))
            setattr(self, f"dates_{s}", np.load(d / f"dates_{s}.npy"))
        self.y_fit_train, self.y_fit_val = np.load(d / "y_fit_train.npy"), np.load(d / "y_fit_val.npy")
        self.predictor, self.x_label = self.meta["predictor"], self.meta["predictor_label"]

    def inputs(self, kind, split):
        return getattr(self, f"x_{split}")[:, None] if kind == "x" else getattr(self, f"xy_{split}")


def make_folds(n, k, scheme, seed):
    if scheme == "blocked":
        e = np.linspace(0, n, k + 1).astype(int)
        return [np.arange(e[i], e[i + 1]) for i in range(k)]
    return [np.sort(a) for a in np.array_split(np.random.RandomState(seed).permutation(n), k)]


# ---------------------------------------------------------------------------------------------------------------
# smoothers
# ---------------------------------------------------------------------------------------------------------------
class Smoother:
    """predict(X) -> fitted values (probabilities for the logistic task); optional df, rss, se(X), leverage(X)."""

    def __init__(self, predict_fn, df=None, rss=None, n=None, se_fn=None, leverage_fn=None, info=None):
        self._predict, self.df, self.rss, self.n = predict_fn, df, rss, n
        self.se_fn, self.leverage_fn, self.info = se_fn, leverage_fn, info or {}

    def predict(self, X, chunk=50000):
        return np.concatenate([self._predict(X[a:a + chunk]) for a in range(0, len(X), chunk)])

    def se(self, X):
        return None if self.se_fn is None else self.se_fn(X)

    def leverage(self, X):
        return None if self.leverage_fn is None else np.concatenate([self.leverage_fn(X[a:a + 50000]) for a in range(0, len(X), 50000)])


def linear_smoother(basis_fn, theta, Ainv, Cov, rss, n, df, info=None):
    def se(X):
        B = basis_fn(X)
        return np.sqrt(max(rss / max(n - df, 1.0), 0.0) * np.maximum(np.sum((B @ Cov) * B, axis=1), 0.0))

    def lev(X):
        B = basis_fn(X)
        return np.sum((B @ Ainv) * B, axis=1)
    return Smoother(lambda X: basis_fn(X) @ theta, df=df, rss=rss, n=n, se_fn=se, leverage_fn=lev, info=info)


def ls_fit(basis_fn, X, y, info=None):
    """Unpenalised least squares through a QR decomposition (stable even for the ill-conditioned truncated-power basis)."""
    B = basis_fn(X)
    n, m = B.shape
    Q, R = np.linalg.qr(B)
    try:
        th = solve_triangular(R, Q.T @ y)
        Rinv = solve_triangular(R, np.eye(m))
        Ainv = Rinv @ Rinv.T
    except Exception:
        th = np.linalg.lstsq(B, y, rcond=None)[0]
        Ainv = np.linalg.pinv(B.T @ B)
    r = y - B @ th
    inf = dict(info or {}); inf["cond_design"] = float(np.linalg.cond(B)) if m <= 200 else None
    return linear_smoother(basis_fn, th, Ainv, Ainv, float(r @ r), n, float(m), inf)


def gram(B, y):
    return B.T @ B, B.T @ y, float(y @ y)


def penalized_fit(basis_fn, G, c, yy, n, Omega, lam, info=None):
    """Generalised ridge  theta = (G + lam*Omega)^-1 c ;  df = trace(A^-1 G) ;  Cov = A^-1 G A^-1."""
    m = G.shape[0]
    A0 = G + lam * Omega
    cf = None
    for jit in (1e-11, 1e-9, 1e-7, 1e-5):                       # retry with more damping if the system is numerically singular
        try:
            cf = cho_factor(A0 + jit * np.trace(A0) / m * np.eye(m))
            break
        except Exception:
            continue
    if cf is None:
        raise np.linalg.LinAlgError("penalized system not positive definite")
    th = cho_solve(cf, c)
    Ainv = cho_solve(cf, np.eye(m))
    rss = max(yy - 2 * th @ c + th @ G @ th, 0.0)
    df = float(np.trace(Ainv @ G))
    inf = dict(info or {}); inf["lambda"] = float(lam)
    return linear_smoother(basis_fn, th, Ainv, Ainv @ G @ Ainv, rss, n, df, inf)


# ---------------------------------------------------------------------------------------------------------------
# metrics and tests
# ---------------------------------------------------------------------------------------------------------------
def reg_metrics(y, yhat, base_mean):
    e2 = (y - yhat) ** 2
    n = len(y)
    base = float(np.mean((y - base_mean) ** 2))
    return {"n": int(n), "mse": float(e2.mean()), "mse_se": float(e2.std(ddof=1) / np.sqrt(n)), "rmse": float(np.sqrt(e2.mean())),
            "mae": float(np.abs(y - yhat).mean()), "baseline_train_mean_mse": base, "rel_mse_vs_mean_baseline": float(e2.mean() / base),
            "r2_vs_mean_baseline": float(1 - e2.mean() / base)}


def auc_score(y, s):
    pos = int((y == 1).sum()); neg = len(y) - pos
    return float((rankdata(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def delong_var(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = rankdata(pos), rankdata(neg), rankdata(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    return auc, float(np.var((tz[:m] - tx) / n, ddof=1) / m + np.var(1 - (tz[m:] - ty) / m, ddof=1) / n)


def youden_threshold(y, s):
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    J = tp / tp[-1] - fp / fp[-1]
    J = np.where(np.r_[ss[1:] != ss[:-1], True], J, -np.inf)
    i = int(np.argmax(J))
    return float((ss[i] + ss[i + 1]) / 2) if i + 1 < len(ss) else float(ss[i] - 1e-9)


def logloss_vec(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_table(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = [{"bin": i, "n": int((b == i).sum()), "mean_pred": float(p[b == i].mean()), "frac_pos": float(y[b == i].mean())}
            for i in range(len(edges) - 1) if (b == i).any()]
    return pd.DataFrame(rows)


def bin_metrics(y, p, thr):
    n = len(y)
    ll = logloss_vec(y, p)
    yhat = (p > thr).astype(int)
    err = float((yhat != y).mean())
    auc, var = delong_var(y, p)
    cal = calibration_table(y, p)
    tn, fp = int(((y == 0) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 1)).sum())
    fn, tp = int(((y == 1) & (yhat == 0)).sum()), int(((y == 1) & (yhat == 1)).sum())
    null = logloss_vec(y, np.full(n, y.mean()))
    return {"n": int(n), "log_loss": float(ll.mean()), "log_loss_se": float(ll.std(ddof=1) / np.sqrt(n)),
            "null_log_loss": float(null.mean()), "log_loss_improvement_vs_null": float(null.mean() - ll.mean()),
            "auc": auc, "auc_se": float(np.sqrt(var)), "brier": float(np.mean((p - y) ** 2)),
            "ece": float(np.sum(cal["n"] * np.abs(cal["mean_pred"] - cal["frac_pos"])) / n),
            "threshold": float(thr), "error": err, "error_se": float(np.sqrt(err * (1 - err) / n)),
            "majority_class_error": float(min(y.mean(), 1 - y.mean())), "confusion_matrix": [[tn, fp], [fn, tp]]}


def one_se(mean, se, comp):
    mean = np.asarray(mean, float)
    imin = int(np.nanargmin(mean))
    ok = np.where(mean <= mean[imin] + se[imin])[0]
    return int(ok[np.argmin(np.asarray(comp)[ok])]), imin


def one_se_paired(fm, comp):
    """One-SE rule on PAIRED fold differences: with time-blocked folds the folds differ a lot in level (shared by all models), which inflates the
    plain SE of the CV mean; the SE of (fold loss of model h - fold loss of the CV-best model) isolates real differences between models."""
    K = fm.shape[0]
    imin = int(np.argmin(fm.mean(0)))
    d = fm - fm[:, [imin]]
    se_d = d.std(0, ddof=1) / np.sqrt(K)
    ok = np.where(d.mean(0) <= se_d)[0]
    return int(ok[np.argmin(np.asarray(comp)[ok])]), imin


def info_criteria(rss, n, df, sigma2):
    rss, df = np.asarray(rss, float), np.asarray(df, float)
    base = n * np.log(np.maximum(rss, 1e-300) / n)
    return {"aic": base + 2 * df, "bic": base + df * np.log(n), "cp": rss / n + 2 * df * sigma2 / n}


def dm_test(e2_a, e2_b):
    """Diebold-Mariano (h = 1, Bartlett/Newey-West long-run variance, Harvey small-sample correction) on squared-error losses."""
    d = np.asarray(e2_a) - np.asarray(e2_b)
    n = len(d)
    dbar = d.mean()
    L = max(int(n ** (1 / 3)), 1)
    dc = d - dbar
    lr = float(dc @ dc) / n
    for l in range(1, L + 1):
        lr += 2 * (1 - l / (L + 1)) * float(dc[l:] @ dc[:-l]) / n
    if lr <= 0:
        return float(dbar), 0.0, 1.0
    stat = dbar / np.sqrt(lr / n) * np.sqrt((n - 1) / n)
    return float(dbar), float(stat), float(2 * (1 - t_dist.cdf(abs(stat), n - 1)))


def paired_t(e2_a, e2_b):
    d = np.asarray(e2_a) - np.asarray(e2_b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    stat = d.mean() / se if se > 0 else 0.0
    return float(d.mean()), float(stat), float(2 * (1 - t_dist.cdf(abs(stat), n - 1)))


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def moran_i(coords, resid, k=8, perms=99, seed=0, max_n=8000):
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    if len(coords) > max_n:
        s = rng.choice(len(coords), max_n, replace=False)
        coords, resid = coords[s], resid[s]
    z = resid - resid.mean()
    nb = cKDTree(coords).query(coords, k + 1)[1][:, 1:]
    def stat(zz):
        return len(zz) / (len(zz) * k) * np.sum(zz[:, None] * zz[nb]) / np.sum(zz ** 2)
    I = stat(z)
    ps = [stat(rng.permutation(z)) for _ in range(perms)]
    return float(I), float((1 + np.sum(np.abs(ps) >= abs(I))) / (perms + 1))


# ---------------------------------------------------------------------------------------------------------------
def _binned(x, y, nb=40):
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)
    xs, ms, ses = [], [], []
    for i in range(len(e) - 1):
        m = b == i
        if m.sum() > 1:
            xs.append(x[m].mean()); ms.append(y[m].mean()); ses.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(xs), np.array(ms), np.array(ses)


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
def run_smoother(name, prep_dir, out_dir, cfg, progress, *, inputs, make_grid, fit_path, task="regression", want_loocv=False,
                 extra=None, notes="", console=False):
    """
    make_grid(prep, cfg, X, y)        -> list of hyper-parameter dicts (built from TRAIN data only)
    fit_path(X, y, grid, cfg)         -> list[Smoother], one per grid point (called on every CV training fold, on train, on train+val)
    extra(ctx)                        -> optional method-specific plots / tables;  ctx is a dict (prep, out, curve, summary, ...)
    task                              -> "regression" (MSE of raw logerror) | "logistic" (log-loss of the large-miss indicator)
    """
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out, name, console=console)
    prog = progress or Progress(name)
    K = cfg.cv_folds
    prog.set_total(K + 5)
    try:
        prep = Prepared(prep_dir)
        Xtr, Xva, Xte = (prep.inputs(inputs, s) for s in ("train", "val", "test"))
        if task == "regression":
            y_fit_tr, y_ev_tr, y_fit_va, y_ev_va, y_te = prep.y_fit_train, prep.y_train, prep.y_fit_val, prep.y_val, prep.y_test
        else:
            y_fit_tr = y_ev_tr = prep.b_train; y_fit_va = y_ev_va = prep.b_val; y_te = prep.b_test
        base_mean = float(prep.y_train.mean())
        loss = (lambda y, p: float(np.mean((y - p) ** 2))) if task == "regression" else (lambda y, p: float(logloss_vec(y, p).mean()))
        lname = "mse" if task == "regression" else "log_loss"
        grid = make_grid(prep, cfg, Xtr, y_fit_tr)
        H = len(grid)
        log.info("%s | task=%s inputs=%s predictor=%s | train=%d val=%d test=%d | grid size %d", name, task, inputs, prep.predictor,
                 len(y_ev_tr), len(y_ev_va), len(y_te), H)

        models = fit_path(Xtr, y_fit_tr, grid, cfg)
        prog.tick("fit on train")
        df = np.array([np.nan if m.df is None else m.df for m in models], float)
        comp = df if np.all(np.isfinite(df)) else np.arange(H, dtype=float)

        # ---- 10-fold CV (blocked in time) ----
        folds = make_folds(len(y_ev_tr), K, cfg.cv_scheme, cfg.seed)
        fm = np.zeros((K, H))
        for k, idx in enumerate(folds):
            tr = np.ones(len(y_ev_tr), bool); tr[idx] = False
            ms = fit_path(Xtr[tr], y_fit_tr[tr], grid, cfg)
            for h, m in enumerate(ms):
                fm[k, h] = loss(y_ev_tr[idx], m.predict(Xtr[idx]))
            prog.tick(f"CV fold {k + 1}/{K}")
        cv_mean, cv_se = fm.mean(0), fm.std(0, ddof=1) / np.sqrt(K)

        # ---- curves from the train-only fits ----
        val_loss = np.array([loss(y_ev_va, m.predict(Xva)) for m in models])
        curve = pd.DataFrame([{**g} for g in grid])
        curve["df"] = df
        curve["cv_" + lname], curve["cv_se"], curve["val_" + lname] = cv_mean, cv_se, val_loss
        n_tr = len(y_fit_tr)
        if task == "regression" and np.all(np.isfinite(df)) and all(m.rss is not None for m in models):
            rss = np.array([m.rss for m in models])
            big = int(np.argmax(df))
            sig2 = rss[big] / max(n_tr - df[big], 1.0)                 # low-bias model estimate (ESLII)
            curve["train_rss"] = rss
            for kk, v in info_criteria(rss, n_tr, df, sig2).items():
                curve[kk] = v
            curve["gcv"] = (rss / n_tr) / (1 - df / n_tr) ** 2
            if want_loocv and models[0].leverage_fn is not None:
                loo = []
                for m in models:
                    h_ = m.leverage(Xtr)
                    loo.append(float(np.mean(((y_fit_tr - m.predict(Xtr)) / (1 - h_)) ** 2)))
                curve["loocv_mse_fit_target"] = loo
        i_plain, i_min = one_se(cv_mean, cv_se, comp)
        i_sel = one_se_paired(fm, comp)[0] if cfg.one_se_mode == "paired" else i_plain
        curve["is_cv_min"] = False; curve["is_one_se"] = False; curve["is_one_se_plain"] = False
        curve.loc[i_min, "is_cv_min"] = True; curve.loc[i_sel, "is_one_se"] = True; curve.loc[i_plain, "is_one_se_plain"] = True
        for k in range(K):
            curve[f"cv_fold{k + 1}"] = fm[k]
        curve.to_csv(out / "cv_curve.csv", index=False)
        log.info("CV minimum: %s (df %.2f, %s %.6f) ; one-SE choice: %s (df %.2f, %s %.6f)", grid[i_min], comp[i_min], lname,
                 cv_mean[i_min], grid[i_sel], comp[i_sel], lname, cv_mean[i_sel])
        prog.tick("curves")

        # ---- final model ----
        m_tr = models[i_sel]
        if cfg.refit == "train_val":
            Xall = np.vstack([Xtr, Xva]); yall = np.r_[y_fit_tr, y_fit_va]
            m_fin = fit_path(Xall, yall, [grid[i_sel]], cfg)[0]
            n_fit = len(yall)
        else:
            m_fin, n_fit = m_tr, n_tr
        prog.tick("final refit")

        p_va = m_tr.predict(Xva)
        p_te = m_fin.predict(Xte)
        if task == "regression":
            val, test = reg_metrics(prep.y_val, p_va, base_mean), reg_metrics(prep.y_test, p_te, base_mean)
        else:
            thr = youden_threshold(prep.b_val, p_va)
            val, test = bin_metrics(prep.b_val, p_va, thr), bin_metrics(prep.b_test, p_te, thr)
        summary = {"method": name, "task": task, "inputs": inputs, "predictor": prep.predictor if inputs == "x" else "latitude,longitude",
                   "selected": grid[i_sel], "selected_df": float(comp[i_sel]), "cv_min": grid[i_min], "cv_min_df": float(comp[i_min]),
                   "one_se_mode": cfg.one_se_mode, "selected_plain_one_se": grid[i_plain], "selected_plain_one_se_df": float(comp[i_plain]),
                   "cv": {lname: float(cv_mean[i_sel]), "se": float(cv_se[i_sel])}, "validation_train_only_model": val, "test": test,
                   "final_model_df": m_fin.df, "final_fit_rows": int(n_fit), "final_model_info": m_fin.info, "notes": notes,
                   "config": asdict(cfg), "runtime_sec": round(time.time() - t0, 1)}
        for col in ("aic", "bic", "cp", "gcv", "loocv_mse_fit_target"):
            if col in curve:
                summary[f"argmin_{col}_df"] = float(curve.loc[int(np.nanargmin(curve[col].values)), "df"])
        np.save(out / "test_pred.npy", p_te.astype(np.float64))
        np.save(out / "val_pred.npy", p_va.astype(np.float64))

        # ---- plots ----
        xcol = "df" if np.all(np.isfinite(df)) else None
        xx = comp
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        o = np.argsort(xx)
        ax.errorbar(xx[o], cv_mean[o], yerr=cv_se[o], fmt="o-", ms=3, lw=1, capsize=2, label="10-fold CV (train, time-blocked) +- SE")
        ax.plot(xx[o], val_loss[o], "s--", ms=3, lw=1, color="tab:green", label="validation (Mar-Jul 2017)")
        ax.axvline(comp[i_min], color="tab:red", ls=":", label="CV minimum"); ax.axvline(comp[i_sel], color="tab:purple", ls="--", label="one-SE choice")
        ax.set_xlabel("effective degrees of freedom" if xcol else "grid index"); ax.set_ylabel(lname.upper()); ax.set_title(name); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "cv_curve.png", dpi=120); plt.close(fig)
        if "aic" in curve:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            for c_ in ("aic", "bic"):
                ax.plot(curve["df"], curve[c_] - curve[c_].min(), "o-", ms=3, label=c_.upper() + " (- min)")
            ax2 = ax.twinx(); ax2.plot(curve["df"], curve["cp"], "g.--", label="Cp"); ax2.plot(curve["df"], curve["gcv"], "m.--", label="GCV")
            ax.set_xlabel("df"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.set_title(f"{name}: information criteria")
            fig.tight_layout(); fig.savefig(out / "information_criteria.png", dpi=120); plt.close(fig)
        if inputs == "x":
            g = np.linspace(*np.quantile(np.r_[Xtr[:, 0], Xva[:, 0]], [0.005, 0.995]), 200)[:, None]
            fh, se = m_fin.predict(g), m_fin.se(g)
            pd.DataFrame({"x": g[:, 0], "fhat": fh, "se": se if se is not None else np.nan}).to_csv(out / "curve_grid.csv", index=False)
            xs, ms_, ss = _binned(np.r_[Xtr[:, 0], Xva[:, 0]], np.r_[y_ev_tr, y_ev_va], 40)
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            ax.errorbar(xs, ms_, yerr=ss, fmt=".", color="grey", alpha=0.8, label="binned means +- SE (train+val)")
            ax.plot(g[:, 0], fh, "r-", lw=2, label=f"{name} (df {m_fin.df:.1f})")
            if se is not None and task == "regression":
                ax.fill_between(g[:, 0], fh - 2 * se, fh + 2 * se, color="red", alpha=0.2, label="+-2 pointwise SE")
            ax.set_xlabel(prep.x_label); ax.set_ylabel("logerror" if task == "regression" else "P(large miss)"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(out / "fit_curve.png", dpi=120); plt.close(fig)
        else:
            res = prep.y_test - p_te
            I, pI = moran_i(Xte, res)
            summary["moran_I_test_residuals"] = {"I": I, "perm_p": pI}
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
            s = np.random.RandomState(0).choice(len(Xte), min(20000, len(Xte)), replace=False)
            h1 = ax[0].hexbin(Xte[s, 1], Xte[s, 0], C=p_te[s], gridsize=40, reduce_C_function=np.mean, cmap="coolwarm"); fig.colorbar(h1, ax=ax[0]); ax[0].set_title("fitted surface (test rows)")
            h2 = ax[1].hexbin(Xte[s, 1], Xte[s, 0], C=res[s], gridsize=40, reduce_C_function=np.mean, cmap="coolwarm"); fig.colorbar(h2, ax=ax[1]); ax[1].set_title(f"mean test residual (Moran I = {I:.3f}, p = {pI:.2f})")
            for a in ax:
                a.set_xlabel("longitude (z)"); a.set_ylabel("latitude (z)")
            fig.tight_layout(); fig.savefig(out / "fit_surface.png", dpi=120); plt.close(fig)
        if task == "logistic":
            cal = calibration_table(prep.b_test, p_te); cal.to_csv(out / "calibration_test.csv", index=False)
            fig, ax = plt.subplots(figsize=(4.8, 4.5))
            ax.plot(cal["mean_pred"], cal["frac_pos"], "o-"); mx = max(cal["mean_pred"].max(), cal["frac_pos"].max()) * 1.05
            ax.plot([0, mx], [0, mx], "k:"); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed rate"); ax.set_title(f"{name} calibration (test)")
            fig.tight_layout(); fig.savefig(out / "calibration_test.png", dpi=120); plt.close(fig)
        if extra:
            extra({"prep": prep, "out": out, "curve": curve, "summary": summary, "grid": grid, "i_sel": i_sel, "Xtr": Xtr, "Xva": Xva, "Xte": Xte,
                   "y_fit_tr": y_fit_tr, "y_ev_tr": y_ev_tr, "fit_path": fit_path, "cfg": cfg, "m_fin": m_fin, "models": models, "log": log,
                   "folds": folds})
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
        (out / "test_metrics.json").write_text(json.dumps(_jsonable(test), indent=2))
        (out / "validation_metrics.json").write_text(json.dumps(_jsonable(val), indent=2))
        if task == "regression":
            log.info("VALIDATION MSE %.6f | TEST MSE %.6f +- %.6f (mean-baseline %.6f) | %.1fs", val["mse"], test["mse"], test["mse_se"],
                     test["baseline_train_mean_mse"], time.time() - t0)
        else:
            log.info("TEST log-loss %.5f +- %.5f (null %.5f) | AUC %.4f | %.1fs", test["log_loss"], test["log_loss_se"], test["null_log_loss"], test["auc"], time.time() - t0)
        prog.tick("done")
        return summary
    except Exception:
        log.error("FAILED\n%s", traceback.format_exc())
        raise


def method_main(name, run_fn):
    ap = argparse.ArgumentParser(description=f"Standalone run of '{name}'")
    ap.add_argument("--data", default=None, help="zillow.csv (only needed if --prepared does not exist yet)")
    ap.add_argument("--prepared", default="results/prepared")
    ap.add_argument("--results", default="results")
    ap.add_argument("--predictor", default=None)
    ap.add_argument("--cv-folds", type=int, default=None)
    a = ap.parse_args()
    prep_dir = Path(a.prepared)
    cfg_path = prep_dir / "run_config.json"
    cfg = RunConfig.from_json(cfg_path) if cfg_path.exists() else RunConfig()
    if a.predictor:
        cfg.predictor = a.predictor
    if a.cv_folds:
        cfg.cv_folds = a.cv_folds
    if not (prep_dir / "prepared_meta.json").exists():
        if not a.data:
            raise SystemExit("prepared data not found - pass --data zillow.csv to build it")
        import prepare
        prepare.build_prepared(a.data, prep_dir, cfg)
    run_fn(prep_dir, Path(a.results) / name, cfg, Progress(name, print_progress), console=True)
