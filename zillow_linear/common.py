"""
common.py - shared infrastructure for the Zillow linear-methods study.

Everything the 12 method modules need that is NOT method-specific lives here:
  * RunConfig                 - run settings (CV folds, scheme, refit policy, pools ...)
  * Prepared                  - memory-mapped access to the engineered train/val/test arrays
  * Stats / build_stats       - sufficient statistics (X'X, X'y, y'y ...) so every linear
                                method can be fit AND cross-validated from Gram matrices
                                (no repeated passes over the big X matrix)
  * get_bundle                - cached per-fold statistics for 10-fold CV
  * run_path_method           - the common train -> 10-fold CV -> one-SE rule -> validation ->
                                refit -> test -> logging/plots flow. Each method module only
                                supplies its own `fit_path` (and optional hooks).
  * metrics (MSE +- SE, AIC/BIC/Cp, GCV), plotting, logging, CLI helper

Conventions
  * Features are standardised on the TRAIN split only (see features.py).
  * The model target used for FITTING is logerror clipped at the train 1st/99th percentile.
    EVERYTHING that is evaluated (CV MSE, validation MSE, test MSE) uses the RAW logerror.
  * The intercept is never penalised: all methods work on centred statistics.
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

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    cv_folds: int = 10
    cv_scheme: str = "blocked"      # "blocked" (contiguous time blocks) | "random" (shuffled K-fold)
    seed: int = 42
    refit: str = "train_val"        # final model fit on "train" only, or on "train_val"
    n_lambda: int = 60              # length of lambda grids (ridge / lasso / enet / grouped lasso / dantzig)
    subset_pool: int = 20           # best-subset: candidate pool size (exact search up to 22)
    dantzig_pool: int = 30          # Dantzig selector: candidate pool size
    fs_steps: int = 5000            # forward stagewise: number of epsilon-steps
    fs_eps_frac: float = 0.003      # forward stagewise: epsilon = frac * sd(y)
    pls_max_comp: int = 50          # PLS: max number of directions
    enet_alphas: tuple = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)   # ESLII (3.91): alpha weights the L2 term
    group_max_iter: int = 2000      # grouped lasso FISTA iterations per lambda

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        if "enet_alphas" in d:
            d["enet_alphas"] = tuple(d["enet_alphas"])
        return cls(**d)


# --------------------------------------------------------------------------------------
# logging / progress
# --------------------------------------------------------------------------------------
def get_logger(out_dir, name, console=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"zillow.{name}.{out_dir}")
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
    """Tiny progress reporter. `cb(name, done, total, msg)` is supplied by the orchestrator."""

    def __init__(self, name, cb=None, total=1):
        self.name, self.cb, self.total, self.done = name, cb, total, 0

    def set_total(self, total):
        self.total = total

    def tick(self, msg="", n=1):
        self.done = min(self.done + n, self.total)
        if self.cb:
            self.cb(self.name, self.done, self.total, msg)


def print_progress(name, done, total, msg):
    print(f"[{name}] {done}/{total} {msg}", flush=True)


# --------------------------------------------------------------------------------------
# prepared data
# --------------------------------------------------------------------------------------
class Prepared:
    """Engineered arrays written by features.build_prepared (memory-mapped, shared across processes)."""

    def __init__(self, prep_dir):
        d = self.dir = Path(prep_dir)
        self.meta = json.loads((d / "feature_meta.json").read_text())
        self.feature_names = self.meta["feature_names"]
        self.group_ids = np.asarray(self.meta["group_ids"], dtype=int)
        self.p = len(self.feature_names)
        for s in ("train", "val", "test"):
            setattr(self, f"X_{s}", np.load(d / f"X_{s}.npy", mmap_mode="r"))
            setattr(self, f"y_{s}", np.load(d / f"y_raw_{s}.npy"))
        self.y_train_fit = np.load(d / "y_fit_train.npy")
        self.y_val_fit = np.load(d / "y_fit_val.npy")
        self.n_train, self.n_val, self.n_test = len(self.y_train), len(self.y_val), len(self.y_test)
        self.baseline_test_mse = float(np.mean((self.y_test - self.y_train.mean()) ** 2))


# --------------------------------------------------------------------------------------
# sufficient statistics
# --------------------------------------------------------------------------------------
class Stats:
    """Sufficient statistics of (X, y):  n, sum x, X'X, X'y, sum y, y'y."""
    __slots__ = ("n", "sx", "G", "c", "sy", "yy")

    def __init__(self, n, sx, G, c, sy, yy):
        self.n, self.sx, self.G, self.c, self.sy, self.yy = n, sx, G, c, sy, yy

    def __add__(self, o):
        return Stats(self.n + o.n, self.sx + o.sx, self.G + o.G, self.c + o.c, self.sy + o.sy, self.yy + o.yy)

    def __sub__(self, o):
        return Stats(self.n - o.n, self.sx - o.sx, self.G - o.G, self.c - o.c, self.sy - o.sy, self.yy - o.yy)

    def centered(self):
        """Return (Gc, cc, yyc, xbar, ybar): centred Gram, X'y, y'y and the means."""
        n = self.n
        xbar, ybar = self.sx / n, self.sy / n
        Gc = self.G - n * np.outer(xbar, xbar)
        cc = self.c - n * xbar * ybar
        yyc = self.yy - n * ybar * ybar
        return Gc, cc, yyc, xbar, ybar

    def sse(self, B, b0):
        """Sum of squared errors of y ~ b0 + x'B on the rows summarised here. B: (p,) or (p,L)."""
        B = np.asarray(B, dtype=float)
        single = B.ndim == 1
        if single:
            B = B[:, None]
        b0 = np.broadcast_to(np.asarray(b0, dtype=float), (B.shape[1],))
        quad = np.einsum("il,il->l", B, self.G @ B)
        out = self.yy - 2 * b0 * self.sy - 2 * (self.c @ B) + self.n * b0 ** 2 + 2 * b0 * (self.sx @ B) + quad
        return float(out[0]) if single else out


def _chunks(rows, n_total, chunk):
    if rows is None:
        rows = slice(0, n_total)
    if isinstance(rows, slice):
        a, b = rows.start, rows.stop
        for s in range(a, b, chunk):
            yield slice(s, min(s + chunk, b))
    else:
        for part in np.array_split(rows, max(1, int(np.ceil(len(rows) / chunk)))):
            yield part


def build_stats(X, ys, rows=None, chunk=20000):
    """One pass over X[rows]. ys: dict name -> full-length target. Returns dict name -> Stats (G shared)."""
    p = X.shape[1]
    G, sx, n = np.zeros((p, p)), np.zeros(p), 0
    acc = {k: [np.zeros(p), 0.0, 0.0] for k in ys}
    for sl in _chunks(rows, X.shape[0], chunk):
        xb = np.asarray(X[sl], dtype=np.float64)
        G += xb.T @ xb
        sx += xb.sum(0)
        n += xb.shape[0]
        for k, y in ys.items():
            yb = np.asarray(y[sl], dtype=np.float64)
            acc[k][0] += xb.T @ yb
            acc[k][1] += yb.sum()
            acc[k][2] += float(yb @ yb)
    return {k: Stats(n, sx.copy(), G, a[0], a[1], a[2]) for k, a in acc.items()}


def make_folds(n, k, scheme, seed):
    if scheme == "blocked":
        e = np.linspace(0, n, k + 1).astype(int)
        return [slice(int(e[i]), int(e[i + 1])) for i in range(k)]
    perm = np.random.RandomState(seed).permutation(n)
    return [np.sort(a) for a in np.array_split(perm, k)]


class Bundle:
    """Cached statistics: tr_* (all train), folds_* (per CV fold), val_* ; *_fit use clipped y, *_raw raw y."""
    pass


def get_bundle(prep: Prepared, cfg: RunConfig) -> Bundle:
    path = prep.dir / f"bundle_{cfg.cv_scheme}_{cfg.cv_folds}_{cfg.seed}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    b = Bundle()
    folds = make_folds(prep.n_train, cfg.cv_folds, cfg.cv_scheme, cfg.seed)
    ys = {"fit": prep.y_train_fit, "raw": prep.y_train}
    ff, fr = [], []
    for rows in folds:
        d = build_stats(prep.X_train, ys, rows)
        ff.append(d["fit"])
        fr.append(d["raw"])
    b.folds_fit, b.folds_raw = ff, fr
    b.tr_fit = sum(ff[1:], ff[0])
    b.tr_raw = sum(fr[1:], fr[0])
    d = build_stats(prep.X_val, {"fit": prep.y_val_fit, "raw": prep.y_val})
    b.val_fit, b.val_raw = d["fit"], d["raw"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(b, f, protocol=4)
    tmp.replace(path)
    return b


# --------------------------------------------------------------------------------------
# small numerical helpers
# --------------------------------------------------------------------------------------
class Path_:
    """A coefficient path: B (p x L), intercepts b0 (L,), complexity ordinal comp (L,) (increasing = more complex)."""

    def __init__(self, B, b0, comp, **extra):
        self.B = np.asarray(B, float)
        self.b0 = np.asarray(b0, float)
        self.comp = np.asarray(comp, float)
        self.extra = {k: np.asarray(v) for k, v in extra.items()}

    @property
    def L(self):
        return self.B.shape[1]


def ols_solve(Gc, cc, rel_tol=1e-10):
    """Minimum-norm least squares from a (possibly rank-deficient) centred Gram matrix. Returns (beta, rank)."""
    w, V = np.linalg.eigh(Gc)
    keep = w > w.max() * rel_tol * len(w)
    beta = V[:, keep] @ ((V[:, keep].T @ cc) / w[keep])
    return beta, int(keep.sum())


def sigma2_full(st: Stats):
    """Noise variance from the full OLS model (used by Cp):  RSS / (n - rank - 1)."""
    Gc, cc, yyc, xbar, ybar = st.centered()
    beta, rank = ols_solve(Gc, cc)
    rss = yyc - 2 * beta @ cc + beta @ Gc @ beta
    return float(rss / max(st.n - rank - 1, 1))


def info_criteria(rss, n, df, sigma2):
    """Gaussian AIC / BIC / Mallows Cp (all from training residuals). rss, df may be arrays."""
    rss, df = np.asarray(rss, float), np.asarray(df, float)
    base = n * np.log(np.maximum(rss, 1e-300) / n)
    return {"aic": base + 2 * df, "bic": base + df * np.log(n), "cp": rss / n + 2 * df * sigma2 / n}


def one_se(mean, se, comp):
    """ESLII one-standard-error rule: least complex model whose CV error <= min + SE(min). Returns (i_1se, i_min)."""
    mean = np.asarray(mean, float)
    imin = int(np.nanargmin(mean))
    ok = np.where(mean <= mean[imin] + se[imin])[0]
    return int(ok[np.argmin(np.asarray(comp)[ok])]), imin


def predict(X, beta, b0, chunk=50000):
    out = np.empty(X.shape[0])
    for a in range(0, X.shape[0], chunk):
        out[a:a + chunk] = np.asarray(X[a:a + chunk], dtype=np.float64) @ beta + b0
    return out


def eval_predictions(y, yhat):
    e2 = (y - yhat) ** 2
    n = len(y)
    mse = float(e2.mean())
    return {"n": int(n), "mse": mse, "mse_se": float(e2.std(ddof=1) / np.sqrt(n)), "rmse": float(np.sqrt(mse)),
            "mae": float(np.abs(y - yhat).mean()), "r2": float(1 - e2.sum() / ((y - y.mean()) ** 2).sum())}


def forward_pool(Gc, cc, k, tol=1e-6):
    """Greedy forward selection of k well-conditioned candidate columns (maximum RSS reduction each step)."""
    p = len(cc)
    diag = np.diag(Gc)
    S = []
    for _ in range(min(k, p)):
        if S:
            A = np.linalg.inv(Gc[np.ix_(S, S)])
            U = Gc[S, :]
            denom = diag - np.einsum("ij,ij->j", U, A @ U)
            num = (cc - U.T @ (A @ cc[S])) ** 2
        else:
            denom, num = diag.copy(), cc ** 2
        score = np.where(denom > tol * diag, num / np.maximum(denom, 1e-300), -np.inf)
        score[S] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        S.append(j)
    return np.array(S, dtype=int)


def align(v, L):
    v = np.asarray(v, float)
    if len(v) >= L:
        return v[:L]
    return np.concatenate([v, np.full(L - len(v), v[-1])])


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def plot_cv_curve(curve, i_1se, i_min, xlabel, path, xlog=False, title=""):
    x = curve["comp_plot"].values if "comp_plot" in curve else curve["comp"].values
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(x, curve["cv_mse"], yerr=curve["cv_se"], fmt="o-", ms=3, lw=1, capsize=2, label="10-fold CV MSE (train, raw y) +- SE")
    if "val_mse" in curve:
        ax.plot(x, curve["val_mse"], "s--", ms=3, lw=1, color="tab:green", label="validation MSE (Mar-Jul 2017)")
    ax.axvline(x[i_min], color="tab:red", ls=":", label="CV minimum")
    ax.axvline(x[i_1se], color="tab:purple", ls="--", label="one-SE choice")
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE of logerror")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coefficients(beta, names, path, top=30):
    order = np.argsort(-np.abs(beta))[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(order) + 1.5))
    ax.barh([names[i] for i in order], beta[order])
    ax.set_title("Largest standardised coefficients (final model)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


# --------------------------------------------------------------------------------------
# the common method flow
# --------------------------------------------------------------------------------------
def run_path_method(name, prep_dir, out_dir, cfg, progress, *, make_grid, fit_path, comp_label,
                    comp_log=False, extra_curve=None, final_extras=None, select=None,
                    fit_one=None, extra_plot=None, notes="", console=False):
    """
    Train -> 10-fold CV (one-SE rule) -> validation -> refit -> test -> log/plot.

    make_grid(bundle, prep, cfg)           -> any object describing the tuning grid
    fit_path(Stats_fit, grid, cfg)         -> Path_   (called on each CV-training fold, on train, on train+val)
    extra_curve(curve_df, path, bundle)    -> curve_df with additional metric columns (AIC, GCV, df ...)
    final_extras(beta, b0, stats, bundle)  -> dict of extra metrics for the final model (e.g. AIC/BIC/Cp)
    select(curve_df, path)                 -> (i_1se, i_min)           [default: one-SE rule on `comp`]
    fit_one(stats, grid, cfg, i, path)     -> (beta, b0)               [default: refit path & pick same comp]
    extra_plot(out_dir, curve_df, summary) -> None
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out, name, console=console)
    prog = progress or Progress(name)
    K = cfg.cv_folds
    prog.set_total(K + 5)
    try:
        prep = Prepared(prep_dir)
        bundle = get_bundle(prep, cfg)
        log.info("%s | n_train=%d n_val=%d n_test=%d p=%d | cv=%d-fold (%s) refit=%s", name, prep.n_train,
                 prep.n_val, prep.n_test, prep.p, K, cfg.cv_scheme, cfg.refit)
        prog.tick("data ready")

        grid = make_grid(bundle, prep, cfg)
        path = fit_path(bundle.tr_fit, grid, cfg)
        L = path.L
        log.info("fit on full train: path with %d points", L)
        prog.tick("fit on train")

        # ---- 10-fold CV (evaluated on RAW logerror) ----
        fm = np.empty((K, L))
        for k in range(K):
            pk = fit_path(bundle.tr_fit - bundle.folds_fit[k], grid, cfg)
            fm[k] = align(bundle.folds_raw[k].sse(pk.B, pk.b0) / bundle.folds_raw[k].n, L)
            prog.tick(f"CV fold {k + 1}/{K}")
        cv_mean, cv_se = fm.mean(0), fm.std(0, ddof=1) / np.sqrt(K)

        # ---- validation curve (train-only fits, Mar-Jul 2017) ----
        val_mse = bundle.val_raw.sse(path.B, path.b0) / bundle.val_raw.n
        curve = pd.DataFrame({"comp": path.comp, "cv_mse": cv_mean, "cv_se": cv_se, "val_mse": val_mse,
                              "train_rss": bundle.tr_fit.sse(path.B, path.b0),
                              "n_nonzero": (np.abs(path.B) > 1e-12).sum(0)})
        for k, v in path.extra.items():
            curve[k] = v
        if extra_curve:
            curve = extra_curve(curve, path, bundle)
        i_1se, i_min = select(curve, path) if select else one_se(cv_mean, cv_se, path.comp)
        for k in range(K):
            curve[f"cv_fold{k + 1}"] = fm[k]
        curve["is_cv_min"] = False
        curve["is_one_se"] = False
        curve.loc[i_min, "is_cv_min"] = True
        curve.loc[i_1se, "is_one_se"] = True
        curve.to_csv(out / "cv_curve.csv", index=False)
        log.info("CV minimum at index %d (comp=%.6g, CV MSE=%.6f); one-SE choice index %d (comp=%.6g, CV MSE=%.6f)",
                 i_min, path.comp[i_min], cv_mean[i_min], i_1se, path.comp[i_1se], cv_mean[i_1se])
        prog.tick("validation curve")

        # ---- final model ----
        beta_tr, b0_tr = path.B[:, i_1se], float(path.b0[i_1se])        # train-only model (for validation metrics)
        if cfg.refit == "train_val":
            st_final = bundle.tr_fit + bundle.val_fit
            if fit_one:
                beta_f, b0_f = fit_one(st_final, grid, cfg, i_1se, path)
            else:
                pf = fit_path(st_final, grid, cfg)
                j = int(np.argmin(np.abs(pf.comp - path.comp[i_1se])))
                beta_f, b0_f = pf.B[:, j], float(pf.b0[j])
        else:
            st_final, beta_f, b0_f = bundle.tr_fit, beta_tr, b0_tr
        prog.tick("final refit")

        # ---- validation + test (raw logerror) ----
        val = eval_predictions(prep.y_val, predict(prep.X_val, beta_tr, b0_tr))
        yhat_test = predict(prep.X_test, beta_f, b0_f)
        test = eval_predictions(prep.y_test, yhat_test)
        test["baseline_train_mean_mse"] = prep.baseline_test_mse
        test["rel_mse_vs_mean_baseline"] = test["mse"] / prep.baseline_test_mse
        extras = final_extras(beta_f, b0_f, st_final, bundle) if final_extras else {}
        sel = {"index": i_1se, "comp": float(path.comp[i_1se]), "cv_mse": float(cv_mean[i_1se]),
               "cv_se": float(cv_se[i_1se]), "n_nonzero_final": int((np.abs(beta_f) > 1e-12).sum()),
               "cv_min_index": i_min, "cv_min_comp": float(path.comp[i_min])}
        for col in ("aic", "bic", "cp", "gcv"):
            if col in curve:
                sel[f"argmin_{col}_comp"] = float(path.comp[int(np.nanargmin(curve[col].values))])
        for k, v in path.extra.items():
            sel[f"{k}_at_choice"] = float(np.asarray(v)[i_1se]) if np.ndim(v) == 1 else None
        summary = {"method": name, "complexity": comp_label, "selection": sel, "validation_train_only_model": val,
                   "test": test, "final_model_info_criteria_or_extras": extras, "notes": notes,
                   "config": asdict(cfg), "runtime_sec": round(time.time() - t0, 1)}

        # ---- write results ----
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
        (out / "test_metrics.json").write_text(json.dumps(_jsonable({"test": test, "extras": extras}), indent=2))
        (out / "validation_metrics.json").write_text(json.dumps(_jsonable({"model": "train-only, chosen by one-SE", **val}), indent=2))
        np.save(out / "test_predictions.npy", yhat_test.astype(np.float32))
        pd.DataFrame({"feature": prep.feature_names, "coef_standardised": beta_f,
                      "coef_train_only_model": beta_tr}).to_csv(out / "coefficients.csv", index=False)
        if L > 1:
            plot_cv_curve(curve.assign(comp_plot=curve["comp"] if "comp_plot" not in curve else curve["comp_plot"]),
                          i_1se, i_min, comp_label, out / "cv_curve.png", xlog=comp_log, title=name)
        plot_coefficients(beta_f, prep.feature_names, out / "coefficients_top.png")
        if extra_plot:
            extra_plot(out, curve, summary)
        log.info("VALIDATION (train-only model) MSE=%.6f +- %.6f | TEST MSE=%.6f +- %.6f | %.1fs", val["mse"],
                 val["mse_se"], test["mse"], test["mse_se"], time.time() - t0)
        prog.tick("done")
        return summary
    except Exception:
        log.error("FAILED\n%s", traceback.format_exc())
        raise


# --------------------------------------------------------------------------------------
# CLI helper so that every method file can be run on its own
# --------------------------------------------------------------------------------------
def method_main(name, run_fn):
    ap = argparse.ArgumentParser(description=f"Standalone run of '{name}'")
    ap.add_argument("--data", default=None, help="zillow.csv (only needed if --prepared does not exist yet)")
    ap.add_argument("--prepared", default="results/prepared")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--cv-scheme", choices=["blocked", "random"], default=None)
    ap.add_argument("--refit", choices=["train", "train_val"], default=None)
    a = ap.parse_args()
    prep_dir = Path(a.prepared)
    if not (prep_dir / "feature_meta.json").exists():
        if not a.data:
            raise SystemExit("prepared data not found - pass --data zillow.csv to build it")
        import features
        features.build_prepared(a.data, prep_dir, RunConfig())
    cfg_path = prep_dir / "run_config.json"
    cfg = RunConfig.from_json(cfg_path) if cfg_path.exists() else RunConfig()
    for k, v in (("cv_folds", a.cv_folds), ("cv_scheme", a.cv_scheme), ("refit", a.refit)):
        if v is not None:
            setattr(cfg, k, v)
    run_fn(prep_dir, Path(a.results) / name, cfg, Progress(name, print_progress), console=True)
