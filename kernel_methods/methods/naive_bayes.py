"""
Naive Bayes (ESLII 6.6.3) - CLASSIFICATION on ALL 200 Santander features (rank-gaussed).
Loss: class-conditional log-likelihood with independent features, log f_j(X) = sum_k log f_jk(X_k) (6.26); each marginal f_jk is a 1-D Gaussian kernel density
(binned on a fine grid and Gaussian-filtered; bandwidth = c * sd * n_j^(-1/5)). Log-odds = log prior odds + sum_k [log f_1k - log f_0k]  (6.27: additive in the features).
c is the smoothing parameter (CV log-loss, one-SE rule).
Metrics: test error +- SE, log-loss, AUC, calibration, plus a comparison with logistic regression and a GAM-style logistic regression (spline expansion per feature).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d

import common

NAME, TASK, INPUT = "naive_bayes", "classification", "x200"
CS = [0.25, 0.5, 1.0, 2.0, 4.0]
LO, HI, NB = -6.0, 6.0, 2400
BW = (HI - LO) / NB


def make_grid(cfg, prep, Xs, ys):
    return [{"c": c} for c in CS]


def complexity(hp):
    return 1.0 / hp["c"]


def _logdens(Xc, c):
    n, p = Xc.shape
    idx = np.clip(((Xc - LO) / BW).astype(int), 0, NB - 1)
    sd = Xc.std(0) * n ** (-0.2) * c
    out = np.empty((p, NB))
    for j in range(p):
        cnt = np.bincount(idx[:, j], minlength=NB).astype(float)
        out[j] = np.log(gaussian_filter1d(cnt, max(sd[j] / BW, 0.3), mode="constant") / (n * BW) + 1e-12)
    return out


def _gather(tab, Xq):
    pos = (np.clip(Xq, LO, HI - 1e-9) - LO) / BW - 0.5
    i0 = np.clip(np.floor(pos).astype(int), 0, NB - 2)
    fr = np.clip(pos - i0, 0, 1)
    cols = np.arange(tab.shape[0])[None, :]
    return tab[cols, i0] * (1 - fr) + tab[cols, i0 + 1] * fr


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    prior = np.log(yref.mean() / (1 - yref.mean()))
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        diff = _logdens(Xref[yref == 1], hp["c"]) - _logdens(Xref[yref == 0], hp["c"])
        contrib = np.concatenate([_gather(diff, Xq[a:a + 20000]).sum(1) for a in range(0, len(Xq), 20000)])
        out[h] = common.sigmoid(np.clip(contrib + prior, -30, 30))
        if ctx.get("final"):
            ctx["store"]["diff_tables"] = diff
    return out


def extra(ctx):
    cfg = ctx["cfg"]
    if cfg.nb_compare_rows <= 0:
        return
    from sklearn.linear_model import LogisticRegression
    from scipy.interpolate import BSpline
    X_tr, y_tr, X_va, y_va, X_te, y_te = ctx["X_tr"], ctx["y_tr"], ctx["X_va"], ctx["y_va"], ctx["X_te"], ctx["y_te"]
    rng = np.random.RandomState(cfg.seed)
    sub = np.sort(rng.choice(len(y_tr), min(cfg.nb_compare_rows, len(y_tr)), replace=False))
    res = {"rows_used_for_comparison_fits": int(len(sub)), "naive_bayes": ctx["summary"]["test"]}

    def evaluate(model_p_va, model_p_te):
        thr = common.youden_threshold(y_va, common.clip_prob(model_p_va))
        return common.clf_metrics(y_te, common.clip_prob(model_p_te), thr)
    lr = LogisticRegression(C=1.0, max_iter=200).fit(X_tr[sub], y_tr[sub])
    res["logistic_regression"] = evaluate(lr.predict_proba(X_va)[:, 1], lr.predict_proba(X_te)[:, 1])
    qs = np.quantile(X_tr[sub], [0.1, 0.35, 0.65, 0.9], axis=0)

    def expand(X):
        cols = []
        for j in range(X.shape[1]):
            t = np.r_[[-5.0] * 4, qs[:, j], [5.0] * 4]
            cols.append(BSpline.design_matrix(np.clip(X[:, j], -5.0, 5.0), t, 3).toarray()[:, 1:].astype(np.float32))
        return np.hstack(cols)
    Btr = expand(X_tr[sub])
    gam = LogisticRegression(C=0.5, max_iter=150).fit(Btr, y_tr[sub])
    res["gam_logistic_spline_expansion"] = evaluate(gam.predict_proba(expand(X_va))[:, 1], gam.predict_proba(expand(X_te))[:, 1])
    res["note"] = "logistic / GAM fitted on a training subsample (rows_used_for_comparison_fits); naive Bayes used all training rows. GAM = logistic regression on a cubic B-spline basis per feature (additive, like (6.27))."
    (ctx["out"] / "nb_vs_logistic_gam.json").write_text(json.dumps(common._jsonable(res), indent=2))
    ctx["summary"]["comparison"] = {k: {m: v[m] for m in ("log_loss", "auc", "error")} for k, v in res.items() if isinstance(v, dict)}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    complexity=complexity, extra=extra, notes="1-D KDE marginals on a 2400-bin grid with linear interpolation.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
