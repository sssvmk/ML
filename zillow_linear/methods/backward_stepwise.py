"""
Backward stepwise selection.  Greedily drop the predictor with the smallest Z-score (= least increase in RSS).
Metrics: 10-fold CV MSE vs model size k (one-SE rule), AIC.
Runs on ALL engineered features (requires N > p). Uses O(p^2) rank-one downdates of (X'X)^-1.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_, info_criteria

NAME = "backward_stepwise"


def make_grid(bundle, prep, cfg):
    return None


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    p = len(cc)
    jitter = 1e-7 * np.mean(np.diag(Gc))                   # keeps exactly collinear columns invertible (they drop first)
    P = np.linalg.inv(Gc + jitter * np.eye(p))
    act = np.arange(p)
    beta = P @ cc
    B = np.zeros((p, p + 1))
    B[:, p] = beta
    for size in range(p, 1, -1):
        # Z_j^2 = b_j^2 / (sigma^2 * P_jj); sigma^2 is common to all j, so minimise b_j^2 / P_jj
        jj = int(np.argmin(beta ** 2 / np.diag(P)))
        keep = np.arange(len(act)) != jj
        P = P[np.ix_(keep, keep)] - np.outer(P[keep, jj], P[jj, keep]) / P[jj, jj]
        act = act[keep]
        beta = P @ cc[act]
        B[act, size - 1] = beta
    return Path_(B, ybar - xbar @ B, np.arange(p + 1, dtype=float))


def extra_curve(curve, path, bundle):
    ic = info_criteria(curve["train_rss"].values, bundle.tr_fit.n, path.comp + 1, 0.0)
    curve["aic"], curve["bic"] = ic["aic"], ic["bic"]
    return curve


def final_extras(beta, b0, st, bundle):
    k = int((np.abs(beta) > 1e-12).sum())
    rss = st.sse(beta, b0)
    ic = info_criteria(rss, st.n, k + 1, 0.0)
    return {"k": k, "training_rss": rss, "AIC": float(ic["aic"]), "BIC": float(ic["bic"])}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="model size k", extra_curve=extra_curve, final_extras=final_extras,
                                  notes="Drop rule: smallest Z-score == smallest RSS increase (common sigma^2).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
