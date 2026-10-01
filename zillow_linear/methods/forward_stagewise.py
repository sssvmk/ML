"""
Forward stagewise regression (incremental, epsilon-steps; ESLII Algorithm 3.8).
Objective: at each step add +-epsilon to the coefficient of the predictor most correlated with the current residual
(greedy RSS reduction in tiny steps). Merged with 'incremental forward stagewise' as agreed.
Metrics: 10-fold CV MSE vs step count (one-SE rule), AIC (df approximated by the number of non-zero coefficients).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_, info_criteria

NAME = "forward_stagewise"


def make_grid(bundle, prep, cfg):
    Gc, cc, yyc, xbar, ybar = bundle.tr_fit.centered()
    sd_y = np.sqrt(yyc / bundle.tr_fit.n)
    T = int(cfg.fs_steps)
    steps = np.unique(np.concatenate([[0], np.round(np.geomspace(1, T, 300)).astype(int)]))
    return {"eps": cfg.fs_eps_frac * sd_y, "T": T, "steps": steps}


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    n = st.n
    Gn, cn = Gc / n, cc / n
    sd = np.sqrt(np.maximum(np.diag(Gn), 1e-12))
    p = len(cn)
    beta, r = np.zeros(p), cn.copy()                       # r = X'(y - Xb)/n
    eps, T = grid["eps"], grid["T"]
    wanted = set(int(s) for s in grid["steps"])
    cols, steps = [beta.copy()], [0]
    for t in range(1, T + 1):
        j = int(np.argmax(np.abs(r) / sd))
        s = np.sign(r[j])
        beta[j] += s * eps
        r -= s * eps * Gn[j]
        if t in wanted:
            cols.append(beta.copy()); steps.append(t)
    B = np.array(cols).T
    return Path_(B, ybar - xbar @ B, np.array(steps, float))


def extra_curve(curve, path, bundle):
    ic = info_criteria(curve["train_rss"].values, bundle.tr_fit.n, curve["n_nonzero"].values + 1, 0.0)
    curve["aic"], curve["bic"] = ic["aic"], ic["bic"]
    return curve


def final_extras(beta, b0, st, bundle):
    k = int((np.abs(beta) > 1e-12).sum())
    rss = st.sse(beta, b0)
    return {"n_nonzero": k, "training_rss": rss, "AIC": float(info_criteria(rss, st.n, k + 1, 0.0)["aic"])}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="step count", comp_log=True, extra_curve=extra_curve, final_extras=final_extras,
                                  notes=f"epsilon = {cfg.fs_eps_frac} * sd(y); {cfg.fs_steps} steps; path snapshots log-spaced.",
                                  console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
