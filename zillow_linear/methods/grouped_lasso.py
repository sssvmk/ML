"""
Grouped lasso.  Objective: RSS + lam * sum_g sqrt(p_g) * ||b_g||_2
Groups: all one-hot dummies of one categorical column form a group; every numeric feature / flag / missing-indicator
is its own group (as agreed). Solver: accelerated proximal gradient (FISTA with restart) on the Gram matrix.
Metrics: 10-fold CV MSE vs lambda (one-SE rule).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_

NAME = "grouped_lasso"


def make_grid(bundle, prep, cfg):
    gid = prep.group_ids
    _, inv = np.unique(gid, return_inverse=True)
    sizes = np.bincount(inv)
    Gc, cc, yyc, xbar, ybar = bundle.tr_fit.centered()
    cn = cc / bundle.tr_fit.n
    gnorm = np.sqrt(np.bincount(inv, weights=cn ** 2, minlength=len(sizes)))
    lam_max = 2 * bundle.tr_fit.n * np.max(gnorm / np.sqrt(sizes))
    return {"inv": inv, "sqrtp": np.sqrt(sizes), "ng": len(sizes),
            "lams": lam_max * np.geomspace(1.0, 1e-3, cfg.n_lambda), "max_iter": cfg.group_max_iter}


def _prox(z, thr, inv, sqrtp, ng):
    norms = np.sqrt(np.bincount(inv, weights=z * z, minlength=ng))
    scale = np.maximum(0.0, 1.0 - thr * sqrtp / np.maximum(norms, 1e-300))
    return z * scale[inv]


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    n = st.n
    Gn, cn = Gc / n, cc / n
    Lc = float(np.linalg.eigvalsh(Gn)[-1])
    step = 1.0 / Lc
    inv, sqrtp, ng = grid["inv"], grid["sqrtp"], grid["ng"]
    lams = grid["lams"]
    p = len(cn)
    beta = np.zeros(p)
    B = np.empty((p, len(lams)))
    ngroups_active = np.zeros(len(lams))
    for l, lam in enumerate(lams):
        lg = lam / (2 * n)                                   # RSS + lam*pen  ->  (1/2n)RSS + lg*pen
        y, tk = beta.copy(), 1.0
        for it in range(grid["max_iter"]):
            new = _prox(y - step * (Gn @ y - cn), step * lg, inv, sqrtp, ng)
            if (y - new) @ (new - beta) > 0:                 # gradient-based restart
                tk, y = 1.0, new.copy()
                beta = new
                continue
            tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            y = new + ((tk - 1) / tk1) * (new - beta)
            done = np.max(np.abs(new - beta)) < 1e-7 * (1.0 + np.max(np.abs(new)))
            beta, tk = new, tk1
            if done:
                break
        B[:, l] = beta
        ngroups_active[l] = (np.bincount(inv, weights=np.abs(beta), minlength=ng) > 1e-12).sum()
    return Path_(B, ybar - xbar @ B, -np.log(lams), lam=lams, n_groups_nonzero=ngroups_active)


def extra_curve(curve, path, bundle):
    curve["comp_plot"] = curve["lam"]
    return curve


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="lambda (small = complex)", comp_log=True, extra_curve=extra_curve,
                                  notes="Reference level of each categorical is dropped in features.py, so each group holds the remaining dummies.",
                                  console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
