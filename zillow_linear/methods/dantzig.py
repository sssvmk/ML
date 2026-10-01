"""
Dantzig selector.  Objective: min ||b||_1  subject to  ||X'(y - Xb)||_inf <= s   (solved as a linear program, HiGHS).
Metrics: 10-fold CV MSE vs the tuning bound (one-SE rule).
Runs on a candidate pool of `cfg.dantzig_pool` columns (greedy forward selection), re-selected inside every CV fold
and in every refit. Bound is expressed relative to its largest useful value: t = s / max|X'y| in (0, 1].
"""
import _bootstrap  # noqa: F401
import numpy as np
from scipy.optimize import linprog

import common
from common import Path_, forward_pool

NAME = "dantzig"


def make_grid(bundle, prep, cfg):
    return {"pool": int(cfg.dantzig_pool), "t": np.geomspace(1.0, 1e-3, 40)}


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    p, n = len(cc), st.n
    pool = forward_pool(Gc, cc, grid["pool"])
    m = len(pool)
    H, cn = Gc[np.ix_(pool, pool)] / n, cc[pool] / n
    smax = np.abs(cn).max()
    g = cn / smax                                           # beta = smax * theta
    A_ub = np.block([[H, -H], [-H, H]])
    obj = np.ones(2 * m)
    B = np.zeros((p, len(grid["t"])))
    prev = np.zeros(m)
    for l, t in enumerate(grid["t"]):
        res = linprog(obj, A_ub=A_ub, b_ub=np.concatenate([g + t, t - g]), bounds=(0, None), method="highs")
        if res.status == 0:
            prev = res.x[:m] - res.x[m:]
        B[pool, l] = smax * prev
    return Path_(B, ybar - xbar @ B, -np.log(grid["t"]), t=grid["t"], pool_size=np.full(len(grid["t"]), m))


def extra_curve(curve, path, bundle):
    curve["comp_plot"] = curve["t"]
    return curve


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="bound t = s / max|X'y|  (small = complex)", comp_log=True,
                                  extra_curve=extra_curve, notes=f"LP on a pool of {cfg.dantzig_pool} columns.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
