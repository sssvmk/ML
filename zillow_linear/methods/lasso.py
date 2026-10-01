"""
Lasso.  Objective: RSS + lam * sum |b_j|.   Solver: pathwise coordinate descent (solvers.py).
Metrics: 10-fold CV MSE vs shrinkage factor s = |b(lam)|_1 / |b(smallest lam)|_1 (one-SE rule).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_
from solvers import enet_path_abs, lam_max_abs

NAME = "lasso"


def make_grid(bundle, prep, cfg):
    return lam_max_abs(bundle.tr_fit, 0.0) * np.geomspace(1.0, 1e-4, cfg.n_lambda)


def fit_path(st, lams, cfg):
    B, b0 = enet_path_abs(st, lams, 0.0)
    l1 = np.abs(B).sum(0)
    s = l1 / max(l1[-1], 1e-300)
    return Path_(B, b0, s, lam=lams, s=s)


def extra_curve(curve, path, bundle):
    curve["comp_plot"] = curve["s"]
    return curve


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="shrinkage factor s", extra_curve=extra_curve,
                                  notes="s is relative to the smallest-lambda solution on the full-train path.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
