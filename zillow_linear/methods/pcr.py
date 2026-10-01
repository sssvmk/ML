"""
Principal Components Regression.  Objective: RSS on the first M principal components of the standardised inputs.
Metrics: 10-fold CV MSE vs number of components M (one-SE rule).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_

NAME = "pcr"


def make_grid(bundle, prep, cfg):
    return None


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    p = len(cc)
    w, V = np.linalg.eigh(Gc)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    keep = w > w[0] * 1e-9
    w, V = w[keep], V[:, keep]
    z = (V.T @ cc) / w
    B = np.hstack([np.zeros((p, 1)), np.cumsum(V * z[None, :], axis=1)])
    return Path_(B, ybar - xbar @ B, np.arange(B.shape[1], dtype=float))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="number of components M", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
