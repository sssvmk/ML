"""
Partial Least Squares (SIMPLS, single response, computed from the Gram matrix).
Objective: per direction maximise Var(X a) * Corr^2(y, X a); coefficients from RSS on the first M directions.
Metrics: 10-fold CV MSE vs number of directions (one-SE rule).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_

NAME = "pls"


def make_grid(bundle, prep, cfg):
    return int(cfg.pls_max_comp)


def fit_path(st, A, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    p = len(cc)
    A = min(A, p)
    R, Q, Vm = np.zeros((p, A)), np.zeros(A), np.zeros((p, A))
    S = cc.copy()
    used = 0
    for a in range(A):
        r = S.copy()
        tt = r @ Gc @ r
        if tt <= 1e-14 * max(Gc.trace(), 1.0) * max(r @ r, 1e-300):
            break
        r /= np.sqrt(tt)
        load = Gc @ r
        v = load - Vm[:, :a] @ (Vm[:, :a].T @ load) if a else load.copy()
        nv = np.linalg.norm(v)
        if nv < 1e-12 * max(np.linalg.norm(load), 1e-300):
            break
        v /= nv
        Vm[:, a], R[:, a], Q[a] = v, r, cc @ r
        S = S - v * (v @ S)
        used += 1
    B = np.hstack([np.zeros((p, 1)), np.cumsum(R[:, :used] * Q[None, :used], axis=1)])
    return Path_(B, ybar - xbar @ B, np.arange(B.shape[1], dtype=float))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="number of PLS directions", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
