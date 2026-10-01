"""
Least Angle Regression (ESLII Algorithm 3.2, plain LAR - no lasso modification).
Objective: no explicit loss; each step moves the active coefficients along the equiangular direction until a new
variable is as correlated with the residual as the active ones.
Metrics: 10-fold CV MSE vs LAR step (one-SE rule); the L1 fraction s is logged next to each step.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_

NAME = "lar"


def make_grid(bundle, prep, cfg):
    return None


def lar_knots(Gc, cc):
    p = len(cc)
    beta = np.zeros(p)
    inA = np.zeros(p, bool)
    banned = np.zeros(p, bool)
    signs = np.zeros(p)
    active = []
    knots = [beta.copy()]
    c = cc.copy()
    while True:
        cand = np.where(~inA & ~banned)[0]
        if len(cand) == 0:
            break
        j = int(cand[np.argmax(np.abs(c[cand]))])
        if active:
            v = np.linalg.solve(Gc[np.ix_(active, active)], Gc[active, j])
            piv = Gc[j, j] - Gc[j, active] @ v
        else:
            piv = Gc[j, j]
        if piv < 1e-8 * Gc[j, j]:                           # (near-)collinear with the active set
            banned[j] = True
            continue
        active.append(j); inA[j] = True; signs[j] = np.sign(c[j]) if c[j] != 0 else 1.0
        A = np.array(active)
        s = signs[A]
        w = np.linalg.solve(Gc[np.ix_(A, A)], s)
        AA = 1.0 / np.sqrt(s @ w)
        d = AA * w
        a = Gc[:, A] @ d
        C = np.abs(c[A]).max()
        gam_end = C / AA
        gam = gam_end
        inact = np.where(~inA & ~banned)[0]
        if len(inact):
            ci, ai = c[inact], a[inact]
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.concatenate([(C - ci) / (AA - ai), (C + ci) / (AA + ai)])
            g = g[np.isfinite(g) & (g > 1e-12)]
            if len(g):
                gam = min(gam, g.min())
        beta[A] += gam * d
        c = cc - Gc @ beta
        knots.append(beta.copy())
    return np.array(knots).T


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    B = lar_knots(Gc, cc)
    l1 = np.abs(B).sum(0)
    return Path_(B, ybar - xbar @ B, np.arange(B.shape[1], dtype=float), l1_fraction=l1 / max(l1[-1], 1e-300))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="LAR step (variables entered)",
                                  notes="Plain LAR; (near-)collinear candidates are skipped (banned) when they add no new direction.",
                                  console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
