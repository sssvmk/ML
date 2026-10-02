"""
Natural cubic splines (ESLII 5.2.1).  Objective: the same RSS as the regression spline, with the extra constraint that the fit is
LINEAR beyond the boundary knots.  Basis N_1 = 1, N_2 = x, N_{k+2} = d_k - d_{K-1}; K knots (data min, data max and K-2 quantiles) give df = K.
Metrics: test MSE +- SE, 10-fold CV MSE vs knots / df (one-SE rule), AIC / BIC / Cp, pointwise SE bands, and a comparison with the
cubic regression spline at the SAME df (CV MSE and pointwise SE at the boundary vs the middle).
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import common
from bases import ncs_knots, natural_cubic_basis, quantile_knots, trunc_power_basis
from common import ls_fit

NAME = "natural_cubic_splines"
KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def make_grid(prep, cfg, X, y):
    nu = len(np.unique(X[:, 0]))
    return [{"K": K} for K in KS if K + 1 <= max(nu // 4, 3)]


def fit_path(X, y, grid, cfg):
    xs, out = X[:, 0], []
    for hp in grid:
        kn = ncs_knots(xs, hp["K"])
        out.append(ls_fit(lambda Z, kn=kn: natural_cubic_basis(Z[:, 0], kn), X, y, info={"knots": kn.tolist()}))
    return out


def _rs_fit(X, y, df):
    xs = X[:, 0]
    kn = quantile_knots(xs, df - 4)
    return ls_fit(lambda Z, kn=kn: trunc_power_basis(Z[:, 0], kn, 3), X, y)


def extra(ctx):
    Xtr, y_fit, y_ev, folds, cv = ctx["Xtr"], ctx["y_fit_tr"], ctx["y_ev_tr"], ctx["folds"], ctx["curve"]
    rows = []
    for _, r in cv.iterrows():
        K = int(r["K"])
        if K < 4:
            continue
        fm = []
        for idx in folds:
            tr = np.ones(len(y_ev), bool); tr[idx] = False
            m = _rs_fit(Xtr[tr], y_fit[tr], K)
            fm.append(float(np.mean((y_ev[idx] - m.predict(Xtr[idx])) ** 2)))
        full = _rs_fit(Xtr, y_fit, K)
        g = np.quantile(Xtr[:, 0], [0.005, 0.5, 0.995])[:, None]
        ncs = ctx["models"][int(r.name)]
        s_n, s_r = ncs.se(g), full.se(g)
        rows.append({"df": K, "cv_mse_natural": r["cv_mse"], "cv_mse_regression_spline": float(np.mean(fm)),
                     "se_ratio_regression_over_natural_left_boundary": float(s_r[0] / s_n[0]),
                     "se_ratio_regression_over_natural_middle": float(s_r[1] / s_n[1]),
                     "se_ratio_regression_over_natural_right_boundary": float(s_r[2] / s_n[2])})
    pd.DataFrame(rows).to_csv(ctx["out"] / "matched_df_vs_regression_spline.csv", index=False)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, extra=extra,
                               notes="Boundary knots at the data min/max; linear beyond.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
