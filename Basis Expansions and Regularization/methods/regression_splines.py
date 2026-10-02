"""
Piecewise polynomials and regression splines (ESLII 5.2).
Objective: residual sum of squares of y on a spline basis, sum_i (y_i - sum_j b_j h_j(x_i))^2, knots (at predictor quantiles) and degree fixed in advance.
Grid: degree 0 (piecewise constant), 1 (continuous piecewise linear), 3 (cubic spline, C2) x number of interior knots K; df = number of basis functions.
Metrics: test MSE +- SE, 10-fold CV MSE vs knots / df (one-SE rule), AIC / BIC / Cp, pointwise standard-error bands.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import common
from bases import quantile_knots, trunc_power_basis
from common import ls_fit

NAME = "regression_splines"
KS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30]


def make_grid(prep, cfg, X, y):
    nu = len(np.unique(X[:, 0]))
    return [{"degree": d, "K": K} for d in (0, 1, 3) for K in KS if K + d + 1 <= max(nu // 4, 4)]


def fit_path(X, y, grid, cfg):
    xs, out = X[:, 0], []
    for hp in grid:
        kn = quantile_knots(xs, hp["K"])
        fn = (lambda Z, kn=kn, d=hp["degree"]: trunc_power_basis(Z[:, 0], kn, d))
        out.append(ls_fit(fn, X, y, info={"degree": hp["degree"], "knots": kn.tolist()}))
    return out


def extra(ctx):
    c = ctx["curve"]
    rows = []
    for d, g in c.groupby("degree"):
        i = g["cv_mse"].idxmin()
        rows.append({"degree": d, "best_K": int(g.loc[i, "K"]), "best_df": g.loc[i, "df"], "cv_mse": g.loc[i, "cv_mse"], "cv_se": g.loc[i, "cv_se"]})
    pd.DataFrame(rows).to_csv(ctx["out"] / "best_by_degree.csv", index=False)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, extra=extra,
                               notes="One predictor; truncated-power basis fitted by QR least squares; knots at quantiles of the fitting data.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
