"""
Smoothing splines (ESLII 5.4).  Objective: penalised RSS  sum_i (y_i - f(x_i))^2 + lambda * integral f''(t)^2 dt ;
theta = (N'N + lambda*Omega)^-1 N'y (generalised ridge), df = trace(S_lambda).
Computed in a cubic B-spline basis with `smooth_knots` interior knots at predictor quantiles (reduced knot set, as R's smooth.spline does for
large n); the penalty matrix is the exact integral of B'' B''.  lambda is parameterised by its effective df (ESLII 5.5.1).
Metrics: 10-fold CV MSE vs lambda / df (one-SE rule), leave-one-out CV from the smoother-matrix shortcut, GCV, test MSE +- SE, pointwise SE bands.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from bases import bs_design, bs_knots, bs_penalty, df_to_lambda, quantile_knots
from common import gram, penalized_fit

NAME = "smoothing_splines"
TARGET_DF = [2.2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 90]


def _setup(xs, m):
    m = int(min(m, max(4, len(np.unique(xs)) // 3)))            # ties (e.g. integer years): keep every B-spline supported by data
    t = bs_knots(xs.min(), xs.max(), quantile_knots(xs, m))
    return t, bs_penalty(t)


def make_grid(prep, cfg, X, y):
    xs = X[:, 0]
    t, Om = _setup(xs, cfg.smooth_knots)
    B = bs_design(xs, t)
    lams, nd = df_to_lambda(B.T @ B, Om, [d for d in TARGET_DF if d < B.shape[1] - 0.5])
    return [{"df_target": d, "lam": l} for d, l in zip([d for d in TARGET_DF if d < B.shape[1] - 0.5], lams)]


def fit_path(X, y, grid, cfg):
    xs = X[:, 0]
    t, Om = _setup(xs, cfg.smooth_knots)
    B = bs_design(xs, t)
    G, c, yy = gram(B, y)
    fn = lambda Z: bs_design(Z[:, 0], t)
    return [penalized_fit(fn, G, c, yy, len(y), Om, hp["lam"], info={"df_target": hp["df_target"], "n_basis": B.shape[1]}) for hp in grid]


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, want_loocv=True,
                               notes=f"Penalised cubic B-spline representation, {cfg.smooth_knots} interior knots; lambda grid defined by df targets on the training data.",
                               console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
