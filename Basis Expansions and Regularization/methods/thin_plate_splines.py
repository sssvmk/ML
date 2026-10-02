"""
Multidimensional / thin-plate splines (ESLII 5.7) on (latitude, longitude).
Objective: sum_i (y_i - f(x_i))^2 + lambda * J(f),  J(f) = integral integral [f_11^2 + 2 f_12^2 + f_22^2] dx1 dx2.
Reduced-rank thin-plate regression spline: f(x) = sum_j a_j eta(|x - xi_j|) + b0 + b1 x1 + b2 x2, eta(r) = r^2 log r, `tps_knots` k-means knots,
constraint T'a = 0 (planar part unpenalised), penalty a'Ea.  lambda is parameterised by effective df.
Metrics: 10-fold CV MSE vs lambda / df (one-SE rule, time-blocked folds) PLUS a spatially blocked CV (longitude strips) at the chosen df,
test MSE +- SE, Moran's I of the test residuals (leftover spatial pattern) and fitted-surface / residual maps.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
from bases import df_to_lambda, kmeans2d, tps_design, tps_setup
from common import gram, penalized_fit

NAME = "thin_plate_splines"
TARGET_DF = [3.5, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 90, 130, 170]
_ST = {}


def make_grid(prep, cfg, X, y):
    knots = kmeans2d(X, cfg.tps_knots, seed=cfg.seed)
    _ST["tps"] = tps_setup(knots)
    B = tps_design(X, _ST["tps"])
    tg = [d for d in TARGET_DF if d < B.shape[1] - 1]
    lams, _ = df_to_lambda(B.T @ B, _ST["tps"]["Omega"], tg)
    return [{"df_target": d, "lam": l} for d, l in zip(tg, lams)]


def fit_path(X, y, grid, cfg):
    st = _ST["tps"]
    B = tps_design(X, st)
    G, c, yy = gram(B, y)
    fn = lambda Z: tps_design(Z, st)
    return [penalized_fit(fn, G, c, yy, len(y), st["Omega"], hp["lam"], info={"df_target": hp["df_target"]}) for hp in grid]


def extra(ctx):
    Xtr, y_fit, y_ev, grid, i = ctx["Xtr"], ctx["y_fit_tr"], ctx["y_ev_tr"], ctx["grid"], ctx["i_sel"]
    K = ctx["cfg"].cv_folds
    order = np.argsort(Xtr[:, 1])                                   # longitude strips = spatial blocks
    errs = []
    for blk in np.array_split(order, K):
        tr = np.ones(len(y_ev), bool); tr[blk] = False
        m = fit_path(Xtr[tr], y_fit[tr], [grid[i]], ctx["cfg"])[0]
        errs.append(float(np.mean((y_ev[blk] - m.predict(Xtr[blk])) ** 2)))
    base = [float(np.mean((y_ev[b] - y_fit.mean()) ** 2)) for b in np.array_split(order, K)]
    res = {"spatial_block_cv_mse": float(np.mean(errs)), "spatial_block_cv_se": float(np.std(errs, ddof=1) / np.sqrt(K)),
           "spatial_block_cv_mean_baseline_mse": float(np.mean(base)), "chosen_df": ctx["summary"]["selected_df"]}
    (ctx["out"] / "spatial_block_cv.json").write_text(json.dumps(res, indent=2))
    ctx["summary"]["spatial_block_cv"] = res
    ctx["log"].info("spatial block CV MSE %.6f +- %.6f (mean baseline %.6f)", res["spatial_block_cv_mse"], res["spatial_block_cv_se"], res["spatial_block_cv_mean_baseline_mse"])


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="xy", make_grid=make_grid, fit_path=fit_path, extra=extra,
                               notes="Inputs are the z-scored (latitude, longitude); k-means knots from the training coordinates.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
