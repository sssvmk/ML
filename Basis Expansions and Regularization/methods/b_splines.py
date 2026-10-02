"""
B-splines (ESLII appendix).  A BASIS, not a separate smoother: here the loss is the least-squares RSS of a cubic regression spline written in the
B-spline basis (the penalised B-spline form is used by smoothing_splines.py).  Same function space as the truncated-power regression spline, so
the fitted values must agree; what B-splines add is numerical conditioning (a local, banded, partition-of-unity basis).
Metrics: the same as regression splines (test MSE +- SE, 10-fold CV MSE vs knots / df with the one-SE rule, AIC/BIC/Cp, SE bands) plus the
condition number of the design matrix and the maximum fitted-value difference vs the truncated-power fit.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
from bases import bs_design, bs_knots, quantile_knots, trunc_power_basis
from common import ls_fit

NAME = "b_splines"
KS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30]


def make_grid(prep, cfg, X, y):
    nu = len(np.unique(X[:, 0]))
    return [{"K": K} for K in KS if K + 4 <= max(nu // 4, 4)]


def fit_path(X, y, grid, cfg):
    xs, out = X[:, 0], []
    for hp in grid:
        t = bs_knots(xs.min(), xs.max(), quantile_knots(xs, hp["K"]))
        out.append(ls_fit(lambda Z, t=t: bs_design(Z[:, 0], t), X, y, info={"n_interior_knots": hp["K"]}))
    return out


def extra(ctx):
    Xtr, y = ctx["Xtr"], ctx["y_fit_tr"]
    xs = Xtr[:, 0]
    rows = []
    for hp, m in zip(ctx["grid"], ctx["models"]):
        kn = quantile_knots(xs, hp["K"])
        Bt = trunc_power_basis(xs, kn, 3)
        tp = ls_fit(lambda Z, kn=kn: trunc_power_basis(Z[:, 0], kn, 3), Xtr, y)
        rows.append({"K": hp["K"], "df": m.df, "cond_bspline": m.info["cond_design"], "cond_truncated_power": tp.info["cond_design"],
                     "max_abs_fitted_diff_vs_truncated_power": float(np.max(np.abs(m.predict(Xtr) - tp.predict(Xtr))))})
    df = pd.DataFrame(rows)
    df.to_csv(ctx["out"] / "conditioning_vs_truncated_power.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(df["df"], df["cond_bspline"], "o-", label="B-spline basis"); ax.semilogy(df["df"], df["cond_truncated_power"], "s-", label="truncated-power basis")
    ax.set_xlabel("df"); ax.set_ylabel("condition number of design matrix"); ax.legend(); ax.set_title("B-splines: numerical conditioning")
    fig.tight_layout(); fig.savefig(ctx["out"] / "conditioning.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, extra=extra,
                               notes="Cubic B-spline regression; equals the truncated-power cubic regression spline up to rounding.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
