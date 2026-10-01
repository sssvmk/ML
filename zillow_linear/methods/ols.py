"""Ordinary least squares.  Objective: RSS = sum (y_i - x_i'b)^2.   Metrics: test MSE +- SE, AIC / BIC / Cp."""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_, info_criteria, ols_solve, sigma2_full

NAME = "ols"


def make_grid(bundle, prep, cfg):
    return None


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    beta, rank = ols_solve(Gc, cc)
    return Path_(beta[:, None], [ybar - xbar @ beta], [float(rank)], rank=[rank])


def extra_curve(curve, path, bundle):
    n, s2 = bundle.tr_fit.n, sigma2_full(bundle.tr_fit)
    ic = info_criteria(curve["train_rss"].values, n, path.extra["rank"] + 1, s2)
    for k, v in ic.items():
        curve[k] = v
    return curve


def final_extras(beta, b0, st, bundle):
    rss = st.sse(beta, b0)
    rank = int(ols_solve(st.centered()[0], st.centered()[1])[1])
    ic = info_criteria(rss, st.n, rank + 1, sigma2_full(st))
    return {"training_rss": rss, "df": rank + 1, "AIC": float(ic["aic"]), "BIC": float(ic["bic"]), "Cp": float(ic["cp"]),
            "note": "AIC/BIC/Cp computed on the data the final model was fit on (train, or train+val if refit)"}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="effective rank", extra_curve=extra_curve, final_extras=final_extras,
                                  notes="Minimum-norm OLS (eigen-decomposition) so collinear columns are handled.",
                                  console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
