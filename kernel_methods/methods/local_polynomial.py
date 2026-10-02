"""
Local linear and local polynomial regression (ESLII 6.1.1, 6.1.2) - REGRESSION on the single chosen Zillow predictor.
Loss: min_beta(x0) sum_i K_lambda(x0, x_i) (y_i - b(x_i)' beta(x0))^2  (6.12), b(x) = (1, x, .., x^d)  (d = 1, 2, 3).
Metrics: test MSE +- SE, 10-fold CV MSE over (span, degree) (one-SE rule on df = trace(S)), LOO-CV, GCV, Cp, pointwise SE bands, and the error in the
BOUNDARY regions (outer 5 % of the predictor at each end) for the chosen degree vs the locally constant (N-W) fit at the same span.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
import kern

NAME, TASK, INPUT = "local_polynomial", "regression", "x1"
SPANS = [0.02, 0.05, 0.1, 0.2, 0.4]
DEGREES = [1, 2, 3]


def make_grid(cfg, prep, Xs, ys):
    return [{"span": s, "degree": d} for d in DEGREES for s in SPANS]


def _sorted(X, y):
    o = np.argsort(X[:, 0], kind="stable")
    return X[o, 0], y[o]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    xs, ys = _sorted(Xref, yref)
    gp = kern.make_grid_points(xs)
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        r = kern.local_poly_1d(xs, ys, gp, hp["span"], hp["degree"])
        out[h] = np.interp(Xq[:, 0], gp, r["fit"])
        if ctx.get("final"):
            d = kern.diag_from_grid(xs, ys, gp, r)
            sig2 = d["rss"] / max(d["n"] - d["df"], 1.0)
            ctx["store"]["curve"] = (gp, r["fit"], np.sqrt(sig2 * r["varfac"]))
            ctx["store"]["summary"] = {"df_full_data": d["df"], "loocv_full_data": d["loocv"], "gcv_full_data": d["gcv"]}
    return out


def diagnostics(Xs, ys, grid, cfg):
    xs, ys = _sorted(Xs, ys)
    gp = kern.make_grid_points(xs)
    rows = [kern.diag_from_grid(xs, ys, gp, kern.local_poly_1d(xs, ys, gp, hp["span"], hp["degree"])) for hp in grid]
    df, rss, n = np.array([r["df"] for r in rows]), np.array([r["rss"] for r in rows]), rows[0]["n"]
    big = int(np.argmax(df))
    sig2 = rss[big] / max(n - df[big], 1.0)
    return {"df": df, "gcv": [r["gcv"] for r in rows], "loocv": [r["loocv"] for r in rows], "cp": rss / n + 2 * sig2 * df / n}


def extra(ctx):
    hp = ctx["grid"][ctx["i_sel"]]
    X_tr, y_fit, X_te, y_te = ctx["X_tr"], ctx["y_fit_tr"], ctx["X_te"], ctx["y_te"]
    lo, hi = np.quantile(X_tr[:, 0], [0.05, 0.95])
    boundary = (X_te[:, 0] < lo) | (X_te[:, 0] > hi)
    P = predict_path(X_tr, y_fit, [{"span": hp["span"], "degree": 0}, hp], X_te, ctx["cfg"], {"final": False})
    res = {"span": hp["span"], "chosen_degree": hp["degree"], "n_boundary_rows": int(boundary.sum()),
           "boundary_mse_locally_constant": float(np.mean((y_te[boundary] - P[0][boundary]) ** 2)), "boundary_mse_chosen_degree": float(np.mean((y_te[boundary] - P[1][boundary]) ** 2)),
           "interior_mse_locally_constant": float(np.mean((y_te[~boundary] - P[0][~boundary]) ** 2)), "interior_mse_chosen_degree": float(np.mean((y_te[~boundary] - P[1][~boundary]) ** 2))}
    (ctx["out"] / "boundary_error.json").write_text(json.dumps(res, indent=2))
    ctx["summary"]["boundary_error"] = res


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    diagnostics=diagnostics, extra=extra, notes="Degrees 1-3, tri-cube k-NN kernel.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
