"""
Nadaraya-Watson kernel smoother (ESLII 6.1) - REGRESSION on the single chosen Zillow predictor.
Loss: kernel-weighted squared error with a locally constant fit, min_a sum_i K_lambda(x0, x_i) (y_i - a)^2 -> the weighted average (6.2).
Kernel: tri-cube with a nearest-neighbour bandwidth, span = k/N. Fitted on a grid of target points and interpolated (ESLII 6.9).
Metrics: test MSE +- SE, 10-fold CV MSE vs span / df = trace(S_lambda) (one-SE rule), leave-one-out CV from the smoother matrix (Ex. 6.7), GCV,
C_lambda = ASR + 2 sigma^2 trace(S)/N (6.35), pointwise SE bands.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
import kern

NAME, TASK, INPUT = "nw_regression", "regression", "x1"
SPANS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def make_grid(cfg, prep, Xs, ys):
    return [{"span": s} for s in SPANS]


def _sorted(X, y):
    o = np.argsort(X[:, 0], kind="stable")
    return X[o, 0], y[o]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    xs, ys = _sorted(Xref, yref)
    gp = kern.make_grid_points(xs)
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        r = kern.local_poly_1d(xs, ys, gp, hp["span"], 0)
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
    rows = [kern.diag_from_grid(xs, ys, gp, kern.local_poly_1d(xs, ys, gp, hp["span"], 0)) for hp in grid]
    df, rss, n = np.array([r["df"] for r in rows]), np.array([r["rss"] for r in rows]), rows[0]["n"]
    big = int(np.argmax(df))
    sig2 = rss[big] / max(n - df[big], 1.0)
    return {"df": df, "gcv": [r["gcv"] for r in rows], "loocv": [r["loocv"] for r in rows], "cp": rss / n + 2 * sig2 * df / n}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    diagnostics=diagnostics, notes="Tri-cube kernel, k-NN bandwidth (span = k/N), grid-of-targets fit + interpolation.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
