"""
Local likelihood, GAUSSIAN case (ESLII 6.5) - REGRESSION on the single chosen Zillow predictor.
Loss: local negative log-likelihood sum_i K_lambda(x0, x_i) l(y_i, b(x_i)'beta); with the Gaussian likelihood this IS the local regression loss (the run
checks numerically that the fit equals the local polynomial fit). Fitted by the generic local IRLS routine, family = gaussian.
Metrics: same as local regression (test MSE +- SE, CV MSE over (span, degree), one-SE rule on df, LOO-CV, GCV, Cp, SE bands).
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
import kern

NAME, TASK, INPUT = "local_likelihood_regression", "regression", "x1"
SPANS = [0.02, 0.05, 0.1, 0.2, 0.4]
DEGREES = [1, 2]


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
        r = kern.local_glm_1d(xs, ys, gp, hp["span"], hp["degree"], family="gaussian")
        out[h] = np.interp(Xq[:, 0], gp, r["fit"])
        if ctx.get("final"):
            d = kern.diag_from_grid(xs, ys, gp, r)
            sig2 = d["rss"] / max(d["n"] - d["df"], 1.0)
            ctx["store"]["curve"] = (gp, r["fit"], np.sqrt(sig2 * r["varfac"]))
    return out


def diagnostics(Xs, ys, grid, cfg):
    xs, ys = _sorted(Xs, ys)
    gp = kern.make_grid_points(xs)
    rows = [kern.diag_from_grid(xs, ys, gp, kern.local_glm_1d(xs, ys, gp, hp["span"], hp["degree"], family="gaussian")) for hp in grid]
    df, rss, n = np.array([r["df"] for r in rows]), np.array([r["rss"] for r in rows]), rows[0]["n"]
    sig2 = rss[int(np.argmax(df))] / max(n - df.max(), 1.0)
    return {"df": df, "gcv": [r["gcv"] for r in rows], "loocv": [r["loocv"] for r in rows], "cp": rss / n + 2 * sig2 * df / n}


def extra(ctx):
    hp = ctx["grid"][ctx["i_sel"]]
    xs, ys = _sorted(ctx["X_tr"], ctx["y_fit_tr"])
    gp = kern.make_grid_points(xs)
    a = kern.local_glm_1d(xs, ys, gp, hp["span"], hp["degree"], "gaussian")["fit"]
    b = kern.local_poly_1d(xs, ys, gp, hp["span"], hp["degree"])["fit"]
    ctx["summary"]["max_abs_diff_vs_local_polynomial"] = float(np.max(np.abs(a - b)))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    diagnostics=diagnostics, extra=extra, notes="Gaussian local likelihood == local regression.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
