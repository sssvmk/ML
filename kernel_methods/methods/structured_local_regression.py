"""
Structured local regression (ESLII 6.4) - REGRESSION on the top-6 Zillow predictors.
Structured regression function (6.4.2): VARYING-COEFFICIENT model  f(z, x) = a(z) + sum_{m=1..M} b_m(z) x_m ;  z = the best single predictor (kernel
conditioning variable), x_1..x_M = the next predictors. The kernel acts on z only, so local fitting stays cheap (a weighted least squares with 2 + M(c+1)
columns per grid point) and avoids the curse of dimensionality of a full 6-D kernel. M = 0 is plain local linear regression in z.
Loss: sum_i K_lambda(z0, z_i)(y_i - a - a'(z_i-z0) - sum_m (b_m0 + c * b_m1 (z_i-z0)) x_im)^2   (c = 1: coefficients locally linear in z; c = 0: locally constant).
Metrics: test MSE +- SE, 10-fold CV MSE over (span, M, coefficient degree) (one-SE rule on df = trace(S), the self-leverage sum), GCV, LOO-CV.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
import kern

NAME, TASK, INPUT = "structured_local_regression", "regression", "xm"
SPANS = [0.02, 0.05, 0.1, 0.2, 0.4]
MS = [0, 1, 3, 5]


def make_grid(cfg, prep, Xs, ys):
    return [{"span": s, "m": m, "coef_degree": c} for s in SPANS for m in MS for c in (0, 1) if not (m == 0 and c == 1)]


def _fit(Xref, yref, hp, gp=None):
    o = np.argsort(Xref[:, 0], kind="stable")
    z, Xr, y = Xref[o, 0], Xref[o][:, 1:1 + hp["m"]], yref[o]
    gp = kern.make_grid_points(z, 300) if gp is None else gp
    res = kern.local_vc_1d(z, Xr, y, gp, hp["span"], hp["m"], hp["coef_degree"])
    return res, z, Xr, y


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        res, *_ = _fit(Xref, yref, hp)
        out[h] = kern.local_vc_predict(res, Xq[:, 0], Xq[:, 1:1 + hp["m"]])
        if ctx.get("final"):
            ctx["store"]["vc"] = (res, hp)
    return out


def diagnostics(Xs, ys, grid, cfg):
    df, gcv, loo, n = [], [], [], len(ys)
    for hp in grid:
        res, z, Xr, y = _fit(Xs, ys, hp)
        f = kern.local_vc_predict(res, z, Xr)
        L = np.clip(res["lev"], 0, 0.999)
        rss = float(np.sum((y - f) ** 2)); d = float(L.sum())
        df.append(d); gcv.append((rss / n) / (1 - d / n) ** 2); loo.append(float(np.mean(((y - f) / (1 - L)) ** 2)))
    return {"df": np.array(df), "gcv": gcv, "loocv": loo}


def extra(ctx):
    res, hp = ctx["store"]["vc"]
    names = ctx["prep"].meta["multi_predictors"]
    m = hp["m"]
    g, c = res["grid"], res["coef"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(g, c[:, 0], "k-", lw=2, label=f"a(z)  [z = {names[0]}]")
    col = 2
    for j in range(m):
        ax.plot(g, c[:, col], label=f"b_{j + 1}(z) x = {names[1 + j]}")
        col += 1 + res["coef_degree"]
    ax.set_xlabel("conditioning variable z (z-scored)"); ax.set_ylabel("coefficient"); ax.legend(fontsize=7); ax.set_title("Structured local regression: varying coefficients")
    fig.tight_layout(); fig.savefig(ctx["out"] / "varying_coefficients.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    diagnostics=diagnostics, extra=extra, notes="Varying-coefficient model; kernel on z only. df uses the self-leverage at each point's nearest grid fit.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
