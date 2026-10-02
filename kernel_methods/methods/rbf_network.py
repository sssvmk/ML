"""
Radial basis function network (ESLII 6.7) - REGRESSION on the top-6 Zillow predictors.
Loss (6.29): sum_i (y_i - b0 - sum_j b_j exp(-||x_i - xi_j||^2 / lambda^2))^2.  Main fit: centres xi_j from k-means (unsupervised), common width lambda =
scale x mean nearest-centre spacing, then PLAIN least squares in b (convex). Optional renormalised basis (6.30) avoids 'holes'.
Metrics: test MSE +- SE, 10-fold CV MSE vs number of basis functions M and width (one-SE rule on df = M + 1), and a NONCONVEX check: the full
criterion (6.29) optimised over (xi, lambda, b) by L-BFGS from several random restarts (spread of the training loss = local minima).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
from scipy.optimize import minimize

import common
import kern

NAME, TASK, INPUT = "rbf_network", "regression", "xm"
MS = [5, 10, 20, 40, 80]
SCALES = [0.5, 1.0, 2.0]


def make_grid(cfg, prep, Xs, ys):
    return [{"M": M, "scale": s, "renorm": r} for M in MS for s in SCALES for r in (False, True)]


def _phi(X, C, lam, renorm):
    d2 = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * X @ C.T
    P = np.exp(-np.maximum(d2, 0) / lam ** 2)
    return P / (P.sum(1, keepdims=True) + 1e-12) if renorm else P


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    out = np.empty((len(grid), len(Xq)))
    cache = {}
    n = len(yref)
    for h, hp in enumerate(grid):
        M = hp["M"]
        if M not in cache:
            C = kern.kmeans(Xref, M, seed=cfg.seed)
            d = np.sqrt(np.maximum((C ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * C @ C.T, 0)); np.fill_diagonal(d, np.inf)
            cache[M] = (C, float(np.mean(d.min(1))))
        C, spacing = cache[M]
        lam = hp["scale"] * spacing
        A = np.c_[np.ones(n), _phi(Xref, C, lam, hp["renorm"])]
        beta = np.linalg.solve(A.T @ A + 1e-6 * n * np.eye(A.shape[1]), A.T @ yref)
        out[h] = np.c_[np.ones(len(Xq)), _phi(Xq, C, lam, hp["renorm"])] @ beta
    return out


def diagnostics(Xs, ys, grid, cfg):
    return {"df": np.array([hp["M"] + 1.0 for hp in grid])}


def _loss_grad(theta, X, y, M, p, ridge=1e-6):
    b0, beta = theta[0], theta[1:1 + M]
    C = theta[1 + M:1 + M + M * p].reshape(M, p)
    lam = np.exp(theta[1 + M + M * p:])
    diff = X[:, None, :] - C[None]
    d2 = (diff ** 2).sum(2)
    P = np.exp(-d2 / lam[None] ** 2)
    r = b0 + P @ beta - y
    L = float(r @ r + ridge * beta @ beta)
    rp = (2 * r)[:, None] * P * beta[None]
    gC = (rp[:, :, None] * diff * (2 / lam[None, :, None] ** 2)).sum(0)
    gl = (rp * 2 * d2 / lam[None] ** 2).sum(0)
    return L, np.concatenate([[2 * r.sum()], 2 * (r @ P) + 2 * ridge * beta, gC.ravel(), gl])


def extra(ctx):
    cfg, hp = ctx["cfg"], ctx["grid"][ctx["i_sel"]]
    Xs, ys = ctx["Xs"][:8000], ctx["ys_fit"][:8000]
    M, p = hp["M"], Xs.shape[1]
    rng = np.random.RandomState(cfg.seed)
    C0 = kern.kmeans(Xs, M, seed=cfg.seed)
    d = np.sqrt(np.maximum((C0 ** 2).sum(1)[:, None] + (C0 ** 2).sum(1)[None] - 2 * C0 @ C0.T, 0)); np.fill_diagonal(d, np.inf)
    lam0 = hp["scale"] * float(np.mean(d.min(1)))
    runs = []
    for r in range(cfg.rbf_restarts):
        C = C0 + 0.3 * lam0 * rng.randn(*C0.shape)
        theta0 = np.concatenate([[ys.mean()], 0.01 * rng.randn(M), C.ravel(), np.log(lam0 * np.exp(0.2 * rng.randn(M)))])
        res = minimize(_loss_grad, theta0, args=(Xs, ys, M, p), jac=True, method="L-BFGS-B", options={"maxiter": 150})
        th = res.x
        Cf, lam, beta = th[1 + M:1 + M + M * p].reshape(M, p), np.exp(th[1 + M + M * p:]), th[1:1 + M]
        Pt = np.exp(-((ctx["X_te"][:, None, :] - Cf[None]) ** 2).sum(2) / lam[None] ** 2)
        runs.append({"restart": r, "train_sse": float(res.fun), "test_mse": float(np.mean((ctx["y_te"] - (th[0] + Pt @ beta)) ** 2)), "iterations": int(res.nit)})
    sse = [r["train_sse"] for r in runs]
    out = {"M": M, "n_train_rows_used": len(ys), "restarts": runs, "train_sse_min": min(sse), "train_sse_max": max(sse), "train_sse_rel_spread": (max(sse) - min(sse)) / min(sse),
           "test_mse_best_train_restart": runs[int(np.argmin(sse))]["test_mse"], "test_mse_of_least_squares_model": ctx["summary"]["test"]["mse"]}
    (ctx["out"] / "rbf_nonconvex_restarts.json").write_text(json.dumps(out, indent=2))
    ctx["summary"]["nonconvex_restarts"] = out


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    diagnostics=diagnostics, extra=extra, notes="k-means centres + least squares; nonconvex (6.29) check in rbf_nonconvex_restarts.json.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
