"""
Local likelihood - LOCAL LINEAR LOGISTIC regression (ESLII 6.5) - CLASSIFICATION on the top-8 Santander features.
Loss: kernel-weighted negative binomial log-likelihood, sum_i K_lambda(x0, x_i) [ y_i (b0 + b1'(x_i - x0)) - log(1 + exp(b0 + b1'(x_i - x0))) ] maximised at every
query point (batched IRLS, slopes ridge-penalised); P(y = 1 | x0) = sigmoid(b0). Tri-cube kernel, k-NN bandwidth in R^8.
Metrics: CV deviance (log-loss) vs span (one-SE rule), test error +- SE, AUC, calibration, pointwise standard errors of the fitted logit (summarised).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import common
import kern

NAME, TASK, INPUT = "local_logistic", "classification", "x8"
SPANS = [0.01, 0.02, 0.05, 0.1, 0.2]


def make_grid(cfg, prep, Xs, ys):
    return [{"span": s} for s in SPANS]


def complexity(hp):
    return 1.0 / hp["span"]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    n, p = Xref.shape
    ks = [int(max(p + 3, np.ceil(hp["span"] * n))) for hp in grid]
    tree = cKDTree(Xref)
    out = np.empty((len(grid), len(Xq)))
    se_all = np.empty((len(grid), len(Xq)))
    for a, D, I in kern.knn_chunks(tree, Xq, max(ks), max_elems=8e6):
        Xqc = Xq[a:a + len(D)]
        for h, k in enumerate(ks):
            W = kern.kernel_weights(D, k)
            hbw = D[:, k - 1:k] * (1 + 1e-12) + 1e-12
            Q = max(1, int(2.5e6 // (k * (p + 1))))
            for b in range(0, len(D), Q):
                sl = slice(b, b + Q)
                Xc = (Xref[I[sl, :k]] - Xqc[sl, None, :]) / hbw[sl, :, None]
                eta, se = kern.local_logistic_batch(Xc, yref[I[sl, :k]], W[sl])
                out[h, a + b:a + b + len(eta)] = kern.sigmoid(eta)
                se_all[h, a + b:a + b + len(eta)] = se
    if ctx.get("final"):
        ctx["store"]["se"] = se_all[0]
        ctx["store"]["summary"] = {"mean_se_logit": float(se_all[0].mean()), "median_se_logit": float(np.median(se_all[0]))}
    return out


def extra(ctx):
    se = ctx["store"]["se"]
    nv = len(ctx["y_va"])
    p_te, se_te = ctx["p_te"], se[nv:]
    np.save(ctx["out"] / "pointwise_se_logit_test.npy", se_te)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(se_te, bins=60); ax[0].set_title("pointwise SE of the fitted logit (test)")
    lg = np.log(p_te / (1 - p_te))
    o = np.argsort(lg)[::20]
    ax[1].plot(lg[o], p_te[o], "k.", ms=2, label="p")
    ax[1].fill_between(lg[o][np.argsort(lg[o])], 1 / (1 + np.exp(-(np.sort(lg[o]) - 2 * np.median(se_te)))), 1 / (1 + np.exp(-(np.sort(lg[o]) + 2 * np.median(se_te)))), color="red", alpha=0.3, label="+-2 median SE")
    ax[1].set_xlabel("fitted logit"); ax[1].set_ylabel("probability"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ctx["out"] / "pointwise_se.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    complexity=complexity, extra=extra, notes="Local linear logistic, batched IRLS, ridge on slopes (relative 1e-2).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
