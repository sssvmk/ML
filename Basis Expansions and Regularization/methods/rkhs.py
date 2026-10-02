"""
RKHS methods (ESLII 5.8) - Gaussian-kernel ridge regression on the chosen predictor.
Objective: sum_i (y_i - f(x_i))^2 + lambda * ||f||^2_{H_K},  K(x, x') = exp(-(x-x')^2 / (2 h^2)); minimiser f(x) = sum_j alpha_j K(x, z_j).
Exact kernel ridge needs an n x n matrix (n ~ 10^5), so the expansion is restricted to `rkhs_landmarks` landmarks z_j at predictor quantiles
(Nystrom / subset of regressors): alpha = (K_xz'K_xz + lambda K_zz)^-1 K_xz'y   (with all points as landmarks this is (K + lambda I)^-1 y).
Grid: bandwidth h (in sd units of the z-scored predictor) x lambda = lam_rel * n.
Metrics: 10-fold CV MSE over the (lambda, h) grid (one-SE rule on the effective df), test MSE +- SE.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
from common import gram, penalized_fit

NAME = "rkhs"
HS = [0.05, 0.1, 0.2, 0.4, 0.8]
LAMS_REL = list(np.logspace(-7, -1, 9))


def make_grid(prep, cfg, X, y):
    return [{"h": h, "lam_rel": float(l)} for h in HS for l in LAMS_REL]


def fit_path(X, y, grid, cfg):
    xs = X[:, 0]
    z = np.unique(np.quantile(xs, np.linspace(0, 1, cfg.rkhs_landmarks)))
    n = len(y)
    res = [None] * len(grid)
    for h in sorted(set(g["h"] for g in grid)):
        kern = lambda Z, h=h: np.exp(-((Z[:, 0][:, None] - z[None, :]) ** 2) / (2 * h * h))
        Kxz = kern(X)
        Kzz = np.exp(-((z[:, None] - z[None, :]) ** 2) / (2 * h * h))
        G, c, yy = gram(Kxz, y)
        for i, hp in enumerate(grid):
            if hp["h"] == h:
                res[i] = penalized_fit(kern, G, c, yy, n, Kzz, hp["lam_rel"] * n, info={"h": h, "lam_rel": hp["lam_rel"], "landmarks": len(z)})
    return res


def extra(ctx):
    c = ctx["curve"]
    hs, ls = sorted(c["h"].unique()), sorted(c["lam_rel"].unique())
    M = np.full((len(hs), len(ls)), np.nan)
    for _, r in c.iterrows():
        M[hs.index(r["h"]), ls.index(r["lam_rel"])] = r["cv_mse"]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ls))); ax.set_xticklabels([f"{l:.0e}" for l in ls], rotation=45); ax.set_yticks(range(len(hs))); ax.set_yticklabels(hs)
    ax.set_xlabel("lambda / n"); ax.set_ylabel("kernel bandwidth h"); ax.set_title("RKHS: 10-fold CV MSE"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(ctx["out"] / "rkhs_cv_heatmap.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, extra=extra,
                               notes=f"Nystrom kernel ridge, {cfg.rkhs_landmarks} landmarks; Gaussian kernel only.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
