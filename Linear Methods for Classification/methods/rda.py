"""
Regularized Discriminant Analysis (ESLII 4.3.1).
Objective: Gaussian log-likelihood with shrunken covariances
    S_k(alpha)       = alpha * S_k + (1 - alpha) * S_pooled            (alpha = 1: QDA, alpha = 0: LDA)
    S_k(alpha,gamma) = gamma * S_k(alpha) + (1 - gamma) * s2_k * I     s2_k = trace(S_k(alpha)) / p   (gamma = 1: no shrinkage)
Tuning: stratified 10-fold CV over the (alpha, gamma) grid, one-SE rule on CV log-loss (simplest = smallest alpha + gamma).
Metrics: CV log-loss / error / AUC over the grid, test error +- SE, log-loss, AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
from common import gaussian_model

NAME = "rda"


def make_grid(prep, cfg):
    return [{"alpha": a, "gamma": g} for a in cfg.rda_alphas for g in cfg.rda_gammas]


def complexity(hp):
    return hp["alpha"] + hp["gamma"]


def fit(view, hp):
    cs = view.cs
    mu, W = cs.moments()
    N = cs.n.sum()
    p = len(mu[0])
    pooled = (W[0] + W[1]) / (N - 2)
    a, g = hp["alpha"], hp["gamma"]
    sig = [a * (W[k] / (cs.n[k] - 1)) + (1 - a) * pooled for k in (0, 1)]
    sig = [g * s + (1 - g) * (np.trace(s) / p) * np.eye(p) for s in sig]
    return gaussian_model(mu, sig[0] if a == 0 else tuple(sig), cs.n / N)


def extra_plot(out, cv_df, summary):
    if cv_df is None:
        return
    al, ga = sorted(cv_df["alpha"].unique()), sorted(cv_df["gamma"].unique())
    M = np.full((len(al), len(ga)), np.nan)
    for _, r in cv_df.iterrows():
        M[al.index(r["alpha"]), ga.index(r["gamma"])] = r["cv_logloss"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ga))); ax.set_xticklabels(ga); ax.set_yticks(range(len(al))); ax.set_yticklabels(al)
    ax.set_xlabel("gamma (shrink to scaled identity)"); ax.set_ylabel("alpha (0 = LDA, 1 = QDA)"); ax.set_title("RDA: CV log-loss")
    fig.colorbar(im)
    for i in range(len(al)):
        for j in range(len(ga)):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", color="w", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "rda_alpha_gamma_heatmap.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=True,
                                 complexity=complexity, extra_plot=extra_plot,
                                 notes=f"grid alpha={list(cfg.rda_alphas)} x gamma={list(cfg.rda_gammas)}", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
