"""
Reduced-rank LDA (Fisher's canonical discriminants).
Objective: max a'Ba subject to a'Wa = 1 (between- vs within-class variance), equivalently the Gaussian MLE with rank-L class means.
Rank is capped at min(K-1, p) = 1 for K = 2, so reduced-rank LDA == LDA; the fit below is computed through the canonical
coordinate and the run logs the numerical agreement with plain LDA (a built-in correctness check).
Classification: nearest class centroid in the canonical space plus log prior.
Metrics: 10-fold CV error / log-loss / AUC vs rank L (only L = 1 exists), test error +- SE, log-loss, AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Model, gaussian_model

NAME = "reduced_rank_lda"


def make_grid(prep, cfg):
    return [{"rank": 1}]


def fit(view, hp):
    cs = view.cs
    mu, W = cs.moments()
    N = cs.n.sum()
    pri = cs.n / N
    Sw = (W[0] + W[1]) / (N - 2)
    p = len(mu[0])
    Sw = Sw + 1e-8 * np.trace(Sw) / p * np.eye(p)
    M = mu[1] - mu[0]
    a = np.linalg.solve(Sw, M)                         # K=2: the single canonical direction
    a = a / np.sqrt(a @ Sw @ a)                        # a' W a = 1
    mubar = pri @ mu
    z = a @ (mu - mubar).T                             # class centroids in canonical space (2,)
    lp = np.log(pri[1] / pri[0])
    coef = (z[1] - z[0]) * a
    b = -(z[1] - z[0]) * (a @ mubar) - 0.5 * (z[1] ** 2 - z[0] ** 2) + lp
    ref = gaussian_model(mu, (W[0] + W[1]) / (N - 2), pri)
    diff = float(np.abs(coef - ref.coef).max() / max(np.abs(ref.coef).max(), 1e-300))
    return Model(lambda X: X @ coef + b, coef=coef, intercept=b,
                 info={"rank": 1, "relative_max_coef_difference_vs_LDA": diff,
                       "between_over_within_variance": float((a @ M) ** 2 * pri[0] * pri[1])})


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=True,
                                 notes="K=2 -> rank 1 only; identical to LDA by construction (see model_info).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
