"""
Quadratic Discriminant Analysis.   Objective: the same Gaussian log-likelihood with a separate covariance per class;
delta_k(x) = -0.5 log|S_k| - 0.5 (x-mu_k)'S_k^-1 (x-mu_k) + log pi_k.   Score = log posterior odds.
Metrics: test error +- SE, log-loss, AUC, 10-fold CV error / log-loss / AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import gaussian_model

NAME = "qda"


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    cs = view.cs
    mu, W = cs.moments()
    N = cs.n.sum()
    return gaussian_model(mu, (W[0] / (cs.n[0] - 1), W[1] / (cs.n[1] - 1)), cs.n / N)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=True,
                                 notes="Separate class covariances (tiny relative jitter on the diagonal for numerical safety).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
