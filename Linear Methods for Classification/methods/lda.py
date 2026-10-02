"""
Linear Discriminant Analysis.   Objective: maximise the Gaussian class-conditional log-likelihood with a common covariance;
equivalently classify by the linear discriminant delta_k(x) = x'S^-1 mu_k - 0.5 mu_k'S^-1 mu_k + log pi_k.
Score = log posterior odds.  Metrics: test error +- SE, log-loss, AUC, 10-fold CV error / log-loss / AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import gaussian_model

NAME = "lda"


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    cs = view.cs
    mu, W = cs.moments()
    N = cs.n.sum()
    sigma = (W[0] + W[1]) / (N - 2)
    return gaussian_model(mu, sigma, cs.n / N)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=True,
                                 notes="Pooled covariance, class priors from the training proportions.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
