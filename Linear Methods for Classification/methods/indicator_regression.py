"""
Linear regression of an indicator matrix.   Objective: residual sum of squares of the 0/1 indicator columns, sum ||Y - XB||^2.
Both indicator columns are fitted; class = column with the largest fitted value (for K = 2: f1 > f0, i.e. f1 > 0.5).
Score = f1 - f0.  Outputs are NOT probabilities (can leave [0,1]) -> no log-loss.
Metrics: test misclassification error +- SE (validation-tuned threshold and default argmax rule), confusion matrix, AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Model, ols_solve

NAME = "indicator_regression"


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    cs = view.cs
    N = cs.n.sum()
    S, sx = cs.S.sum(0), cs.s.sum(0)
    xbar = sx / N
    Gc = S - N * np.outer(xbar, xbar)
    B, b0 = [], []
    for k in (0, 1):                                  # Y[:, k] = 1{class == k};  X'Y_k = class-k sums
        ybar = cs.n[k] / N
        beta, _ = ols_solve(Gc, cs.s[k] - N * xbar * ybar)
        B.append(beta); b0.append(ybar - xbar @ beta)
    d, d0 = B[1] - B[0], b0[1] - b0[0]
    return Model(lambda X: X @ d + d0, coef=B[1], intercept=b0[1],
                 info={"max_abs_f0_plus_f1_minus_1": float(np.abs(B[0] + B[1]).max()), "columns_sum_to_one": True})


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=False,
                                 notes="Fitted from class sufficient statistics (normal equations). For K=2 the two indicator fits sum to 1.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
