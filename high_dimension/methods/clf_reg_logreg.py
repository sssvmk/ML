"""
Regularized logistic regression (classification) - scikit-learn multinomial LogisticRegression with a ridge (L2) penalty on standardised genes (18.10): minimise the negative multinomial log-likelihood + (1/2C) sum_kj b_kj^2.
Tuning: 12 values of C by repeated stratified CV, one-SE rule (smallest C). Metrics: test error +- SE, log-loss +- SE, macro one-vs-rest AUC, CV error vs C. All genes are used.
"""
import _bootstrap  # noqa: F401
import numpy as np

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "reg_logreg", "classification"
GRID = {"lr__C": list(np.logspace(-5, 0.5, 12))}


def make_est(wins, out):
    from sklearn.linear_model import LogisticRegression
    return hl.make_pipeline(hl.prefix_steps(wins) + [("lr", LogisticRegression(max_iter=5000))], out)


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: p["lr__C"], lambda m, p: int(m.named_steps["var"].get_support().sum()), None, notes="ridge multinomial logistic regression")
