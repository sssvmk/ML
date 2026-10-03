"""
Support vector classifier in high dimensions (classification) - scikit-learn LinearSVC (one-vs-rest hinge loss + L2 penalty) on standardised genes. Loss: sum_i [1 - y_i f(x_i)]_+ + (1/2C)||b||^2.
With p >> N the classes are usually separable, so the regularization matters little (the book found no gain from it). Tuning: 10 values of C by repeated stratified CV, one-SE rule (smallest C).
Metrics: test error +- SE, macro one-vs-rest AUC from the margins, CV error vs C. No probabilities (no log-loss) and no gene selection.
"""
import _bootstrap  # noqa: F401
import numpy as np

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "linear_svc", "classification"
GRID = {"svc__C": list(np.logspace(-5, -0.5, 10))}


def make_est(wins, out):
    from sklearn.svm import LinearSVC
    return hl.make_pipeline(hl.prefix_steps(wins) + [("svc", LinearSVC(max_iter=50000))], out)


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: p["svc__C"], lambda m, p: int(m.named_steps["var"].get_support().sum()), None, notes="scikit-learn LinearSVC")
