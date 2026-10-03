"""Ridge regression BASELINE (scikit-learn Ridge on all standardised genes; lambda by repeated CV with the one-SE rule). Loss: sum (y - b0 - X b)^2 + lambda ||b||^2. The book's p >> N computational shortcut (18.15) gives the same fit."""
import _bootstrap  # noqa: F401
import numpy as np

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "ridge", "regression"
GRID = {"ridge__alpha": list(np.logspace(0, 6.5, 27))}


def make_est(wins, out):
    from sklearn.linear_model import Ridge
    return hl.make_pipeline(hl.prefix_steps(wins) + [("ridge", Ridge())], out)


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: -p["ridge__alpha"], lambda m, p: int(m.named_steps["var"].get_support().sum()), None, baseline=True, notes="ridge baseline")
