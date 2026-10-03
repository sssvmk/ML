"""Predict-the-mean BASELINE (scikit-learn DummyRegressor): the training (+ validation) mean of y for every test sample. Loss: squared error."""
import _bootstrap  # noqa: F401
from hd_runner import run_method

NAME, TASK = "mean", "regression"


def make_est(wins, out):
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy="mean")


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, {}, lambda p: 0, lambda m, p: 0, None, baseline=True, notes="predict-the-mean baseline")
