"""Majority-class BASELINE (scikit-learn DummyClassifier, strategy 'prior'): predicts the most frequent training class; its probabilities are the class proportions (the log-loss reference)."""
import _bootstrap  # noqa: F401
from hd_runner import run_method

NAME, TASK = "majority", "classification"


def make_est(wins, out):
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="prior")


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, {}, lambda p: 0, lambda m, p: 0, None, baseline=True, notes="majority-class baseline")
