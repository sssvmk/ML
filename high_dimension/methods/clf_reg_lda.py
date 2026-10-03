"""
Regularized LDA (classification) - scikit-learn LinearDiscriminantAnalysis(solver='lsqr', shrinkage=gamma) on standardised genes: the within-class covariance is shrunk toward a scaled identity,
Sigma(gamma) = (1 - gamma) S + gamma * (trace S / p) I, so it stays invertible when p >> N. Loss: the Gaussian log-likelihood. Tuning: 9 values of gamma by repeated stratified CV, one-SE rule (largest gamma).
Metrics: test error +- SE, log-loss +- SE, macro one-vs-rest AUC, CV error vs gamma. All genes are used (no selection).
"""
import _bootstrap  # noqa: F401
import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "reg_lda", "classification"
GRID = {"lda__shrinkage": [0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]}


def make_est(wins, out):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return hl.make_pipeline(hl.prefix_steps(wins) + [("lda", LinearDiscriminantAnalysis(solver="lsqr"))], out)


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: -p["lda__shrinkage"], lambda m, p: int(m.named_steps["var"].get_support().sum()), None, notes="shrinkage LDA")
