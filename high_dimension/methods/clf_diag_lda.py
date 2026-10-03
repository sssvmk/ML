"""
Diagonal LDA (classification) - scikit-learn LinearDiscriminantAnalysis on genes scaled by their POOLED WITHIN-CLASS standard deviation with a fully shrunken (identity) covariance: that is diagonal LDA (18.2),
the naive-Bayes / independence rule with a common diagonal covariance, delta_k(x) = -sum_j (x_j - xbar_kj)^2 / s_j^2 + 2 log pi_k. No hyper-parameter; no gene selection (all genes are used).
Loss: the Gaussian log-likelihood (classification by the largest score). Metrics: test error +- SE, log-loss +- SE from the posteriors, macro one-vs-rest AUC, CV error, confusion matrix.
"""
import _bootstrap  # noqa: F401
import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "diag_lda", "classification"


def make_est(wins, out):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    steps = hl.prefix_steps(wins, scale=False) + [("wcs", hl.WithinClassScaler()), ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1.0))]
    return hl.make_pipeline(steps, out)


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, {}, lambda p: 0, lambda m, p: int(m.named_steps["var"].get_support().sum()), None, notes="within-class-scaled LDA with identity covariance = diagonal LDA")
