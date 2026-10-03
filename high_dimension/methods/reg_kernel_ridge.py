"""
Kernel methods (regression) - scikit-learn KernelRidge with a LINEAR or an RBF kernel on the standardised expression values (string kernels do not apply: there are no sequences; the book's other kernels
are built from distances (18.27), which is what the RBF kernel is). Response standardised by TransformedTargetRegressor.
Loss: sum_i (y_i - f(x_i))^2 + lambda ||f||^2 in the kernel's feature space, f = K (K + lambda I)^-1 y.
Tuning: linear: 13 values of lambda; RBF: 4 widths gamma = c / p x 9 values of lambda; repeated K-fold CV, one-SE rule (linear kernel, small gamma and large lambda count as simpler).
Metrics: test MSE +- SE (+ MAE, R2), CV MSE over (kernel, lambda, gamma). Kernel methods use ALL genes (they cannot select or standardise variables, as the book cautions).
"""
import _bootstrap  # noqa: F401
import numpy as np

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "kernel_ridge", "regression"


def GRID(p):
    return [{"krr__regressor__kernel": ["linear"], "krr__regressor__alpha": list(np.logspace(0, 6, 13))},
            {"krr__regressor__kernel": ["rbf"], "krr__regressor__gamma": [c / p for c in (0.1, 0.3, 1.0, 3.0)], "krr__regressor__alpha": list(np.logspace(-3, 1, 9))}]


def make_est(wins, out):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    return hl.make_pipeline(hl.prefix_steps(wins) + [("krr", TransformedTargetRegressor(regressor=KernelRidge(), transformer=StandardScaler()))], out)


def complexity(p):
    return (100.0 if p["krr__regressor__kernel"] == "rbf" else 0.0) + 1e4 * p.get("krr__regressor__gamma", 0.0) - np.log10(p["krr__regressor__alpha"])


def n_features(model, p):
    return int(model.named_steps["var"].get_support().sum())


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, complexity, n_features, None, notes="scikit-learn KernelRidge, linear / RBF kernels.")
