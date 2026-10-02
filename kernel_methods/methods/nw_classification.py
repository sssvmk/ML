"""
Nadaraya-Watson (locally constant) - CLASSIFICATION on the top-8 Santander features (rank-gaussed), ESLII 6.1 / Ex. 6.5, 6.8.
Loss: kernel-weighted squared error on the class indicator, min_a sum_i K_lambda(x0, x_i) (1{y_i = 1} - a)^2; the estimate is the kernel-weighted class
proportion (a locally constant logit, Ex. 6.5). Tri-cube kernel, k-NN bandwidth in R^8 (span = k/N), exact KD-tree neighbours. Probabilities clipped to [1e-4, 1-1e-4].
Metrics: test error +- SE (validation-tuned Youden threshold), log-loss, Brier, AUC, calibration, CV error / log-loss / AUC vs span (one-SE rule).
"""
import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

import common
import kern

NAME, TASK, INPUT = "nw_classification", "classification", "x8"
SPANS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def make_grid(cfg, prep, Xs, ys):
    return [{"span": s} for s in SPANS]


def complexity(hp):
    return 1.0 / hp["span"]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    n = len(yref)
    ks = [int(max(2, np.ceil(hp["span"] * n))) for hp in grid]
    tree = cKDTree(Xref)
    out = np.empty((len(grid), len(Xq)))
    for a, D, I in kern.knn_chunks(tree, Xq, max(ks)):
        Y = yref[I]
        for h, k in enumerate(ks):
            W = kern.kernel_weights(D, k)
            out[h, a:a + len(D)] = (W * Y[:, :k]).sum(1) / np.maximum(W.sum(1), 1e-12)
    return out


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_kernel_method(NAME, TASK, INPUT, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path,
                                    complexity=complexity, notes="k-NN tri-cube kernel in R^8; spans are fractions of the reference set.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
