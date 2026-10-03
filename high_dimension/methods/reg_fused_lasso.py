"""
Fused lasso (regression) - cvxpy (standard convex-optimisation package) inside a scikit-learn pipeline [variance filter -> (winsorise) -> standardise -> HIERARCHICAL GENE ORDERING -> fused lasso].
Riboflavin genes have no natural order, so (as agreed) the order is DATA-DRIVEN: the leaf order of an average-linkage clustering on 1 - correlation, fitted on the training fold only. This is an experiment:
the fusion penalty is meaningful only to the extent that neighbouring genes are similar, and the fitted blocks are blocks of the clustering order, not of a genome.
Loss: (1/2n) sum (y - b0 - X b)^2 + lambda1 sum |b_j| + lambda2 sum |b_{j+1} - b_j|; lambda1, lambda2 are given relative to lambda_max = max |X'y_c| / n.
Tuning: 3 x 4 grid over (lambda1, lambda2) by repeated K-fold CV, one-SE rule (least complex = largest penalties).
Metrics: test MSE +- SE (+ MAE, R2), CV MSE over (lambda1, lambda2), number of nonzero coefficients, nonzero BLOCKS (runs of consecutive nonzero coefficients) and JUMPS (|b_{j+1} - b_j| > 0).
'Do the nonzero regions line up with known peaks' does not apply (no known peaks in expression data); the blocks are listed with their genes instead.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "fused_lasso", "regression"
GRID = {"fl__lam1": [0.0, 0.1, 0.3], "fl__lam2": [0.01, 0.03, 0.1, 0.3]}


class FusedLassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam1=0.0, lam2=0.05):
        self.lam1, self.lam2 = lam1, lam2

    def fit(self, X, y):
        import cvxpy as cp
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        lmax = float(np.max(np.abs(X.T @ yc))) / n
        b = cp.Variable(p)
        obj = cp.sum_squares(yc - X @ b) / (2 * n) + self.lam1 * lmax * cp.norm1(b) + self.lam2 * lmax * cp.norm1(cp.diff(b))
        prob = cp.Problem(cp.Minimize(obj))
        try:
            prob.solve(solver=cp.CLARABEL)                                              # interior-point conic solver: ~10x faster than cvxpy's default (OSQP) on these wide problems
        except Exception:
            prob.solve()
        coef = np.asarray(b.value, float).ravel()
        coef[np.abs(coef) < 1e-5 * max(np.abs(coef).max(), 1e-12)] = 0.0                 # interior-point solutions are only approximately sparse
        self.coef_ = coef
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def make_est(wins, out):
    return hl.make_pipeline(hl.prefix_steps(wins) + [("order", hl.GeneOrderer()), ("fl", FusedLassoRegressor())], out)


def complexity(p):
    return -(p["fl__lam1"] + p["fl__lam2"])


def n_features(model, p):
    return int(np.count_nonzero(model.named_steps["fl"].coef_))


def runs(mask):
    r, start = [], None
    for j, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = j
        if not v and start is not None:
            r.append((start, j - 1)); start = None
    return r


def extra(ctx):
    m, out = ctx["model"], ctx["out"]
    b = m.named_steps["fl"].coef_
    order = m.named_steps["order"].order_
    keep = np.where(m.named_steps["var"].get_support())[0]
    genes_ord = np.array(ctx["genes"])[keep][order]
    tol = 1e-3 * max(np.abs(b).max(), 1e-12)                                    # interior-point solutions are only approximately piecewise constant: relative tolerance
    blocks = runs(np.abs(b) > tol)
    jumps = int(np.sum(np.abs(np.diff(b)) > tol))
    rows = [{"block": i, "start": s, "end": e, "n_genes": e - s + 1, "mean_coefficient": float(b[s:e + 1].mean()), "first_genes": ", ".join(genes_ord[s:min(e + 1, s + 5)])} for i, (s, e) in enumerate(blocks)]
    pd.DataFrame(rows).to_csv(out / "nonzero_blocks.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(b, lw=0.8); ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("gene position in the hierarchical-clustering order"); ax.set_ylabel("coefficient")
    ax.set_title(f"fused lasso coefficient profile: {int(np.count_nonzero(b))} nonzero, {len(blocks)} blocks, {jumps} jumps")
    fig.tight_layout(); fig.savefig(out / "coefficient_profile.png", dpi=120); plt.close(fig)
    ctx["summary"]["fused_lasso"] = {"nonzero_coefficients": int(np.count_nonzero(b)), "nonzero_blocks": len(blocks), "jumps": jumps, "lambda1_rel": ctx["best"]["fl__lam1"], "lambda2_rel": ctx["best"]["fl__lam2"],
                                     "ordering": "average-linkage hierarchical clustering on 1 - correlation (leaf order), training rows only"}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, complexity, n_features, extra, notes="cvxpy fused lasso on hierarchically ordered genes (an experiment).")
