"""
Nonparametric logistic regression (ESLII 5.6).  Binary target: large_miss = 1{ |logerror| > train 90th percentile of |logerror| }.
Objective: penalised negative log-likelihood  -sum[y log p + (1-y) log(1-p)] + (lambda/2) * integral f''(t)^2 dt,  logit p = f(x), f a cubic spline
(B-spline basis with `logit_knots` interior knots), fitted by penalised IRLS (Newton).  lambda is parameterised by effective df.
Metrics: 10-fold CV deviance (log-loss) vs lambda / df (one-SE rule), test log-loss +- SE (against the base-rate model), AUC, calibration,
test error +- SE at the Youden threshold tuned on validation.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from bases import bs_design, bs_knots, bs_penalty, df_to_lambda, quantile_knots
from common import Smoother, sigmoid

NAME = "nonparametric_logistic"
TARGET_DF = [2.2, 3, 4, 5, 6, 8, 10, 12, 15, 20]


def _setup(xs, m):
    m = int(min(m, max(4, len(np.unique(xs)) // 3)))
    t = bs_knots(xs.min(), xs.max(), quantile_knots(xs, m))
    return t, bs_penalty(t)


def make_grid(prep, cfg, X, y):
    xs = X[:, 0]
    t, Om = _setup(xs, cfg.logit_knots)
    B = bs_design(xs, t)
    pb = float(y.mean())
    tg = [d for d in TARGET_DF if d < B.shape[1] - 0.5]
    lams, _ = df_to_lambda(pb * (1 - pb) * (B.T @ B), Om, tg)
    return [{"df_target": d, "lam": l} for d, l in zip(tg, lams)]


def fit_path(X, y, grid, cfg):
    xs = X[:, 0]
    t, Om = _setup(xs, cfg.logit_knots)
    B = bs_design(xs, t)
    m = B.shape[1]
    pb = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    theta = np.full(m, np.log(pb / (1 - pb)))                      # B rows sum to 1 -> constant logit
    out = []
    for hp in sorted(range(len(grid)), key=lambda i: -grid[i]["lam"]):          # warm start from smoother to rougher
        lam = grid[hp]["lam"]
        for it in range(40):
            eta = B @ theta
            p = sigmoid(eta)
            w = np.clip(p * (1 - p), 1e-8, None)
            z = eta + (y - p) / w
            Bw = B * w[:, None]
            G, c = Bw.T @ B, Bw.T @ z
            A = G + lam * Om + 1e-10 * np.trace(G) / m * np.eye(m)
            new = np.linalg.solve(A, c)
            done = np.max(np.abs(new - theta)) < 1e-9
            theta = new
            if done:
                break
        Ainv = np.linalg.inv(G + lam * Om + 1e-10 * np.trace(G) / m * np.eye(m))
        th = theta.copy()
        out.append((hp, Smoother(lambda Z, th=th: sigmoid(bs_design(Z[:, 0], t) @ th), df=float(np.trace(Ainv @ G)),
                                 info={"lambda": lam, "irls_iterations": it + 1})))
    res = [None] * len(grid)
    for i, mod in out:
        res[i] = mod
    return res


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, task="logistic",
                               notes="Penalised IRLS on a cubic B-spline basis; target = large miss (|logerror| above the train 90th percentile).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
