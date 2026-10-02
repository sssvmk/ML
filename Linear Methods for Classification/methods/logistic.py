"""
Logistic regression (maximum likelihood).   Objective: negative binomial log-likelihood (deviance).
Solver: Newton / IRLS on the Gram matrix of the weighted design (row weights 0/1 select the CV training fold); folds are
warm-started from the previous fit so they converge in 2-3 Newton steps.
Metrics: test error +- SE, log-loss (deviance), AUC, calibration (reliability curve, Brier, ECE); CV deviance / error / AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Model, sigmoid

NAME = "logistic"
_WARM = {}


def _pass(X, y, w, beta, chunk=20000):
    p = X.shape[1]
    H, g, dev = np.zeros((p + 1, p + 1)), np.zeros(p + 1), 0.0
    for a in range(0, X.shape[0], chunk):
        sl = slice(a, a + chunk)
        wb = w[sl]
        if not wb.any():
            continue
        xb = np.asarray(X[sl], dtype=np.float64)
        eta = xb @ beta[:p] + beta[p]
        mu = sigmoid(eta)
        dev += float(np.sum(wb * (np.logaddexp(0.0, eta) - y[sl] * eta)))
        wt = wb * mu * (1 - mu)
        r = wb * (y[sl] - mu)
        xw = xb * wt[:, None]
        H[:p, :p] += xw.T @ xb
        H[:p, p] += xw.sum(0); H[p, p] += wt.sum()
        g[:p] += xb.T @ r; g[p] += r.sum()
    H[p, :p] = H[:p, p]
    return H, g, dev


def irls(X, y, w, beta0, ridge=1e-8, max_iter=40, tol=1e-7, max_step=5.0):
    """Newton / IRLS with Levenberg damping: an absolute damping floor keeps the system solvable when the weights collapse
    (quasi-separation), steps are capped, and an iteration that increases the deviance is rolled back with stronger damping."""
    p = X.shape[1]
    beta, damp = beta0.copy(), ridge
    prev = None
    last_dev = np.inf
    for it in range(max_iter):
        H, g, dev = _pass(X, y, w, beta)
        if prev is not None and dev > prev[3] * (1 + 1e-10) + 1e-10:        # overshoot -> roll back, damp harder
            beta, H, g, dev = prev[0].copy(), prev[1], prev[2], prev[3]
            damp = min(damp * 100.0, 1e4)
        prev = (beta.copy(), H, g, dev)
        last_dev = dev
        step = np.linalg.solve(H + (damp * np.trace(H) / (p + 1) + 1e-4) * np.eye(p + 1), g)
        mx = np.abs(step).max()
        if mx > max_step:
            step *= max_step / mx
        beta = beta + step
        if np.abs(step).max() < tol:
            break
    return beta, it + 1, last_dev


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    p = view.p
    beta0 = _WARM.get(p)
    if beta0 is None:
        beta0 = np.zeros(p + 1)
        pbar = float(view.w @ view.y / view.w.sum())
        beta0[p] = np.log(pbar / (1 - pbar))
    beta, iters, dev = irls(view.X, view.y.astype(float), view.w, beta0)
    _WARM[p] = beta.copy()
    b, b0 = beta[:p].copy(), float(beta[p])
    sep = bool(np.abs(b).max() > 25)
    return Model(lambda X: X @ b + b0, coef=b, intercept=b0,
                 info={"newton_iterations": iters, "train_deviance_per_row": dev / view.w.sum(),
                       "quasi_separation_suspected": sep,
                       "warning": "coefficients very large: classes (nearly) separable, MLE does not exist; a tiny ridge keeps it finite" if sep else ""})


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=True,
                                 do_cv=cfg.cv_logistic, notes="Unpenalised MLE via damped IRLS (tiny damping floor; flags quasi-separation).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
