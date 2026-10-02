"""
L1-regularised logistic regression (lasso logistic).   Objective: -(1/n) log-likelihood + lambda * ||beta||_1  (intercept free).
Solver: glmnet-style proximal Newton - for each lambda (warm-started, large to small) an IRLS step builds a weighted
least-squares problem that is solved by cyclic coordinate descent with an active-set strategy.
Tuning: stratified 10-fold CV deviance over the lambda path, one-SE rule (largest lambda within 1 SE of the minimum).
Metrics: CV deviance / error / AUC vs lambda, number of non-zero coefficients, test error +- SE, log-loss, AUC.
"""
import _bootstrap  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

import common
from common import Model, sigmoid

NAME = "l1_logistic"
_XF = {}


def _design(view):
    key = id(view.X)
    if key not in _XF:
        _XF.clear()
        _XF[key] = np.asfortranarray(np.asarray(view.X, dtype=np.float64))     # column access is contiguous
    return _XF[key]


def make_grid(prep, cfg):
    y = prep.y_train.astype(float)
    n, p = prep.X_train.shape
    g = np.zeros(p)
    for a in range(0, n, 20000):
        g += np.asarray(prep.X_train[a:a + 20000], dtype=np.float64).T @ (y[a:a + 20000] - y.mean())
    lam_max = float(np.abs(g).max() / n)
    return [{"lam": lam} for lam in lam_max * np.geomspace(1.0, cfg.l1_ratio, cfg.l1_nlambda)]


def complexity(hp):
    return -np.log(hp["lam"])


def _soft(z, t):
    return np.sign(z) * max(abs(z) - t, 0.0)


def lasso_logistic_path(Xf, y, w, lams, max_outer=4, tol=1e-9, max_cycles=50, chunk=20000):
    n, p = Xf.shape
    neff = w.sum()
    beta = np.zeros(p)
    pbar = (w @ y) / neff
    b0 = float(np.log(pbar / (1 - pbar)))
    models = []
    for lam in lams:
        for outer in range(max_outer):
            eta = Xf @ beta + b0
            mu = sigmoid(eta)
            v = np.clip(mu * (1 - mu), 1e-5, None)
            wt = w * v
            r = (y - mu) / v                                   # working residual z - eta
            sw = wt.sum()
            h = np.zeros(p)
            for a in range(0, n, chunk):
                xb = Xf[a:a + chunk]
                h += wt[a:a + chunk] @ (xb * xb)
            h = np.maximum(h / neff, 1e-12)
            beta_old = beta.copy()

            def cycle(idx):
                nonlocal b0
                maxd = 0.0
                for j in idx:
                    xj = Xf[:, j]
                    zj = (wt * r) @ xj / neff + h[j] * beta[j]
                    new = _soft(zj, lam) / h[j]
                    d = new - beta[j]
                    if d != 0.0:
                        r[:] -= d * xj
                        beta[j] = new
                        maxd = max(maxd, h[j] * d * d)
                db = (wt @ r) / sw
                b0 += db
                r[:] -= db
                return maxd

            allidx = np.arange(p)
            for _ in range(max_cycles):
                cycle(allidx)
                active = np.flatnonzero(beta)
                for _ in range(max_cycles):
                    if cycle(active) < tol:
                        break
                if cycle(allidx) < tol:
                    break
            if np.abs(beta - beta_old).max() < 1e-6:
                break
        models.append((beta.copy(), float(b0)))
    return models


def fit_path(view, grid):
    Xf = _design(view)
    lams = [h["lam"] for h in grid]
    out = []
    for b, b0 in lasso_logistic_path(Xf, view.y.astype(float), view.w, lams, max_outer=_CFG["max_outer"]):
        out.append(Model((lambda X, b=b, b0=b0: X @ b + b0), coef=b, intercept=b0, info={"lambda": None, "n_nonzero": int((b != 0).sum())}))
    for m, h in zip(out, grid):
        m.info["lambda"] = h["lam"]
    return out


_CFG = {"max_outer": 4}


def extra_plot(out, cv_df, summary):
    if cv_df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.errorbar(cv_df["lam"], cv_df["cv_logloss"], yerr=cv_df["cv_logloss_se"], fmt="o-", ms=3, capsize=2)
    ax.axvline(cv_df.loc[cv_df.is_cv_best, "lam"].iloc[0], color="red", ls=":", label="CV best")
    ax.axvline(cv_df.loc[cv_df.is_one_se_choice, "lam"].iloc[0], color="purple", ls="--", label="one-SE choice")
    ax.set_xscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("10-fold CV deviance (log-loss)"); ax.legend(); ax.set_title("L1 logistic")
    fig.tight_layout(); fig.savefig(out / "cv_deviance_vs_lambda.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    _CFG["max_outer"] = cfg.l1_max_outer
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path, probabilistic=True,
                                 complexity=complexity, extra_plot=extra_plot,
                                 notes=f"{cfg.l1_nlambda} lambdas, ratio lambda_min/lambda_max = {cfg.l1_ratio}.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
