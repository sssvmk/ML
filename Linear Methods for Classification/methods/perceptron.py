"""
Rosenblatt's perceptron learning algorithm (ESLII 4.5.1).
Objective: D(beta, beta0) = - sum_{i in M} y_i (x_i'beta + beta0) over misclassified points M, minimised by stochastic gradient steps
    (beta, beta0) <- (beta, beta0) + rho * (y_i x_i, y_i),  rho = 1,  y in {-1,+1},  points visited in random order each epoch.
Santander is NOT linearly separable, so the algorithm cannot converge: training stops at the epoch cap and the convergence
flag is False.  The model is the LAST iterate (plain algorithm); set cfg.perceptron_pocket=True to keep the best-training-error epoch.
Metrics: training error per epoch, number of updates / epochs, convergence flag; test error +- SE, AUC from the raw score.
"""
import _bootstrap  # noqa: F401
import time

import numpy as np

import common
from common import Model

NAME = "perceptron"


def _epoch_py(X, ys, beta, b0, order):
    upd = 0
    for i in order:
        xi = X[i]
        if ys[i] * (xi @ beta + b0) <= 0.0:
            beta += ys[i] * xi
            b0 += ys[i]
            upd += 1
    return b0, upd


_epoch = _epoch_py
try:                                                    # optional acceleration
    from numba import njit

    @njit(cache=False)
    def _epoch_nb(X, ys, beta, b0, order):
        upd = 0
        p = X.shape[1]
        for i in order:
            s = b0
            for j in range(p):
                s += X[i, j] * beta[j]
            if ys[i] * s <= 0.0:
                for j in range(p):
                    beta[j] += ys[i] * X[i, j]
                b0 += ys[i]
                upd += 1
        return b0, upd
    _epoch = _epoch_nb
except Exception:
    pass


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    X = np.ascontiguousarray(np.asarray(view.X, dtype=np.float64))
    ys = 2.0 * view.y - 1.0
    n, p = X.shape
    rng = np.random.RandomState(_CFG["seed"])
    beta, b0 = np.zeros(p), 0.0
    hist, best = [], (np.inf, None, None)
    converged = False
    t0 = time.time()
    for ep in range(_CFG["epochs"]):
        order = rng.permutation(n)
        try:
            b0, upd = _epoch(X, ys, beta, b0, order)
        except Exception:
            b0, upd = _epoch_py(X, ys, beta, b0, order)
        err = float(((X @ beta + b0 > 0).astype(int) != view.y).mean())
        hist.append({"epoch": ep + 1, "updates": int(upd), "train_error": err})
        if err < best[0]:
            best = (err, beta.copy(), b0)
        if upd == 0:
            converged = True
            break
    if _CFG["pocket"] and best[1] is not None:
        beta, b0 = best[1], best[2]
    b, bb = beta.copy(), float(b0)
    return Model(lambda Z: Z @ b + bb, coef=b, intercept=bb,
                 info={"converged": converged, "epochs_run": len(hist), "total_updates": int(sum(h["updates"] for h in hist)),
                       "final_train_error": hist[-1]["train_error"], "best_train_error": best[0], "pocket_used": _CFG["pocket"],
                       "history": hist, "seconds": round(time.time() - t0, 1),
                       "separable": bool(converged)})


_CFG = {"epochs": 50, "pocket": False, "seed": 42}


def extra_plot(out, cv_df, summary):
    import matplotlib.pyplot as plt
    h = summary["model_info"]["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot([r["epoch"] for r in h], [r["train_error"] for r in h], "o-", ms=3); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training error")
    ax[1].plot([r["epoch"] for r in h], [r["updates"] for r in h], "o-", ms=3); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("# updates (misclassified visits)")
    fig.suptitle(f"Perceptron (converged={summary['model_info']['converged']})"); fig.tight_layout(); fig.savefig(out / "perceptron_training.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    _CFG.update(epochs=cfg.perceptron_max_epochs, pocket=cfg.perceptron_pocket, seed=cfg.seed)
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=False, do_cv=False,
                                 extra_plot=extra_plot, notes="Plain perceptron; no CV (no hyper-parameters). Non-separable data => convergence flag False.",
                                 console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
