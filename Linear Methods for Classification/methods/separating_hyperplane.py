"""
Optimal separating hyperplane - SOFT-MARGIN fit (as agreed: Santander is not separable, so the hard-margin problem is infeasible).
Hard-margin objective (ESLII 4.48):  min 1/2 ||beta||^2  s.t.  y_i (x_i'beta + beta0) >= 1.
Fitted here: min 1/2 ||beta||^2 + C sum_i xi_i,  xi_i = max(0, 1 - y_i f(x_i))   with a LARGE C (cfg.svm_C), liblinear dual coordinate descent.
This is a soft-margin fit, NOT the optimal separating hyperplane of the separable case; the output labels it as such.
Metrics: margin M = 1/||beta||, number of support points (y f(x) <= 1), convergence flag; test error +- SE, AUC from the raw score.
"""
import _bootstrap  # noqa: F401
import warnings

import numpy as np

import common
from common import Model

NAME = "separating_hyperplane"
_CFG = {"C": 100.0, "max_iter": 300, "seed": 42}


def make_grid(prep, cfg):
    return [{}]


def fit(view, hp):
    from sklearn.svm import LinearSVC
    X = np.ascontiguousarray(np.asarray(view.X, dtype=np.float64))
    y = view.y
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        svc = LinearSVC(loss="hinge", C=_CFG["C"], dual=True, max_iter=_CFG["max_iter"], tol=1e-4, random_state=_CFG["seed"]).fit(X, y)
    b, b0 = svc.coef_.ravel().copy(), float(svc.intercept_[0])
    f = X @ b + b0
    ys = 2.0 * y - 1.0
    marg = ys * f
    nrm = float(np.linalg.norm(b))
    return Model(lambda Z: Z @ b + b0, coef=b, intercept=b0,
                 info={"fit_type": "SOFT-MARGIN (hard margin infeasible on non-separable data)", "C": _CFG["C"],
                       "margin_1_over_norm_beta": 1.0 / max(nrm, 1e-300), "n_support_points": int((marg <= 1 + 1e-6).sum()),
                       "n_training_errors": int((marg <= 0).sum()), "training_error": float((marg <= 0).mean()),
                       "iterations": int(np.max(svc.n_iter_)), "converged": bool(np.max(svc.n_iter_) < _CFG["max_iter"]),
                       "hard_margin_feasible": bool((marg > 0).all())})


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    _CFG.update(C=cfg.svm_C, max_iter=cfg.svm_max_iter, seed=cfg.seed)
    return common.run_classifier(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit=fit, probabilistic=False, do_cv=False,
                                 notes=f"SOFT-MARGIN linear SVM, hinge loss, C={cfg.svm_C}. Not a hard-margin optimal hyperplane.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
