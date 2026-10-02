"""
Shared implementation of "SVM for regression" (ESLII 12.3.6-12.3.7) for the three kernel variants run as separate entries (svr_linear, svr_rbf, svr_poly).
Loss: sum_i V_eps(y_i - f(x_i)) + (lambda/2) ||beta||^2  (12.36) with V_eps(r) = max(0, |r| - eps) (12.37), C = 1/lambda;  kernel form f(x) = sum_i (alpha*_i - alpha_i) K(x, x_i) + b0 (12.40, 12.49).
scikit-learn SVR (rbf / poly kernels) and LinearSVR (linear kernel, all training rows); the response is standardised (TransformedTargetRegressor) so C and eps are scale free (eps is in sd units).
Hyper-parameters (CV, random search): C, eps, gamma (as gamma_c / k) and degree for the kernels, and k = number of top-ranked variables (variable selection inside every CV fold). CV selects on MSE of the RAW
logerror (the winner criterion); the CV eps-insensitive loss and MAE of every configuration are logged too. Kernel variants: tuned on the tuning subsample, final fit on <= svm_final_rows rows.
Metrics: test MSE +- SE, test MAE, test eps-insensitive loss, CV curves over (C, eps, gamma), number / fraction of support vectors (points with |r| >= eps).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import svmlib as sl

SPACES = {
    "linear": {"frac": sl.FRACS, "C": [0.001, 0.01, 0.1, 1.0], "eps": [0.05, 0.2, 0.5]},
    "rbf": {"frac": sl.FRACS, "C": [0.1, 1.0, 10.0, 100.0], "eps": [0.05, 0.2, 0.5], "gc": [0.1, 0.3, 1.0, 3.0]},
    "poly": {"frac": sl.FRACS, "C": [0.1, 1.0, 10.0], "eps": [0.05, 0.2, 0.5], "gc": [0.1, 0.3, 1.0], "degree": [2, 3]},
}


def build_factory(kernel):
    def build(hp, task, cfg, y):
        from sklearn.svm import SVR, LinearSVR
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        if kernel == "linear":
            est = LinearSVR(C=hp["C"], epsilon=hp["eps"], loss="epsilon_insensitive", dual=True, max_iter=20000, tol=1e-3, random_state=cfg.seed)
        elif kernel == "rbf":
            est = SVR(kernel="rbf", C=hp["C"], epsilon=hp["eps"], gamma=hp["gc"] / hp["k"], cache_size=1000, max_iter=300000)
        else:
            est = SVR(kernel="poly", degree=hp["degree"], coef0=1.0, C=hp["C"], epsilon=hp["eps"], gamma=hp["gc"] / hp["k"], cache_size=1000, max_iter=300000)
        return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return build


def diag(model, hp, Xr, y, Xq, yq, task):
    d = {}
    if yq is not None:
        f = model.predict(Xq)
        eps_y = hp["eps"] * float(model.transformer_.scale_[0])
        d["eps_loss"] = sl.eps_loss(yq, f, eps_y)
        d["mae"] = float(np.mean(np.abs(yq - f)))
    return d


def make(kernel, name):
    capped = (lambda hp: kernel != "linear")

    def make_grid(cfg, prep, Xs, ys):
        return sl.with_k(sl.sample_configs(SPACES[kernel], cfg.n_configs_svm, cfg.seed), prep.p)

    def complexity(hp):
        return sl.drop_penalty(hp) + 10 * np.log10(hp["C"]) + (10 * np.log10(hp["gc"]) if "gc" in hp else 0) - 5 * hp["eps"]

    def extra(ctx):
        S = ctx["store"]
        model, hp = S["model"], S["hp"]
        Xf, yf = S["Xfit"], S["yfit"]
        reg = model.regressor_
        scale = float(model.transformer_.scale_[0])
        eps_y = hp["eps"] * scale
        f_tr = model.predict(Xf)
        if hasattr(reg, "support_"):
            n_sv = int(len(reg.support_))
        else:
            n_sv = int(np.sum(np.abs(yf - f_tr) >= eps_y))                       # LinearSVR: points on / outside the eps-tube
        res = {"n_support_vectors": n_sv, "support_fraction": n_sv / len(yf), "eps_in_y_units": eps_y,
               "test_eps_insensitive_loss": sl.eps_loss(ctx["y_te"], ctx["p_te"], eps_y), "test_mae": float(np.mean(np.abs(ctx["y_te"] - ctx["p_te"]))),
               "validation_eps_insensitive_loss": sl.eps_loss(ctx["y_va"], ctx["p_va"], eps_y), "fit_rows": int(len(yf)), "kernel": kernel}
        ctx["summary"]["svr"] = res
        (ctx["out"] / "svr_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
        cu = ctx["curve"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for e, d in cu.groupby("eps"):
            b = d.groupby("C")["cv_mse"].min()
            ax[0].plot(b.index, b.values, "o-", label=f"eps = {e}")
        ax[0].set_xscale("log"); ax[0].set_xlabel("C = 1/lambda"); ax[0].set_ylabel("best CV MSE"); ax[0].legend(); ax[0].set_title(f"{name}: CV MSE vs C")
        if "cv_eps_loss" in cu:
            ax[1].scatter(cu["cv_mse"], cu["cv_eps_loss"], s=14); ax[1].set_xlabel("CV MSE"); ax[1].set_ylabel("CV eps-insensitive loss (own eps)"); ax[1].set_title("MSE vs the training loss")
        fig.tight_layout(); fig.savefig(ctx["out"] / "svr_cv_curves.png", dpi=120); plt.close(fig)

    def run(prep_dir, out_dir, cfg, progress=None, console=False):
        task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
        if task != "regression":
            raise ValueError("SVM for regression is a regression method - run it with --task regression")
        return common.run_method(name, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=sl.make_predict_path(build_factory(kernel), sl.margin_score, capped, diag),
                                 complexity=complexity, extra=extra, select_by="mse", notes=f"scikit-learn {'LinearSVR' if kernel == 'linear' else 'SVR'} ({kernel} kernel), standardised response.", console=console)
    return make_grid, run
