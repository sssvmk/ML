"""
Lasso (regression) - scikit-learn Lasso in a pipeline [variance filter -> (winsorise) -> standardise -> Lasso], all refitted inside every CV fold.
Loss: sum_i (y_i - b0 - sum_j x_ij b_j)^2 + lambda sum_j |b_j|   (in scikit-learn's 1/(2n) scaling). With p >> N at most N genes can be selected.
Tuning: 36 values of lambda by repeated K-fold CV, one-SE rule (largest lambda within one corrected SE).
Metrics: test MSE +- SE (+ MAE, R2), CV MSE vs lambda, number of nonzero coefficients, coefficient path, selected-variable STABILITY over 80 % sub-samples (selection frequency per gene).
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "lasso", "regression"
GRID = {"lasso__alpha": list(np.logspace(-3.2, -0.1, 32))}


def make_est(wins, out):
    from sklearn.linear_model import Lasso
    return hl.make_pipeline(hl.prefix_steps(wins) + [("lasso", Lasso(max_iter=100000, tol=1e-6))], out)


def complexity(p):
    return -p["lasso__alpha"]


def support(pipe, p):
    m = np.zeros(p)
    idx = np.where(pipe.named_steps["var"].get_support())[0]
    m[idx[np.abs(pipe.named_steps["lasso"].coef_) > 0]] = 1
    return m


def n_features(model, p):
    return int(np.count_nonzero(model.named_steps["lasso"].coef_))


def extra(ctx):
    from sklearn.linear_model import lasso_path
    m, out, genes = ctx["model"], ctx["out"], np.array(ctx["genes"])
    Xs = m[:-1].transform(ctx["Xf"])
    yc = ctx["yf"] - ctx["yf"].mean()
    alphas = np.array(sorted(GRID["lasso__alpha"], reverse=True))
    _, coefs, _ = lasso_path(Xs, yc, alphas=alphas)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j in np.argsort(-np.abs(coefs[:, -1]))[:25]:
        ax[0].plot(alphas, coefs[j], lw=1)
    ax[0].set_xscale("log"); ax[0].invert_xaxis(); ax[0].set_xlabel("lambda"); ax[0].set_ylabel("coefficient (standardised gene)"); ax[0].axvline(ctx["best"]["lasso__alpha"], color="k", ls=":"); ax[0].set_title("lasso path (25 largest coefficients)")
    ax[1].plot(alphas, (coefs != 0).sum(0), "o-", ms=3); ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_xlabel("lambda"); ax[1].set_ylabel("nonzero coefficients"); ax[1].set_title("sparsity vs lambda (<= N)")
    fig.tight_layout(); fig.savefig(out / "lasso_path.png", dpi=120); plt.close(fig)
    freq = hl.stability_selection(lambda X, y: ctx["make"]().fit(X, y), ctx["Xf"], ctx["yf"], support, ctx["cfg"].stability_runs, ctx["cfg"].seed)
    sel = np.zeros(ctx["p"], bool); sel[np.where(m.named_steps["var"].get_support())[0][np.abs(m.named_steps["lasso"].coef_) > 0]] = True
    coef_full = np.zeros(ctx["p"]); coef_full[np.where(m.named_steps["var"].get_support())[0]] = m.named_steps["lasso"].coef_
    df = pd.DataFrame({"gene": genes, "coefficient_standardised": coef_full, "selection_frequency": freq, "selected_in_final_model": sel}).sort_values(["selected_in_final_model", "selection_frequency"], ascending=False)
    df.head(60).to_csv(out / "selected_genes_and_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.hist(freq[freq > 0], bins=20); ax.set_xlabel("selection frequency over sub-samples"); ax.set_ylabel("genes"); ax.set_title("lasso selection stability")
    fig.tight_layout(); fig.savefig(out / "selection_stability.png", dpi=120); plt.close(fig)
    ctx["summary"]["stability"] = {"genes_ever_selected": int((freq > 0).sum()), "genes_selected_in_>=80%_of_runs": int((freq >= 0.8).sum()), "runs": ctx["cfg"].stability_runs}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, complexity, n_features, extra, notes="scikit-learn Lasso, one-SE rule.")
