"""
L1 logistic regression / lasso (classification) - scikit-learn multinomial LogisticRegression with an L1 penalty (saga) on standardised genes: minimise the negative multinomial log-likelihood + (1/C) sum_kj |b_kj|.
Tuning: 12 values of C by repeated stratified CV, one-SE rule (smallest C). Metrics: test error +- SE, log-loss +- SE, macro one-vs-rest AUC, CV error vs C, number of genes with a nonzero coefficient in any class,
selection STABILITY over stratified 80 % sub-samples.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "l1_logreg", "classification"
GRID = {"lr__C": list(np.logspace(-2, 1.5, 12))}


def make_est(wins, out):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    kw = dict(l1_ratio=1.0) if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8) else dict(penalty="l1")
    return hl.make_pipeline(hl.prefix_steps(wins) + [("lr", LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, **kw))], out)


def support(pipe, p):
    m = np.zeros(p)
    idx = np.where(pipe.named_steps["var"].get_support())[0]
    m[idx[np.any(pipe.named_steps["lr"].coef_ != 0, axis=0)]] = 1
    return m


def n_features(model, p):
    return int(np.any(model.named_steps["lr"].coef_ != 0, axis=0).sum())


def extra(ctx):
    m, out = ctx["model"], ctx["out"]
    freq = hl.stability_selection(lambda X, y: ctx["make"]().fit(X, y), ctx["Xf"], ctx["yf"], support, ctx["cfg"].stability_runs, ctx["cfg"].seed, stratify=True)
    sel = support(m, ctx["p"]).astype(bool)
    df = pd.DataFrame({"gene": ctx["genes"], "selection_frequency": freq, "selected_in_final_model": sel}).sort_values(["selected_in_final_model", "selection_frequency"], ascending=False)
    df.head(60).to_csv(out / "selected_genes_and_stability.csv", index=False)
    ctx["summary"]["stability"] = {"genes_ever_selected": int((freq > 0).sum()), "genes_selected_in_>=80%_of_runs": int((freq >= 0.8).sum()), "runs": ctx["cfg"].stability_runs}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: p["lr__C"], n_features, extra, notes="saga L1 multinomial logistic regression")
