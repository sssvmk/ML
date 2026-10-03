"""
Supervised principal components (classification) - scikit-learn pipeline [variance filter -> (winsorise) -> standardise -> SelectKBest(f_classif) -> PCA -> LogisticRegression]; the screening is refitted inside every
CV fold (selecting genes with the labels BEFORE cross-validation would bias the error, as the book warns). Loss: multinomial log-likelihood of the logistic regression on the components.
Tuning: k in {10 ... 1000} genes x m in {1 ... 5} components by repeated stratified CV, one-SE rule (fewest genes and components). Metrics: test error +- SE, log-loss +- SE, macro one-vs-rest AUC, CV error over (k, m), genes kept.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "supervised_pca", "classification"
GRID = {"screen__k": [10, 25, 50, 100, 250, 500, 1000], "pca__n_components": [1, 2, 3, 4, 5]}


def make_est(wins, out):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    return hl.make_pipeline(hl.prefix_steps(wins) + [("screen", SelectKBest(f_classif)), ("pca", PCA(random_state=0)), ("lr", LogisticRegression(C=10.0, max_iter=5000))], out)


def n_features(model, p):
    return int(model.named_steps["screen"].get_support().sum())


def extra(ctx):
    m = ctx["model"]
    keep = np.where(m.named_steps["var"].get_support())[0]
    sc = m.named_steps["screen"]
    sel = keep[sc.get_support()]
    pd.DataFrame({"gene": np.array(ctx["genes"])[sel], "F_score": sc.scores_[sc.get_support()]}).sort_values("F_score", ascending=False).to_csv(ctx["out"] / "kept_genes.csv", index=False)
    ctx["summary"]["supervised_pca"] = {"genes_kept": int(len(sel)), "components": int(ctx["best"]["pca__n_components"]), "explained_variance_ratio": m.named_steps["pca"].explained_variance_ratio_.round(4).tolist()}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, lambda p: p["screen__k"] + 50 * p["pca__n_components"], n_features, extra, notes="screen, PCA, logistic regression")
