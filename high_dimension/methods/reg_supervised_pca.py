"""
Supervised principal components (regression) - scikit-learn pipeline [variance filter -> (winsorise) -> standardise -> SelectKBest(f_regression) -> PCA -> LinearRegression]; the screening is
refitted inside every CV fold (selecting genes with the response BEFORE cross-validation would bias the error estimate, as the book warns).
Steps (Bair et al.): score each gene's univariate association with y, keep the k highest (the threshold theta), take the leading m principal components of the kept genes, regress y on them.
Loss: squared error of the final regression. Tuning: k in {10 ... 2000} x m in {1 ... 6} by repeated K-fold CV, one-SE rule (fewest genes and components).
Metrics: test MSE +- SE and R2 (+ MAE), CV MSE over (k, m), number of genes kept. For censored survival outcomes the concordance index would replace MSE (not used here).
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import hd_lib as hl
from hd_runner import run_method

NAME, TASK = "supervised_pca", "regression"
GRID = {"screen__k": [10, 25, 50, 100, 250, 500, 1000, 2000], "pca__n_components": [1, 2, 3, 4, 5, 6]}


def make_est(wins, out):
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    return hl.make_pipeline(hl.prefix_steps(wins) + [("screen", SelectKBest(f_regression)), ("pca", PCA(random_state=0)), ("lm", LinearRegression())], out)


def complexity(p):
    return p["screen__k"] + 50 * p["pca__n_components"]


def n_features(model, p):
    return int(model.named_steps["screen"].get_support().sum())


def extra(ctx):
    m, out = ctx["model"], ctx["out"]
    keep = np.where(m.named_steps["var"].get_support())[0]
    sc = m.named_steps["screen"]
    sel = keep[sc.get_support()]
    df = pd.DataFrame({"gene": np.array(ctx["genes"])[sel], "F_score": sc.scores_[sc.get_support()]}).sort_values("F_score", ascending=False)
    df.to_csv(out / "kept_genes.csv", index=False)
    ctx["summary"]["supervised_pca"] = {"genes_kept": int(len(sel)), "components": int(ctx["best"]["pca__n_components"]), "explained_variance_ratio": m.named_steps["pca"].explained_variance_ratio_.round(4).tolist()}


def run(bundle_dir, out_dir, cfg, prog):
    return run_method(NAME, TASK, bundle_dir, out_dir, cfg, prog, make_est, GRID, complexity, n_features, extra, notes="screen genes by F-score, PCA, linear regression - all inside the CV folds.")
