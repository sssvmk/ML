"""
CART trees (ESLII 9.2) - REGRESSION and CLASSIFICATION; the task follows the prepared data.
regression     : greedy binary splits minimising the summed squared error  sum (y_i - c_m)^2  over the two child regions (9.12);
classification : greedy splits minimising Gini node impurity; leaf class proportions are the scores (probabilities for log-loss, clipped).
Both: cost-complexity pruning  C_alpha(T) = sum_m N_m Q_m(T) + alpha |T|  (scikit-learn ccp_alpha; alpha is expressed relative to the root impurity).
Hyper-parameters (CV): k = number of top-ranked variables, alpha. Regression selects on CV MSE; classification on CV AUC (error at a fixed threshold is degenerate
with ~10 % positives) - CV error at the pooled Youden threshold is logged too.
Metrics: regression: test MSE +- SE, CV MSE vs alpha / |T| (one-SE rule), number of leaves, variable importance.
         classification: test error +- SE, AUC, log-loss (proportions clipped), confusion matrix, CV error vs alpha / |T|, ROC under UNEQUAL LOSSES (L01 = 5).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common

NAME = "cart"
KS = [10, 20, 40, 80]
ALPHAS_REL = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def make_grid(cfg, prep, Xs, ys):
    ks = [k for k in KS if k <= prep.p] or [prep.p]
    return [{"k": k, "alpha_rel": a} for k in ks for a in ALPHAS_REL]


def _scale(y, task):
    return float(np.var(y)) if task == "regression" else float(2 * y.mean() * (1 - y.mean()))


def _tree(task, alpha, seed, class_weight=None):
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    if task == "regression":
        return DecisionTreeRegressor(min_samples_leaf=20, ccp_alpha=alpha, random_state=seed)
    return DecisionTreeClassifier(criterion="gini", min_samples_leaf=50, ccp_alpha=alpha, class_weight=class_weight, random_state=seed)


def _predict(tree, X, task):
    return tree.predict(X) if task == "regression" else tree.predict_proba(X)[:, 1]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    task, order = ctx["task"], ctx["order"]
    sc = _scale(yref, task)
    out = np.empty((len(grid), len(Xq)))
    for h, hp in enumerate(grid):
        cols = order[:hp["k"]]
        t = _tree(task, hp["alpha_rel"] * sc, cfg.seed).fit(Xref[:, cols], yref if task == "regression" else yref.astype(int))
        out[h] = _predict(t, Xq[:, cols], task)
        if ctx.get("final"):
            ctx["store"]["tree"], ctx["store"]["cols"], ctx["store"]["task"] = t, cols, task
            ctx["store"]["summary"] = {"n_leaves": int(t.get_n_leaves()), "depth": int(t.get_depth())}
    return out


def diagnostics(Xs, ys, grid, cfg, order):
    task = "regression" if not np.all((ys == 0) | (ys == 1)) else "classification"
    sc = _scale(ys, task)
    leaves = []
    for hp in grid:
        cols = order[:hp["k"]]
        leaves.append(_tree(task, hp["alpha_rel"] * sc, cfg.seed).fit(Xs[:, cols], ys if task == "regression" else ys.astype(int)).get_n_leaves())
    return {"complexity": np.array(leaves, float), "n_leaves": leaves}


def extra(ctx):
    t, cols, task = ctx["store"]["tree"], ctx["store"]["cols"], ctx["store"]["task"]
    names = np.array(ctx["prep"].feature_names)[cols]
    imp = pd.DataFrame({"variable": names, "importance": t.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(ctx["out"] / "variable_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp.head(15)[::-1]
    ax.barh(top["variable"], top["importance"]); ax.set_title("CART variable importance (final tree)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "variable_importance.png", dpi=120); plt.close(fig)
    if t.get_n_leaves() <= 40:
        from sklearn.tree import export_text
        (ctx["out"] / "tree_rules.txt").write_text(export_text(t, feature_names=list(names), max_depth=8))
    if task == "classification":                      # ROC under unequal losses: class-1 observations weighted L01 = 5 (ESLII Sec. 9.2.5)
        hp = ctx["grid"][ctx["i_sel"]]
        sc = _scale(ctx["y_fit_tr"], task)
        t5 = _tree(task, hp["alpha_rel"] * sc, ctx["cfg"].seed, class_weight={0: 1, 1: 5}).fit(np.asarray(ctx["X_tr"][:, cols]), ctx["y_tr"].astype(int))
        res, fig, ax = {}, *plt.subplots(figsize=(5.5, 5))
        for lab, model in (("symmetric losses", t), ("L01 = 5 (class 1 errors cost 5x)", t5)):
            p = _predict(model, np.asarray(ctx["X_te"][:, cols]), task)
            o = np.argsort(-p, kind="stable"); y = ctx["y_te"][o]
            ax.plot(np.cumsum(1 - y) / (1 - y).sum(), np.cumsum(y) / y.sum(), label=f"{lab} (AUC {common.auc_score(ctx['y_te'], p):.4f})")
            res[lab] = {"auc": common.auc_score(ctx["y_te"], p), "n_leaves": int(model.get_n_leaves())}
        ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8); ax.set_title("CART: ROC under unequal losses (test)")
        fig.tight_layout(); fig.savefig(ctx["out"] / "roc_unequal_losses.png", dpi=120); plt.close(fig)
        (ctx["out"] / "unequal_losses.json").write_text(json.dumps(res, indent=2))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    prep_task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, diagnostics=diagnostics, extra=extra,
                             select_by="mse" if prep_task == "regression" else "auc", notes="scikit-learn CART + cost-complexity pruning; top-k variables from the fold-wise ranking.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
