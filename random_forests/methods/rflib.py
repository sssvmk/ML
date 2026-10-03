"""
rflib.py - glue around scikit-learn random forests (RandomForestRegressor / Classifier, ExtraTreesRegressor / Classifier). All numerical work is done by scikit-learn:
fitting, OOB predictions (oob_prediction_ / oob_decision_function_), split-based importance (feature_importances_), permutation importance (sklearn.inspection.permutation_importance),
leaf co-occurrence (apply) and MDS (sklearn.manifold.MDS). The glue only loops over the fitted trees (curves vs number of trees) and accumulates the proximity matrix.

Variable policy (as in the earlier chapters): k = round(frac * p) fold-ranked variables is tuned together with the forest hyper-parameters; frac < 1 counts as MORE complex, so every variable is
kept unless dropping some is clearly better by the one-SE rule.
"""
import _bootstrap  # noqa: F401
import itertools
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common

FRACS = [1.0, 0.5, 0.25]


def sample_configs(space, n, seed):
    keys = list(space)
    combos = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.RandomState(seed)
    if n and len(combos) > n:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n, replace=False))]
    cfgs = [dict(zip(keys, c)) for c in combos]
    if "frac" in space and 1.0 in space["frac"] and not any(c["frac"] == 1.0 for c in cfgs):
        cfgs[-1] = {**cfgs[-1], "frac": 1.0}
    return cfgs


def with_k(configs, p):
    return [{**c, "k": max(1, int(round(c["frac"] * p)))} for c in configs]


def drop_penalty(hp):
    return 1e6 * (1.0 - hp["frac"])


def mf_value(hp):
    """m / p as a number (for ordering): 'sqrt' -> 0.05."""
    return 0.05 if hp["mf"] == "sqrt" else float(hp["mf"])


def build_forest(kind, hp, task, cfg, final):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
    kw = dict(n_estimators=cfg.rf_trees_final if final else cfg.rf_trees_cv, max_features=hp["mf"], min_samples_leaf=hp["leaf"], n_jobs=cfg.n_jobs, random_state=cfg.seed)
    reg = task == "regression"
    if kind == "et":
        kw["bootstrap"] = False                                              # standard ExtraTrees: no bootstrap, hence no OOB
        cls = ExtraTreesRegressor if reg else ExtraTreesClassifier
    else:
        kw.update(bootstrap=True, oob_score=True)                            # bootstrap samples + OOB predictions
        cls = RandomForestRegressor if reg else RandomForestClassifier
    if not reg:
        kw["class_weight"] = hp.get("cw")
    return cls(**kw)


def oob_diag(model, hp, Xr, y, Xq, yq, task):
    """OOB metrics of a fitted bootstrap forest (averaged over the CV folds by the driver)."""
    if not hasattr(model, "oob_prediction_") and not hasattr(model, "oob_decision_function_"):
        return {}
    if task == "regression":
        return {"oob_mse": float(np.mean((y - model.oob_prediction_) ** 2))}
    p = model.oob_decision_function_[:, 1]
    ok = np.isfinite(p)
    return {"oob_auc": common.auc_score(y[ok], p[ok]), "oob_log_loss": float(common.logloss_vec(y[ok], common.clip_prob(p[ok])).mean())}


def make_predict_path(kind):
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        import warnings
        task, order = ctx["task"], ctx["order"]
        out = np.empty((len(grid), len(Xq)))
        met = {}
        for h, hp in enumerate(grid):
            cols = order[:hp["k"]]
            Xr = np.asarray(Xref[:, cols], np.float32)
            y = np.asarray(yref).astype(int) if task == "classification" else np.asarray(yref, float)
            model = build_forest(kind, hp, task, cfg, bool(ctx.get("final")))
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")                              # "some inputs do not have OOB scores" with few trees
                model.fit(Xr, y)
            sec = time.time() - t0
            Xqk = np.asarray(Xq[:, cols], np.float32)
            out[h] = model.predict(Xqk) if task == "regression" else model.predict_proba(Xqk)[:, 1]
            for k_, v in oob_diag(model, hp, Xr, y, Xqk, None, task).items():
                met.setdefault(k_, np.full(len(grid), np.nan))[h] = v
            if ctx.get("final"):
                ctx["store"].update(model=model, cols=cols, hp=hp, task=task, kind=kind, n_fit_rows=len(y))
                ctx["store"]["summary"] = {"fit_seconds_final": round(sec, 1), "n_fit_rows": int(len(y)), "n_features_used": int(len(cols)), "n_trees": int(model.n_estimators),
                                           "max_features_effective": int(model.estimators_[0].max_features_),
                                           **({"oob_score_sklearn": float(model.oob_score_)} if hasattr(model, "oob_score_") else {})}
        for k_, v in met.items():
            ctx["metrics"][k_] = v
        return out
    return predict_path


def tree_curves(model, X_eval, y_eval, task, Xs=None, ys=None, n_points=30):
    """Performance vs number of trees from ONE fitted forest: running average of the per-tree predictions on the evaluation rows, and (bootstrap forests) the OOB curve
    from estimators_samples_ (the in-bag indices of every tree)."""
    B = len(model.estimators_)
    Bs = sorted(set(np.unique(np.linspace(1, B, n_points).astype(int))) | {B})
    reg = task == "regression"
    cum = np.zeros(len(X_eval))
    oob_sum = oob_cnt = None
    if Xs is not None and getattr(model, "bootstrap", False):
        oob_sum, oob_cnt = np.zeros(len(Xs)), np.zeros(len(Xs))
    rows = []
    for b, tree in enumerate(model.estimators_, 1):
        cum += tree.predict(X_eval) if reg else tree.predict_proba(X_eval)[:, 1]
        if oob_sum is not None:
            mask = np.ones(len(Xs), bool)
            mask[model.estimators_samples_[b - 1]] = False
            idx = np.where(mask)[0]
            if len(idx):
                oob_sum[idx] += tree.predict(Xs[idx]) if reg else tree.predict_proba(Xs[idx])[:, 1]
                oob_cnt[idx] += 1
        if b in Bs:
            p = cum / b
            r = {"trees": b}
            if reg:
                r["eval_mse"] = float(np.mean((y_eval - p) ** 2))
            else:
                r["eval_auc"], r["eval_log_loss"] = common.auc_score(y_eval, p), float(common.logloss_vec(y_eval, common.clip_prob(p)).mean())
            if oob_sum is not None:
                ok = oob_cnt > 0
                po = oob_sum[ok] / oob_cnt[ok]
                if reg:
                    r["oob_mse"] = float(np.mean((ys[ok] - po) ** 2))
                else:
                    r["oob_auc"], r["oob_log_loss"] = common.auc_score(ys[ok], po), float(common.logloss_vec(ys[ok], common.clip_prob(po)).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def proximity_matrix(model, X, n_trees):
    """P[i, j] = fraction of trees in which rows i and j fall in the same terminal node (scikit-learn apply())."""
    leaves = model.apply(X)[:, :n_trees]
    n = len(X)
    P = np.zeros((n, n), np.float32)
    for b in range(leaves.shape[1]):
        P += leaves[:, b][:, None] == leaves[:, b][None, :]
    return P / leaves.shape[1]


def mds_2d(D, seed):
    from sklearn.manifold import MDS
    kw = dict(n_components=2, n_init=1, max_iter=100, random_state=seed)
    try:
        return MDS(metric="precomputed", init="random", **kw).fit_transform(D)
    except TypeError:
        return MDS(dissimilarity="precomputed", **kw).fit_transform(D)


def permutation_importance_threads(model, X, y, scoring, n_repeats, seed, n_jobs):
    """scikit-learn permutation_importance executed on THREADS. With the default (loky) backend the fitted forest - often several GB - is pickled to every worker process, which fails for large forests
    (PicklingError) and leaves loky worker processes to be killed on shutdown (PermissionError on Windows). Threads share the forest; tree prediction releases the GIL."""
    from joblib import parallel_backend
    from sklearn.inspection import permutation_importance
    with parallel_backend("threading", n_jobs=n_jobs):
        return permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)


def cap_rows(n, cap):
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).astype(int))


def standard_extra(ctx, name):
    """Importances (split-based + permutation), curves vs trees (validation + OOB), proximity plot, calibrated votes (classification)."""
    S, cfg = ctx["store"], ctx["cfg"]
    model, cols, hp, task = S["model"], S["cols"], S["hp"], S["task"]
    reg = task == "regression"
    out = ctx["out"]
    names = np.array(ctx["prep"].feature_names)[cols]
    # ---- variable importance: split-based (impurity) and permutation (scikit-learn, on validation rows) ----
    r = cap_rows(len(ctx["y_va"]), cfg.perm_rows)
    Xv = np.asarray(ctx["X_va"][r][:, cols], np.float32)
    yv = ctx["y_va"][r]
    pi = permutation_importance_threads(model, Xv, yv, "neg_mean_squared_error" if reg else "roc_auc", cfg.perm_repeats, cfg.seed, cfg.n_jobs)
    imp = model.feature_importances_
    df = pd.DataFrame({"variable": names, "split_importance": imp, "split_rel_0_100": 100 * imp / max(imp.max(), 1e-300),
                       "permutation_importance": pi.importances_mean, "permutation_sd": pi.importances_std,
                       "permutation_rel_0_100": 100 * np.maximum(pi.importances_mean, 0) / max(pi.importances_mean.max(), 1e-300)}).sort_values("split_importance", ascending=False)
    df.to_csv(out / "variable_importance.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    top = df.head(20)[::-1]
    ax[0].barh(top["variable"], top["split_rel_0_100"]); ax[0].set_title("split-based importance (relative, max = 100)")
    top2 = df.sort_values("permutation_importance", ascending=False).head(20)[::-1]
    ax[1].barh(top2["variable"], top2["permutation_rel_0_100"]); ax[1].set_title(f"permutation importance on {len(yv)} validation rows (relative)")
    fig.suptitle(f"{name}: variable importance"); fig.tight_layout(); fig.savefig(out / "variable_importance.png", dpi=120); plt.close(fig)
    # ---- performance vs number of trees: validation curve (+ OOB curve for bootstrap forests) from a refit on the tuning subsample ----
    Xs = np.asarray(ctx["Xs"][:, cols], np.float32)
    ys = ctx["ys_fit"] if reg else ctx["ys_ev"].astype(int)
    from copy import deepcopy
    small = build_forest(S["kind"], hp, task, cfg, False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small.fit(Xs, ys)
    rv = cap_rows(len(ctx["y_va"]), 8000)
    cur = tree_curves(small, np.asarray(ctx["X_va"][rv][:, cols], np.float32), ctx["y_va"][rv], task, Xs if S["kind"] == "rf" else None, ys if S["kind"] == "rf" else None)
    cur.to_csv(out / "curves_vs_trees.csv", index=False)
    mcol = "eval_mse" if reg else "eval_auc"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(cur["trees"], cur[mcol], "o-", ms=3, label="validation")
    if "oob_mse" in cur or "oob_auc" in cur:
        ax.plot(cur["trees"], cur["oob_mse" if reg else "oob_auc"], "s--", ms=3, label="OOB (tuning subsample)")
    ax.set_xlabel("number of trees B"); ax.set_ylabel("MSE" if reg else "AUC"); ax.legend(); ax.set_title(f"{name}: performance vs number of trees")
    fig.tight_layout(); fig.savefig(out / "curves_vs_trees.png", dpi=120); plt.close(fig)
    # ---- proximity plot: leaf co-occurrence over the trees (apply), drawn in 2-D by MDS ----
    rp = cap_rows(len(ys), cfg.prox_rows)
    Xp = np.asarray(ctx["Xs"][rp][:, cols], np.float32)
    P = proximity_matrix(model, Xp, cfg.prox_trees)
    Z = mds_2d(1.0 - P, cfg.seed)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    col = ctx["ys_ev"][rp] if not reg else np.clip(ys[rp], *np.quantile(ys[rp], [0.02, 0.98]))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=col, s=5, alpha=0.6, cmap="coolwarm")
    ax.set_title(f"{name}: proximity plot ({len(rp)} rows, {min(cfg.prox_trees, model.n_estimators)} trees)"); ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); fig.colorbar(sc, ax=ax, label="class" if not reg else "fit target")
    fig.tight_layout(); fig.savefig(out / "proximity_plot.png", dpi=120); plt.close(fig)
    res = {"n_trees_final": int(model.n_estimators), "max_features_effective": int(model.estimators_[0].max_features_), "min_samples_leaf": hp["leaf"], "n_features_used": int(len(cols)),
           "proximity_rows": int(len(rp)), "permutation_rows": int(len(yv)), "top5_split_importance": df["variable"].head(5).tolist(),
           "top5_permutation_importance": df.sort_values("permutation_importance", ascending=False)["variable"].head(5).tolist()}
    # ---- classification: calibrated votes (Platt scaling fitted on VALIDATION) next to the clipped raw vote fractions ----
    if not reg:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=500).fit(ctx["p_va"][:, None], ctx["y_va"])
        pc = common.clip_prob(lr.predict_proba(ctx["p_te"][:, None])[:, 1])
        res["test_log_loss_clipped_votes"] = ctx["summary"]["test"]["log_loss"]
        res["test_brier_clipped_votes"] = ctx["summary"]["test"]["brier"]
        res["test_log_loss_platt_calibrated"] = float(common.logloss_vec(ctx["y_te"], pc).mean())
        res["test_brier_platt_calibrated"] = float(np.mean((pc - ctx["y_te"]) ** 2))
    if "oob_score_sklearn" in S.get("summary", {}):
        res["oob_score_final_forest"] = S["summary"]["oob_score_sklearn"]
    ctx["summary"]["forest"] = res
    (out / "forest_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
    cu = ctx["curve"]
    col_cv = "cv_mse" if reg else "cv_auc"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lf, d in cu.groupby("leaf"):
        ax[0].scatter(d["mf"].map(lambda v: 0.05 if v == "sqrt" else float(v)), d[col_cv], label=f"min node size {lf}", s=28)
    ax[0].set_xlabel("m / p (sqrt -> 0.05)"); ax[0].set_ylabel("CV " + ("MSE" if reg else "AUC")); ax[0].legend(fontsize=7); ax[0].set_title("CV over (m, minimum node size)")
    ocol = "cv_oob_mse" if reg else "cv_oob_auc"
    if ocol in cu:
        ax[1].scatter(cu[col_cv], cu[ocol], s=22); ax[1].set_xlabel("CV " + ("MSE" if reg else "AUC")); ax[1].set_ylabel("OOB " + ("MSE" if reg else "AUC")); ax[1].set_title("OOB vs CV (configurations)")
    fig.tight_layout(); fig.savefig(out / "cv_over_m_and_node_size.png", dpi=120); plt.close(fig)
