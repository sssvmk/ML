"""
boostlib.py - thin glue around PREBUILT boosting libraries (no algorithm code): scikit-learn (GradientBoosting*, AdaBoostClassifier, trees), LightGBM, XGBoost.

* sample_configs / expand     random-search configurations x grid of boosting iterations M (one fit per configuration; all M evaluated from the staged / iteration-limited predictions)
* make_predict_path           the `predict_path` the driver calls (common.run_method): fits one model per (configuration, CV fold) on the top-k variables of that fold's ranking
* predict_stages              predictions after M iterations for sklearn GB (staged_predict[_proba]), AdaBoost (staged_decision_function -> p = 1/(1+exp(-2 f)), ESLII 10.17),
                              LightGBM (num_iteration=M) and XGBoost (iteration_range=(0, M))
* standard_extra              relative variable importance (ESLII 10.13.1), partial-dependence plots, CV curve vs M per group
* staged_curves               training vs test curves along M on the tuning subsample

Variable policy: k = round(frac * p) of the fold-wise ranked variables; frac in {1, 0.5, 0.25} is a hyper-parameter and complexity() ranks frac < 1 as MORE complex, so the one-SE rule keeps all
features unless dropping some is clearly better (features are only dropped when they hinder CV performance).
"""
import _bootstrap  # noqa: F401
import itertools
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
    if len(combos) > n:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n, replace=False))]
    cfgs = [dict(zip(keys, c)) for c in combos]
    if "frac" in space and not any(c["frac"] == 1.0 for c in cfgs):                 # always evaluate "keep every feature"
        cfgs[-1] = {**cfgs[-1], "frac": 1.0}
    return cfgs


def grid_m(m_grid, cap=0):
    return [m for m in m_grid if not cap or m <= cap] or [min(m_grid)]


def expand(configs, m_grid, p, cap=0):
    return [{**c, "k": max(1, int(round(c["frac"] * p))), "M": M} for c in configs for M in grid_m(m_grid, cap)]


def drop_penalty(hp):
    return 1e6 * (1.0 - hp["frac"])


def predict_stages(model, kind, X, Ms, task):
    out = {}
    Mset = set(Ms)
    if kind == "sk":
        it = model.staged_predict(X) if task == "regression" else (p[:, 1] for p in model.staged_predict_proba(X))
        for i, p in enumerate(it, 1):
            if i in Mset:
                out[i] = np.asarray(p, float)
    elif kind == "ada":
        cum = np.cumsum(model.estimator_weights_)
        last = None
        for i, d in enumerate(model.staged_decision_function(X), 1):
            last = 1.0 / (1.0 + np.exp(-2.0 * (cum[min(i, len(cum)) - 1] / 4.0) * d))   # f = (sum alpha / 4) * decision_function = 1/2 sum alpha G ; p = 1/(1+e^-2f)
            if i in Mset:
                out[i] = last
        for M in Ms:                                                                   # AdaBoost stops early on a perfect / degenerate fit: reuse the last fitted stage
            if M not in out and last is not None:
                out[M] = last
    elif kind == "lgb":
        for M in Ms:
            out[M] = model.predict(X, num_iteration=M) if task == "regression" else model.predict_proba(X, num_iteration=M)[:, 1]
    elif kind == "xgb":
        for M in Ms:
            out[M] = model.predict(X, iteration_range=(0, M)) if task == "regression" else model.predict_proba(X, iteration_range=(0, M))[:, 1]
    last = None
    for M in sorted(Ms):                                                           # AdaBoost may stop early: carry the last available stage forward
        if M in out:
            last = out[M]
        elif last is not None:
            out[M] = last
    return out


def cap_rows(n, cap):
    """Systematic (order-preserving) subsample of row positions."""
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).astype(int))


def make_predict_path(build, kind):
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        task, order = ctx["task"], ctx["order"]
        out = np.empty((len(grid), len(Xq)))
        groups = {}
        for h, hp in enumerate(grid):
            groups.setdefault(tuple(sorted((k, v) for k, v in hp.items() if k != "M")), []).append(h)
        for key, idx in groups.items():
            conf = dict(key)
            Ms = sorted({grid[h]["M"] for h in idx})
            cols = order[:conf["k"]]
            y = ctx["yref_raw"] if (task == "regression" and conf.get("fit_target") == "raw") else yref
            y = y.astype(int) if task == "classification" else np.asarray(y, float)
            model = build(conf, max(Ms), task, cfg, y)
            t0 = time.time()
            rows = cap_rows(len(y), cfg.sk_final_rows) if (ctx.get("final") and kind in ("sk", "ada")) else slice(None)       # runtime guard for the exact-greedy methods
            model.fit(np.asarray(Xref[rows][:, cols] if not isinstance(rows, slice) else Xref[:, cols], np.float32), y[rows])
            fit_s = time.time() - t0
            preds = predict_stages(model, kind, np.asarray(Xq[:, cols], np.float32), Ms, task)
            for h in idx:
                out[h] = preds[grid[h]["M"]]
            if ctx.get("final"):
                ctx["store"].update(model=model, cols=cols, conf=conf, kind=kind, fit_seconds=fit_s, task=task)
                ctx["store"]["summary"] = {"fit_seconds_final": round(fit_s, 1), "n_features_used": int(len(cols)), "M": int(max(Ms))}
        return out
    return predict_path


def importance(model, kind):
    if kind == "lgb":
        return np.asarray(model.booster_.feature_importance("gain"), float)
    return np.asarray(model.feature_importances_, float)


def standard_extra(ctx, group_col=None, pdp=True):
    S = ctx["store"]
    model, cols, kind, task = S["model"], S["cols"], S["kind"], S["task"]
    names = np.array(ctx["prep"].feature_names)[cols]
    imp = importance(model, kind)
    df = pd.DataFrame({"variable": names, "importance": imp, "relative_importance_0_100": 100 * imp / max(imp.max(), 1e-300)}).sort_values("importance", ascending=False)
    df.to_csv(ctx["out"] / "variable_importance.csv", index=False)
    top = df.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["variable"], top["relative_importance_0_100"]); ax.set_xlabel("relative importance (max = 100)"); ax.set_title("Relative variable importance (final model)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "variable_importance.png", dpi=120); plt.close(fig)
    cu = ctx["curve"]
    col = "cv_mse" if "cv_mse" in cu else "cv_auc"
    if "M" in cu:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        if group_col and group_col in cu:
            for g, d in cu.groupby(group_col):
                b = d.groupby("M")[col].min() if col == "cv_mse" else d.groupby("M")[col].max()
                ax.plot(b.index, b.values, "o-", ms=3, label=f"{group_col} = {g}")
            ax.legend(fontsize=7)
        else:
            b = cu.groupby("M")[col].min() if col == "cv_mse" else cu.groupby("M")[col].max()
            ax.plot(b.index, b.values, "o-", ms=3)
        ax.set_xlabel("boosting iterations M"); ax.set_ylabel("best " + col + " over the other hyper-parameters"); ax.set_title("CV performance vs M")
        fig.tight_layout(); fig.savefig(ctx["out"] / "cv_by_M.png", dpi=120); plt.close(fig)
    if pdp:
        try:
            from sklearn.inspection import PartialDependenceDisplay
            top_idx = [int(i) for i in np.argsort(-imp)[:4]]
            Xsub = np.asarray(ctx["X_tr"][::max(1, len(ctx["y_tr"]) // 3000)][:, cols], np.float32)
            fig, ax = plt.subplots(1, len(top_idx), figsize=(4 * len(top_idx), 3.4))
            PartialDependenceDisplay.from_estimator(model, Xsub, features=top_idx, feature_names=list(names), grid_resolution=20, ax=ax, kind="average")
            fig.suptitle("Partial dependence of the 4 most important variables"); fig.tight_layout(); fig.savefig(ctx["out"] / "partial_dependence.png", dpi=120); plt.close(fig)
        except Exception as e:                                                      # never let a plot break a run
            ctx["log"].warning("partial dependence plot skipped: %s", e)


def staged_curves(ctx, build, kind, Mmax, step=None):
    """Fit the chosen configuration for Mmax iterations on the tuning subsample and return train / test loss along M."""
    S, cfg = ctx["store"], ctx["cfg"]
    conf, cols, task = S["conf"], S["cols"], S["task"]
    r0 = cap_rows(len(ctx["ys_fit"]), cfg.diag_rows)
    Xs, ys = np.asarray(ctx["Xs"][r0][:, cols], np.float32), (ctx["ys_fit"][r0] if task == "regression" else ctx["ys_ev"][r0].astype(int))
    rows = np.random.RandomState(0).choice(len(ctx["y_te"]), min(10000, len(ctx["y_te"])), replace=False)
    Xte, yte = np.asarray(ctx["X_te"][rows][:, cols], np.float32), ctx["y_te"][rows]
    model = build(conf, Mmax, task, cfg, ys)
    t0 = time.time()
    model.fit(Xs, ys)
    secs = time.time() - t0
    Ms = sorted(set(range(1, Mmax + 1, max(1, (step or Mmax // 40)))) | {Mmax})
    ptr, pte = predict_stages(model, kind, Xs, Ms, task), predict_stages(model, kind, Xte, Ms, task)
    rows_out = []
    for M in Ms:
        if task == "regression":
            rows_out.append({"M": M, "train_mse": float(np.mean((ctx["ys_ev"][r0] - ptr[M]) ** 2)), "test_mse": float(np.mean((yte - pte[M]) ** 2))})
        else:
            rows_out.append({"M": M, "train_log_loss": float(common.logloss_vec(ys, ptr[M]).mean()), "test_log_loss": float(common.logloss_vec(yte, pte[M]).mean()),
                             "train_auc": common.auc_score(ys, ptr[M]), "test_auc": common.auc_score(yte, pte[M])})
    return pd.DataFrame(rows_out), secs


def plot_curves(df, path, title):
    cols = [c for c in df.columns if c != "M"]
    metrics = sorted({c.split("_", 1)[1] for c in cols})
    fig, ax = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4))
    ax = np.atleast_1d(ax)
    for a, m in zip(ax, metrics):
        a.plot(df["M"], df["train_" + m], label="training"); a.plot(df["M"], df["test_" + m], label="test"); a.set_xlabel("iterations M"); a.set_ylabel(m); a.legend(); a.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
