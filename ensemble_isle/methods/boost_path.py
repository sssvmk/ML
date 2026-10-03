"""
Boosting as a regularization path (ESLII 16.2) and the unprocessed boosting reference - scikit-learn GradientBoosting* with staged predictions.
fs_path        epsilon-forward stagewise (Alg. 16.1): shrinkage boosting with a small learning rate nu = epsilon is the book's own approximation of the lasso-like path on the dictionary of J-leaf trees
               (tree boosting with shrinkage 'closely resembles Alg. 16.1', the optimal tree being approximated by greedy induction). Loss: regression squared error; classification exponential loss or
               binomial deviance. CV chooses the stopping point m (one-SE rule), nu and J.
gbm_reference  the same machinery with nu = 0.1: the unprocessed full boosted ensemble used as the reference for ISLE / RuleFit.
Path views: test performance vs the L1 arc length t(m) = sum_{k<=m} nu * sd(T_k) of the path (the coefficient on the unit-variance tree basis is nu * sd(T_k)), the exact lasso (lasso_path / L1 logistic path)
on the SAME tree dictionary drawn on the same axes, shrinking nu to show the epsilon -> 0 limit, coefficient profiles of the lasso path, active trees vs t, and for classification the normalised L1 margin
m(f) = min_i y_i f(x_i) / sum |alpha_k| (16.7) vs the number of trees (alpha_k = nu * max|T_k|).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import isle_lib as il

M_FS = [25, 50, 100, 200, 300, 400, 600]
M_REF = [25, 50, 100, 200, 300, 400]


def build_gb(conf, M, task, cfg):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    kw = dict(learning_rate=conf["nu"], max_leaf_nodes=conf["J"], n_estimators=M, subsample=1.0, random_state=cfg.seed)
    return GradientBoostingRegressor(loss="squared_error", **kw) if task == "regression" else GradientBoostingClassifier(loss=conf.get("loss", "log_loss"), **kw)


def staged_preds(model, X, Ms, task):
    it = model.staged_predict(X) if task == "regression" else (p[:, 1] for p in model.staged_predict_proba(X))
    want = set(Ms)
    return {i: np.asarray(p, float) for i, p in enumerate(it, 1) if i in want}


def make_predict_path():
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        import warnings
        import time
        task, order = ctx["task"], ctx["order"]
        reg = task == "regression"
        out = np.empty((len(grid), len(Xq)))
        groups = {}
        for h, hp in enumerate(grid):
            groups.setdefault(tuple(sorted((k, v) for k, v in hp.items() if k != "M")), []).append(h)
        for key, hs in groups.items():
            conf = dict(key)
            Ms = sorted({grid[h]["M"] for h in hs})
            cols = order[:conf["k"]]
            rows = il.cap_rows(len(yref), cfg.isle_final_rows) if ctx.get("final") else None
            Xr = np.asarray(Xref[:, cols] if rows is None else Xref[rows][:, cols], np.float32)
            y = yref if rows is None else yref[rows]
            y = np.asarray(y, float) if reg else np.asarray(y).astype(int)
            model = build_gb(conf, max(Ms), task, cfg)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xr, y)
            sec = time.time() - t0
            preds = staged_preds(model, np.asarray(Xq[:, cols], np.float32), Ms, task)
            for h in hs:
                out[h] = preds[grid[h]["M"]]
            if ctx.get("final"):
                ctx["store"].update(model=model, conf=conf, cols=cols, task=task, y=y, hp=grid[hs[0]])
                ctx["store"]["summary"] = {"fit_seconds_final": round(sec, 1), "n_fit_rows": int(len(y)), "n_features_used": int(len(cols))}
        return out
    return predict_path


def path_view_extra(ctx, name):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import lasso_path
    S, cfg = ctx["store"], ctx["cfg"]
    conf, cols, task = S["conf"], S["cols"], S["task"]
    reg = task == "regression"
    r0 = il.cap_rows(len(ctx["ys_fit"]), cfg.diag_rows)
    Xd = np.asarray(ctx["Xs"][r0][:, cols], np.float32)
    yd = ctx["ys_fit"][r0] if reg else ctx["ys_ev"][r0].astype(int)
    rt = il.cap_rows(len(ctx["y_te"]), 8000)
    Xte, yte = np.asarray(ctx["X_te"][rt][:, cols], np.float32), ctx["y_te"][rt]
    nu0 = conf["nu"]
    rows, margins = [], []
    models = {}
    for mult in (4, 2, 1):                                                          # epsilon -> 0 at fixed arc length: nu, nu/2, nu/4 with 1, 2, 4 times as many trees
        nu = min(0.5, nu0 * mult)
        M = int(round(50 * 4 / mult))
        gb = build_gb({**conf, "nu": nu}, M, task, cfg).fit(Xd, yd)
        trees = il.trees_of(gb)
        T = il.tree_matrix(trees, Xd)
        t = np.cumsum(nu * T.std(0))
        pr = staged_preds(gb, Xte, range(1, M + 1), task)
        alpha_inf = np.cumsum(nu * np.abs(T).max(0))
        if not reg:
            fst = list(gb.staged_decision_function(Xd))
        for m in range(1, M + 1, max(1, M // 40)):
            r = {"path": f"boosting nu={nu:g}", "trees": m, "L1_arc_length": float(t[m - 1]), "n_active": m, **il.trajectory_metrics(task, yte, pr[m])}
            if not reg:
                f = np.asarray(fst[m - 1]).ravel()
                r["L1_margin"] = il.l1_margin(yd, f, alpha_inf[m - 1])
                r["train_error_sign_f"] = float(np.mean(np.sign(f) != (2 * yd - 1)))
            rows.append(r)
        models[nu] = (gb, trees, T)
    gb, trees, T = models[min(models)]                                              # the dictionary: trees of the smallest-nu model
    sc = StandardScaler().fit(T)
    Ts, Tq = sc.transform(T), sc.transform(il.tree_matrix(trees, Xte))
    if reg:
        alphas, coefs, _ = lasso_path(Ts, yd - yd.mean(), n_alphas=40, eps=1e-3)
        P = Tq @ coefs + yd.mean()
        for j in range(coefs.shape[1]):
            rows.append({"path": "exact lasso (same dictionary)", "trees": int(np.count_nonzero(coefs[:, j])), "L1_arc_length": float(np.abs(coefs[:, j]).sum()), "n_active": int(np.count_nonzero(coefs[:, j])),
                         **il.trajectory_metrics(task, yte, P[:, j])})
        prof = coefs
    else:
        lamrs = il.LAMR_CLF
        preds, nnz, cf = il.logistic_path_fit(Ts, yd, [Tq], lamrs, cfg.seed)
        for j, l in enumerate(lamrs):
            rows.append({"path": "exact L1-logistic path (same dictionary)", "trees": int(nnz[j]), "L1_arc_length": float(np.abs(cf[j][0]).sum()), "n_active": int(nnz[j]), **il.trajectory_metrics(task, yte, preds[0][j])})
        prof = np.column_stack([cf[j][0] for j in range(len(lamrs))])
    df = pd.DataFrame(rows)
    df.to_csv(ctx["out"] / "path_view_curves.csv", index=False)
    mcol = "test_mse" if reg else "test_log_loss"
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for pth, d in df.groupby("path"):
        d = d.sort_values("L1_arc_length")
        ax[0].plot(d["L1_arc_length"], d[mcol], "-" if "lasso" in pth or "L1-logistic" in pth else "o-", ms=2, lw=2.2 if "exact" in pth else 1, label=pth)
        ax[1].plot(d["L1_arc_length"], d["n_active"], "-", label=pth)
    ax[0].set_xlabel("L1 arc length t"); ax[0].set_ylabel("test MSE" if reg else "test log-loss"); ax[0].legend(fontsize=7); ax[0].set_title("boosting path vs exact lasso path (same tree dictionary)")
    ax[1].set_xlabel("L1 arc length t"); ax[1].set_ylabel("active trees"); ax[1].set_title("number of active trees along the path")
    final_t = np.abs(prof).sum(0)
    for j in np.argsort(-np.abs(prof[:, -1]))[:15]:
        ax[2].plot(final_t, prof[j], lw=1)
    ax[2].set_xlabel("L1 norm of the lasso coefficients"); ax[2].set_ylabel("coefficient (unit-variance tree basis)"); ax[2].set_title("lasso coefficient profiles (top 15 trees)")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(ctx["out"] / "path_view.png", dpi=120); plt.close(fig)
    cu = ctx["curve"]
    col = "cv_mse" if reg else "cv_auc"
    sdbar = float(models[min(models)][2].std(0).mean())                              # mean sd of the tree outputs of the chosen-nu dictionary
    fig, ax = plt.subplots(1, 2 if reg else 3, figsize=(12 if reg else 17, 4.4))
    b_ = cu.groupby("M")[col].min() if reg else cu.groupby("M")[col].max()
    ax[0].plot(b_.index, b_.values, "o-"); ax[0].axvline(S["hp"]["M"], color="tab:purple", ls="--", label="one-SE stopping point"); ax[0].set_xlabel("boosting iterations m"); ax[0].set_ylabel(col); ax[0].legend(); ax[0].set_title("CV vs m (stopping point)")
    ax[1].plot(b_.index * min(0.5, nu0) * sdbar, b_.values, "o-")
    ax[1].set_xlabel("approximate L1 arc length t(m) = m * mean(nu * sd T_k)"); ax[1].set_ylabel(col); ax[1].set_title("CV vs arc length")
    if not reg:
        for pth, d in df[df["path"].str.startswith("boosting")].groupby("path"):
            ax[2].plot(d["trees"], d["L1_margin"], label=pth)
        ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlabel("number of trees"); ax[2].set_ylabel("normalised L1 margin m(f)"); ax[2].legend(fontsize=7); ax[2].set_title("L1 margin (16.7): crosses 0 when training error hits 0")
    fig.tight_layout(); fig.savefig(ctx["out"] / "stopping_point_and_margin.png", dpi=120); plt.close(fig)
    ctx["summary"]["sparsity"] = {"n_active": int(S["hp"]["M"]), "n_dictionary": int(S["hp"]["M"])}
    ctx["summary"]["path_view"] = {"nu_chosen": nu0, "stopping_point_M": int(S["hp"]["M"]), "J": conf["J"]}


def reference_extra(ctx, name):
    S = ctx["store"]
    rt = il.cap_rows(len(ctx["y_te"]), 10000)
    Xte, yte = np.asarray(ctx["X_te"][rt][:, S["cols"]], np.float32), ctx["y_te"][rt]
    cur = il.staged_curve(S["model"], Xte, yte, S["task"])
    cur.to_csv(ctx["out"] / "test_vs_number_of_trees.csv", index=False)
    mcol = "test_mse" if S["task"] == "regression" else "test_log_loss"
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(cur["trees"], cur[mcol], "o-", ms=3); ax.set_xlabel("trees"); ax.set_ylabel(mcol); ax.set_title(f"{name}: unprocessed boosted ensemble")
    fig.tight_layout(); fig.savefig(ctx["out"] / "test_vs_number_of_trees.png", dpi=120); plt.close(fig)
    ctx["summary"]["sparsity"] = {"n_active": int(S["hp"]["M"]), "n_dictionary": int(S["hp"]["M"])}


def make(name, mode):
    def make_grid(cfg, prep, Xs, ys):
        reg = prep.task == "regression"
        if mode == "fs_path":
            space = {"frac": il.FRACS, "nu": [0.1, 0.03], "J": [2, 4, 6]}
            if not reg:
                space["loss"] = ["log_loss", "exponential"]
            Ms = M_FS
        else:
            space = {"frac": il.FRACS, "nu": [0.1], "J": [2, 4, 6, 8]}
            Ms = M_REF
        cfgs = il.with_k(il.sample_configs(space, cfg.n_configs_isle, cfg.seed), prep.p)
        return [{**c, "M": M} for c in cfgs for M in Ms]

    def complexity(hp):
        return il.drop_penalty(hp) + hp["M"] * (1 if mode == "fs_path" else hp["J"])

    def extra(ctx):
        (path_view_extra if mode == "fs_path" else reference_extra)(ctx, name)

    def run(prep_dir, out_dir, cfg, progress=None, console=False):
        task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
        return common.run_method(name, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=make_predict_path(), complexity=complexity, extra=extra,
                                 select_by="mse" if task == "regression" else "auc", notes="scikit-learn GradientBoosting with staged predictions.", console=console)
    return make_grid, complexity, run
