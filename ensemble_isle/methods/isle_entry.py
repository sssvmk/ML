"""
Factory for the three post-processed ensemble entries (scikit-learn throughout):

isle_gbm  Importance sampled learning ensemble (ESLII 16.3.1, Alg. 16.2). Stage 1: GradientBoosting with subsample eta (<= 1/2), shrinkage nu, J-leaf trees; the trees T_m(x) are the dictionary.
          Stage 2 (16.9): regression  min sum (y - a0 - sum_m a_m T_m(x))^2 + lambda sum |a_m|  (Lasso path);  classification  the same with the binomial deviance (L1 logistic path).
isle_rf   The same post-processing of a random-forest dictionary (trees grown on subsamples of size eta*N, shallow, random variable subsets), Friedman & Popescu's 'RF (5%, 6)' construction.
rulefit   Rule ensemble (16.3.2): every node of every tree of a boosted ensemble is a rule (16.14-16.15), optionally plus winsorised linear terms; lasso / L1-logistic post-processing:
          min sum L(y, a0 + sum_k a_k r_k(x) + sum_j b_j x_j) + lambda (sum |a_k| + sum |b_j|).
Tuning (driver CV: time-blocked folds for Zillow, stratified for Santander; paired one-SE rule): eta, nu, tree size J, number of trees M, [mf, linear terms], the penalty lambda (as lambda / lambda_max, or C = C_min / lamr) and the
fraction of fold-ranked variables kept. Regression selects on CV MSE of the fit target, classification on CV AUC. The final model uses <= isle_final_rows training rows.
Outputs: CV metric and the number of selected trees / rules vs lambda, trees kept vs the full ensemble, test performance vs number of trees against the unprocessed own ensemble and a forest / GBM reference,
sensitivity to eta, nu and J, and (RuleFit) rules-only vs rules + linear terms, rule and variable importance, rule length and support.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import isle_lib as il

SPACES = {
    "isle_gbm": {"frac": il.FRACS, "eta": [0.1, 0.25, 0.5], "nu": [0.05, 0.1], "J": [3, 6], "M": [150, 300]},
    "isle_rf": {"frac": il.FRACS, "eta": [0.05, 0.1, 0.25, 0.5], "J": [4, 6, 10], "mf": [0.1, 0.3, 0.5], "M": [150, 300]},
    "rulefit": {"frac": il.FRACS, "eta": [0.25, 0.5], "nu": [0.1], "J": [3, 4, 6], "M": [100, 200], "linear": [True, False]},
}


def stage1_factory(flavour):
    def stage1(conf, task, cfg):
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
        reg = task == "regression"
        if flavour == "isle_rf":
            cls = RandomForestRegressor if reg else RandomForestClassifier
            return cls(n_estimators=conf["M"], max_samples=conf["eta"], max_leaf_nodes=conf["J"], max_features=conf["mf"], bootstrap=True, n_jobs=cfg.n_jobs, random_state=cfg.seed)
        kw = dict(subsample=conf["eta"], learning_rate=conf["nu"], max_leaf_nodes=conf["J"], n_estimators=conf["M"], random_state=cfg.seed)
        return GradientBoostingRegressor(loss="squared_error", **kw) if reg else GradientBoostingClassifier(loss="log_loss", **kw)
    return stage1


def reference_curve(ctx, flavour, cols, Xte, yte, M):
    """The comparison ensemble, fitted on the diagnostic subsample: a random forest (for ISLE-GBM) or a gradient-boosting model (for ISLE-RF)."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    cfg, task = ctx["cfg"], ctx["store"]["task"]
    reg = task == "regression"
    r0 = il.cap_rows(len(ctx["ys_fit"]), cfg.diag_rows)
    Xd = np.asarray(ctx["Xs"][r0][:, cols], np.float32)
    yd = ctx["ys_fit"][r0] if reg else ctx["ys_ev"][r0].astype(int)
    M = int(min(M, 200))
    if flavour == "isle_rf":
        gb = (GradientBoostingRegressor(loss="squared_error", learning_rate=0.1, max_leaf_nodes=6, subsample=0.5, n_estimators=M, random_state=cfg.seed) if reg
              else GradientBoostingClassifier(loss="log_loss", learning_rate=0.1, max_leaf_nodes=6, subsample=0.5, n_estimators=M, random_state=cfg.seed)).fit(Xd, yd)
        return "unprocessed GBM (reference, diagnostic subsample)", il.staged_curve(gb, Xte, yte, task)
    rf = (RandomForestRegressor(n_estimators=M, min_samples_leaf=5, max_features=1 / 3, n_jobs=cfg.n_jobs, random_state=cfg.seed) if reg
          else RandomForestClassifier(n_estimators=M, min_samples_leaf=5, max_features="sqrt", n_jobs=cfg.n_jobs, random_state=cfg.seed)).fit(Xd, yd)
    return "random forest (reference, diagnostic subsample)", il.running_curve(il.tree_matrix(il.trees_of(rf), Xte), yte, task)


def cv_panels(ctx, name):
    cu = ctx["curve"]
    reg = ctx["store"]["task"] == "regression"
    col = "cv_mse" if reg else "cv_auc"
    best = (lambda s: s.min()) if reg else (lambda s: s.max())
    groups = [g for g in ("eta", "nu", "J", "mf", "linear") if g in cu and cu[g].nunique() > 1]
    fig, ax = plt.subplots(1, 2 + len(groups), figsize=(4.6 * (2 + len(groups)), 4.2))
    b = cu.groupby("lamr")[col].agg(best)
    ax[0].plot(b.index, b.values, "o-"); ax[0].set_xscale("log"); ax[0].set_xlabel("lambda / lambda_max" if reg else "1 / (C / C_min)"); ax[0].set_ylabel("best " + col); ax[0].set_title("CV vs penalty")
    if "cv_n_active" in cu:
        ax[1].plot(cu.groupby("lamr")["cv_n_active"].mean().index, cu.groupby("lamr")["cv_n_active"].mean().values, "o-"); ax[1].set_xscale("log"); ax[1].set_xlabel("penalty"); ax[1].set_ylabel("selected trees / rules (CV mean)"); ax[1].set_title("sparsity vs penalty")
    for a, g in zip(ax[2:], groups):
        s = cu.groupby(cu[g].astype(str))[col].agg(best)
        a.bar(s.index, s.values); a.set_xlabel(g); a.set_ylabel("best " + col); a.set_title(f"sensitivity to {g}")
        lo, hi = float(s.min()), float(s.max())
        a.set_ylim(lo - 0.5 * (hi - lo + 1e-12) if reg else lo - 0.5 * (hi - lo + 1e-12), hi + 0.5 * (hi - lo + 1e-12))
    fig.suptitle(name); fig.tight_layout(); fig.savefig(ctx["out"] / "cv_vs_penalty_and_sensitivity.png", dpi=120); plt.close(fig)


def isle_extra(ctx, name, flavour):
    S, cfg = ctx["store"], ctx["cfg"]
    D, task, cols, conf = S["D"], S["task"], S["cols"], S["conf"]
    reg = task == "regression"
    rt = il.cap_rows(len(ctx["y_te"]), 10000)
    Xte = np.asarray(ctx["X_te"][rt][:, cols], np.float32)
    yte = ctx["y_te"][rt]
    lamrs = il.LAMR_REG if reg else il.LAMR_CLF
    preds, nnz, _ = il.path_fit(task, S["Ts"], S["y"], [D.transform(Xte)], lamrs, cfg.seed)
    path = pd.DataFrame([{"lamr": l, "trees_kept": int(n), **il.trajectory_metrics(task, yte, p)} for l, n, p in zip(lamrs, nnz, preds[0])])
    path.to_csv(out := ctx["out"] / "isle_path_test.csv", index=False)
    own = il.staged_curve(S["model"], Xte, yte, task) if flavour == "isle_gbm" else il.running_curve(il.tree_matrix(S["trees"], Xte), yte, task)
    ref_name, ref = reference_curve(ctx, flavour, cols, Xte, yte, conf["M"])
    own.to_csv(ctx["out"] / "unprocessed_ensemble_curve.csv", index=False); ref.to_csv(ctx["out"] / "reference_ensemble_curve.csv", index=False)
    mcol = "test_mse" if reg else "test_log_loss"
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(own["trees"], own[mcol], "o-", ms=3, label="unprocessed " + ("boosted ensemble" if flavour == "isle_gbm" else "forest ensemble"))
    ax.plot(ref["trees"], ref[mcol], "s--", ms=3, label=ref_name)
    ax.plot(path["trees_kept"], path[mcol], "^-", ms=4, label="ISLE: lasso post-processed (trees kept)")
    ax.set_xlabel("number of trees"); ax.set_ylabel("test MSE" if reg else "test log-loss (deviance)"); ax.legend(fontsize=8); ax.set_title(f"{name}: test performance vs number of trees")
    fig.tight_layout(); fig.savefig(ctx["out"] / "test_vs_number_of_trees.png", dpi=120); plt.close(fig)
    n_act = int(np.count_nonzero(S["coef"]))
    ctx["summary"]["sparsity"] = {"n_active": n_act, "n_dictionary": int(S["Ts"].shape[1]), "fraction_kept": n_act / S["Ts"].shape[1]}
    cv_panels(ctx, name)


def rulefit_extra(ctx, name):
    S = ctx["store"]
    D, task, cols, coef = S["D"], S["task"], S["cols"], S["coef"]
    names = np.array(ctx["prep"].feature_names)[cols]
    meta = il.rule_meta(S["trees"], names)
    nr = D.n_rules
    rc, lc = coef[:nr], coef[nr:]
    nz = np.where(rc != 0)[0]
    imp = np.abs(rc)                                                          # unit-variance basis: |coefficient| = Friedman's importance |a_k| * std(r_k)
    rules = pd.DataFrame({"rule": [meta[i]["text"] for i in nz], "length": [meta[i]["depth"] for i in nz], "support": D.rule_support[nz], "coefficient_standardised": rc[nz], "importance": imp[nz]}).sort_values("importance", ascending=False)
    rules.to_csv(ctx["out"] / "selected_rules.csv", index=False)
    vimp = np.zeros(len(cols))
    for i in nz:
        fs = meta[i]["features"]
        for f in fs:
            vimp[f] += imp[i] / len(fs)
    if D.linear:
        vimp += np.abs(lc)
    vdf = pd.DataFrame({"variable": names, "importance": vimp, "relative_0_100": 100 * vimp / max(vimp.max(), 1e-300)}).sort_values("importance", ascending=False)
    vdf.to_csv(ctx["out"] / "variable_importance.csv", index=False)
    n_lin = int(np.count_nonzero(lc)) if D.linear else 0
    ctx["summary"]["sparsity"] = {"n_active": int(len(nz) + n_lin), "n_dictionary": int(coef.shape[0]), "rules_kept": int(len(nz)), "linear_terms_kept": n_lin, "rules_in_dictionary": int(nr)}
    cu = ctx["curve"]
    reg = task == "regression"
    col = "cv_mse" if reg else "cv_auc"
    best = (lambda s: s.min()) if reg else (lambda s: s.max())
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    top = vdf.head(15)[::-1]
    ax[0].barh(top["variable"], top["relative_0_100"]); ax[0].set_title("variable importance (rules + linear terms)")
    tr = rules.head(15)[::-1]
    ax[1].barh([r[:45] for r in tr["rule"]], tr["importance"]); ax[1].set_title("top rules"); ax[1].tick_params(axis="y", labelsize=6)
    ax[2].hist(rules["length"], bins=np.arange(0.5, rules["length"].max() + 1.5 if len(rules) else 2), alpha=0.7, label="length"); ax[2].set_xlabel("rule length (conditions)"); ax[2].set_title(f"{len(nz)} rules kept")
    if "linear" in cu:
        s = cu.groupby(cu["linear"].astype(str))[col].agg(best)
        ax[3].bar(["rules only" if k == "False" else "rules + linear" for k in s.index], s.values); ax[3].set_ylabel("best " + col); ax[3].set_title("rules-only vs rules + linear terms (CV)")
        lo, hi = float(s.min()), float(s.max()); ax[3].set_ylim(lo - 0.5 * (hi - lo + 1e-12), hi + 0.5 * (hi - lo + 1e-12))
    fig.suptitle(name); fig.tight_layout(); fig.savefig(ctx["out"] / "rulefit_rules_and_importance.png", dpi=120); plt.close(fig)
    cv_panels(ctx, name)


def make(name, flavour):
    kind = "rules" if flavour == "rulefit" else "trees"

    def make_grid(cfg, prep, Xs, ys):
        lamr = il.LAMR_REG if prep.task == "regression" else il.LAMR_CLF
        cfgs = il.with_k(il.sample_configs(SPACES[flavour], cfg.n_configs_isle, cfg.seed), prep.p)
        return [{**c, "lamr": l} for c in cfgs for l in lamr]

    def complexity(hp):
        return il.drop_penalty(hp) - 100.0 * np.log10(hp["lamr"]) + 0.1 * hp["M"]

    def extra(ctx):
        (rulefit_extra(ctx, name) if flavour == "rulefit" else isle_extra(ctx, name, flavour))

    def run(prep_dir, out_dir, cfg, progress=None, console=False):
        task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
        return common.run_method(name, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=il.make_path_predict(stage1_factory(flavour), kind), complexity=complexity, extra=extra,
                                 select_by="mse" if task == "regression" else "auc", notes=f"scikit-learn stage 1 ({flavour}) + {'Lasso' if task == 'regression' else 'L1 logistic'} path.", console=console)
    return make_grid, complexity, run
