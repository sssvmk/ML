"""
compare.py - cross-method comparison on ONE common held-out test set and winner announcement (both tasks).

regression     : winner = lowest test MSE of the raw target (+- SE). Paired tests vs the winner on per-point squared errors: Diebold-Mariano (time-ordered rows, Newey-West variance)
                 and paired t, Holm-adjusted. References: predict-the-train-mean and OLS on the 10 top-ranked variables.
classification : winner = highest test AUC (+- DeLong SE) (every method yields a ranking score). Error (+- SE, validation-tuned Youden threshold) and log-loss (+- SE) are reported for
                 all methods (MARS after Platt calibration, PRIM = box class-1 rates). Paired tests vs the winner: DeLong (AUC), McNemar (error), paired t on per-point log-loss, all Holm-adjusted.
                 Reference: base-rate model.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import common
from common import Prepared, delong_parts, dm_test, holm, logloss_vec, paired_t


def _dot(df, col, se, label, path, title, ref=None, higher_better=False):
    d = df.sort_values(col, ascending=higher_better)
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(d) + 1.8))
    ax.errorbar(d[col], range(len(d)), xerr=d[se] if se else None, fmt="o", capsize=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["method"]); ax.grid(axis="x", alpha=0.3); ax.set_xlabel(label); ax.set_title(title)
    if ref is not None:
        ax.axvline(ref[0], color="grey", ls="--", label=ref[1]); ax.legend()
    b = df.sort_values(col, ascending=not higher_better).iloc[0]
    ax.scatter([b[col]], [list(d["method"]).index(b["method"])], s=170, facecolors="none", edgecolors="gold", linewidths=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def compare_regression(res, prep, methods, alpha, logger):
    y = prep.y("test")
    P, rows = {}, []
    base = float(prep.y("train").mean())
    P["baseline_mean"] = np.full(len(y), base)
    from sklearn.linear_model import RidgeCV
    rg = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(np.asarray(prep.X("train"), float), prep.y_fit_train)           # ridge baseline on ALL engineered variables
    P["baseline_ridge"] = rg.predict(np.asarray(prep.X("test"), float))
    for nm in ("baseline_mean", "baseline_ridge"):
        e2 = (y - P[nm]) ** 2
        rows.append({"method": nm, "kind": "baseline", "status": "ok", "test_mse": e2.mean(), "test_mse_se": e2.std(ddof=1) / np.sqrt(len(y))})
    for m in methods:
        if not (res / m / "summary.json").exists():
            rows.append({"method": m, "kind": "method", "status": "FAILED/missing"}); continue
        s = json.loads((res / m / "summary.json").read_text())
        P[m] = np.load(res / m / "test_pred.npy"); t = s["test"]
        rows.append({"method": m, "kind": "method", "status": "ok", "test_mse": t["mse"], "test_mse_se": t["mse_se"], "test_rmse": t["rmse"], "test_mae": t["mae"],
                     "rel_mse_vs_mean_baseline": t["rel_mse_vs_mean_baseline"], "test_eps_insensitive_loss": s.get("svr", {}).get("test_eps_insensitive_loss"), "support_fraction": s.get("svr", {}).get("support_fraction"), "val_mse": s["validation"]["mse"], "cv_mse": s["cv"]["mse"], "cv_se": s["cv"]["se"], "k_variables": s["k_variables"], "loss_function": s["selected"].get("loss", s["selected"].get("objective", "squared_error")), "n_iterations_M": s["selected"].get("M"),
                     "selected": json.dumps(s["selected"]), "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].copy()
    elig = ok[ok["kind"] == "method"].sort_values("test_mse")
    if elig.empty:
        logger("no regression method finished"); return None
    win = elig.iloc[0]["method"]
    e2w = (y - P[win]) ** 2
    recs = []
    for m in ok["method"]:
        if m == win:
            recs.append({"method": m}); continue
        e2 = (y - P[m]) ** 2
        dd, ds, dp = dm_test(e2w, e2)
        recs.append({"method": m, "mse_diff_winner_minus_other": dd, "dm_stat": ds, "dm_p": dp, "paired_t_p": paired_t(e2w, e2)[2]})
    pw = pd.DataFrame(recs)
    mk = pw["method"] != win
    pw.loc[mk, "dm_p_holm"] = holm(pw.loc[mk, "dm_p"].values); pw.loc[mk, "paired_t_p_holm"] = holm(pw.loc[mk, "paired_t_p"].values)
    ok = ok.merge(pw, on="method", how="left")
    ok["not_significantly_worse_than_winner"] = (ok["method"] == win) | (ok["dm_p_holm"] > alpha)
    ok = ok.sort_values("test_mse").reset_index(drop=True)
    pd.concat([ok, tab[tab["status"] != "ok"]], ignore_index=True).to_csv(res / "comparison.csv", index=False)
    ties = ok.loc[(ok["kind"] == "method") & ok["not_significantly_worse_than_winner"], "method"].tolist()
    w = ok[ok["method"] == win].iloc[0]
    dmb = dm_test((y - P["baseline_mean"]) ** 2, e2w)
    lines = [f"WINNER (lowest test MSE of raw logerror): {win}",
             f"   test MSE {w['test_mse']:.6f} +- {w['test_mse_se']:.6f} | RMSE {w['test_rmse']:.5f} | {w['rel_mse_vs_mean_baseline']:.4f} x the predict-the-mean MSE | k = {int(w['k_variables'])} variables",
             f"Not significantly worse than the winner (Holm-adjusted Diebold-Mariano p > {alpha}): {', '.join(ties)}",
             f"Winner vs predict-the-mean baseline: {'significantly better' if dmb[0] > 0 and dmb[2] < alpha else 'NOT significantly better'} (DM p = {dmb[2]:.4f})"]
    _dot(ok, "test_mse", "test_mse_se", "test MSE of raw logerror (+- SE)", res / "comparison_test_mse.png", "Regression: SVM for regression variants (test set)", ref=(float(np.mean((y - base) ** 2)), "predict train mean"))
    robust = ok[ok["loss_function"].isin(["absolute_error", "huber", "regression_l1"])] if "loss_function" in ok else ok.iloc[0:0]
    for _, r in robust.iterrows():
        lines.append(f"NOTE: {r['method']} was trained with a robust loss ({r['loss_function']}); it competes on MSE and may lose by design - its test MAE is {r['test_mae']:.5f} (winner: {w['test_mae']:.5f}).")
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + ok.drop(columns=["selected"]).to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": win, "tied_with_winner": ties, "test_mse": w["test_mse"], "test_mse_se": w["test_mse_se"]}, indent=2))
    for l in lines:
        logger(l)
    return ok


def compare_classification(res, prep, methods, alpha, logger, winner_metric="auc"):
    y = prep.y("test").astype(int)
    P, T, rows = {}, {}, []
    for m in methods:
        if not (res / m / "summary.json").exists():
            rows.append({"method": m, "status": "FAILED/missing"}); continue
        s = json.loads((res / m / "summary.json").read_text())
        P[m] = np.load(res / m / "test_pred.npy"); T[m] = s["test"]["threshold"]; t = s["test"]
        rows.append({"method": m, "status": "ok", "test_auc": t["auc"], "test_auc_se": t["auc_se"], "test_error": t["error"], "test_error_se": t["error_se"], "test_log_loss": t["log_loss"],
                     "test_log_loss_se": t["log_loss_se"], "brier": t["brier"], "ece": t["ece"], "majority_class_error": t["majority_class_error"], "null_log_loss": t["null_log_loss"],
                     "threshold": t["threshold"], "sensitivity": t["sensitivity"], "specificity": t["specificity"], "k_variables": s["k_variables"], "n_iterations_M": s["selected"].get("M"), "selected": json.dumps(s["selected"]),
                     "platt_calibrated": s["platt_scaling"] is not None, "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    key, asc = ("test_auc", False) if winner_metric == "auc" else ("test_log_loss", True)
    ok = tab[tab["status"] == "ok"].sort_values(key, ascending=asc).reset_index(drop=True)
    if ok.empty:
        logger("no classification method finished"); return None
    win = ok.loc[0, "method"]
    llw = logloss_vec(y, P[win])
    a1, a01, a10 = delong_parts(y, P[win])
    recs = []
    for m in ok["method"]:
        if m == win:
            recs.append({"method": m}); continue
        ll = logloss_vec(y, P[m])
        ca, cb = (P[win] > T[win]) == y, (P[m] > T[m]) == y
        b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
        a2, b01, b10 = delong_parts(y, P[m])
        c01, c10 = np.cov(a01, b01), np.cov(a10, b10)
        var = (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / len(a01) + (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / len(a10)
        recs.append({"method": m, "auc_diff_winner_minus_other": float(a1 - a2), "delong_p_auc": float(2 * (1 - norm.cdf(abs((a1 - a2) / np.sqrt(var))))) if var > 1e-18 else 1.0,
                     "mcnemar_p_error": 1.0 if b_ + c_ == 0 else float(binomtest(b_, b_ + c_, 0.5).pvalue), "logloss_diff_winner_minus_other": float(np.mean(llw - ll)),
                     "paired_t_p_logloss": paired_t(llw, ll)[2]})
    pw = pd.DataFrame(recs)
    mk = pw["method"] != win
    for c in ("delong_p_auc", "mcnemar_p_error", "paired_t_p_logloss"):
        pw.loc[mk, c + "_holm"] = holm(pw.loc[mk, c].values)
    ok = ok.merge(pw, on="method", how="left")
    pc = "delong_p_auc_holm" if winner_metric == "auc" else "paired_t_p_logloss_holm"
    ok["not_significantly_worse_than_winner"] = (ok["method"] == win) | (ok[pc] > alpha)
    pd.concat([ok, pd.DataFrame([{"method": "baseline_base_rate", "status": "ok", "test_log_loss": float(ok["null_log_loss"].iloc[0]), "test_auc": 0.5,
                                  "test_error": float(ok["majority_class_error"].iloc[0])}]), tab[tab["status"] != "ok"]], ignore_index=True).to_csv(res / "comparison.csv", index=False)
    ties = ok.loc[ok["not_significantly_worse_than_winner"], "method"].tolist()
    w = ok.iloc[0]
    lines = [f"WINNER (highest test AUC): {win}" if winner_metric == "auc" else f"WINNER (lowest test log-loss): {win}",
             f"   AUC {w['test_auc']:.5f} +- {w['test_auc_se']:.5f} | error {w['test_error']:.5f} +- {w['test_error_se']:.5f} (always-majority {w['majority_class_error']:.5f}) | "
             f"log-loss {w['test_log_loss']:.5f} +- {w['test_log_loss_se']:.5f} (base rate {w['null_log_loss']:.5f}) | k = {int(w['k_variables'])} variables",
             f"Not significantly worse than the winner (Holm-adjusted {'DeLong' if winner_metric == 'auc' else 'paired t'} p > {alpha}): {', '.join(ties)}",
             f"Lowest error: {ok.sort_values('test_error').iloc[0]['method']} ({ok['test_error'].min():.5f}) | lowest log-loss: {ok.sort_values('test_log_loss').iloc[0]['method']} ({ok['test_log_loss'].min():.5f})"]
    _dot(ok, "test_auc", "test_auc_se", "test AUC (+- DeLong SE)", res / "comparison_auc.png", "Classification: SVMs and flexible discriminants (test set)", higher_better=True)
    _dot(ok, "test_error", "test_error_se", "test error (+- SE)", res / "comparison_error.png", "Error at validation-tuned threshold", ref=(float(ok["majority_class_error"].iloc[0]), "always majority class"))
    _dot(ok, "test_log_loss", "test_log_loss_se", "test log-loss (+- SE)", res / "comparison_log_loss.png", "Log-loss", ref=(float(ok["null_log_loss"].iloc[0]), "base rate"))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m in ok["method"]:
        o = np.argsort(-P[m], kind="stable"); ys = y[o]
        ax.plot(np.cumsum(1 - ys) / (1 - ys).sum(), np.cumsum(ys) / ys.sum(), lw=1.3, label=m)
    ax.plot([0, 1], [0, 1], "k:"); ax.legend(fontsize=7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Test ROC")
    fig.tight_layout(); fig.savefig(res / "comparison_roc.png", dpi=130); plt.close(fig)
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + ok.drop(columns=["selected"]).to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": win, "criterion": winner_metric, "tied_with_winner": ties, "auc": w["test_auc"], "auc_se": w["test_auc_se"], "error": w["test_error"], "log_loss": w["test_log_loss"]}, indent=2))
    for l in lines:
        logger(l)
    return ok


def compare(results_dir, prep_dir, task, methods, alpha=0.05, winner_metric="auc", logger=print):
    res, prep = Path(results_dir), Prepared(prep_dir)
    if task == "regression":
        return compare_regression(res, prep, methods, alpha, logger)
    return compare_classification(res, prep, methods, alpha, logger, winner_metric)
