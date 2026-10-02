"""
compare.py - cross-method comparison on ONE common test set and winner announcement (both tasks).

regression     : winner = lowest test MSE of the raw logerror (+- SE). Paired tests vs the winner on per-point squared errors: Diebold-Mariano (rows are in time order,
                 Newey-West variance) and paired t, Holm-adjusted. Reference rows: predict-the-train-mean and a straight line in the chosen predictor.
classification : winner = lowest test LOG-LOSS (+- SE) - every method yields probabilities. Error (+- SE, validation-tuned Youden threshold) and AUC (+- DeLong SE) are
                 reported next to it. Paired tests vs the winner: t-test on per-point log-loss, McNemar (error), DeLong (AUC), Holm-adjusted. Reference: base-rate model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import common
from common import Prepared, dm_test, holm, paired_t, logloss_vec, delong_parts


def _dot_plot(df, col, se, label, path, title, ref=None, ascending=True):
    d = df.sort_values(col, ascending=not ascending)
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(d) + 1.8))
    ax.errorbar(d[col], range(len(d)), xerr=d[se], fmt="o", capsize=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["method"]); ax.grid(axis="x", alpha=0.3); ax.set_xlabel(label); ax.set_title(title)
    if ref is not None:
        ax.axvline(ref[0], color="grey", ls="--", label=ref[1]); ax.legend()
    best = df.sort_values(col, ascending=ascending).iloc[0]
    ax.scatter([best[col]], [list(d["method"]).index(best["method"])], s=170, facecolors="none", edgecolors="gold", linewidths=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def compare_regression(res, prep, methods, alpha, logger):
    y = prep.y("test")
    preds, rows = {}, []
    base = float(prep.y("train").mean())
    preds["baseline_mean"] = np.full(len(y), base)
    x1tr, x1te = prep.inputs("x1", "train")[:, 0], prep.inputs("x1", "test")[:, 0]
    b = np.linalg.lstsq(np.c_[np.ones(len(x1tr)), x1tr], prep.y_fit_train, rcond=None)[0]
    preds["baseline_linear_in_predictor"] = b[0] + b[1] * x1te
    for nm in ("baseline_mean", "baseline_linear_in_predictor"):
        e2 = (y - preds[nm]) ** 2
        rows.append({"method": nm, "kind": "baseline", "status": "ok", "test_mse": e2.mean(), "test_mse_se": e2.std(ddof=1) / np.sqrt(len(y))})
    for m in methods:
        if not (res / m / "summary.json").exists():
            rows.append({"method": m, "kind": "method", "status": "FAILED/missing"}); continue
        s = json.loads((res / m / "summary.json").read_text())
        preds[m] = np.load(res / m / "test_pred.npy")
        t = s["test"]
        rows.append({"method": m, "kind": "method", "status": "ok", "test_mse": t["mse"], "test_mse_se": t["mse_se"], "test_rmse": t["rmse"], "test_mae": t["mae"],
                     "rel_mse_vs_mean_baseline": t["rel_mse_vs_mean_baseline"], "val_mse": s["validation"]["mse"], "cv_mse": s["cv"]["mse"], "cv_se": s["cv"]["se"],
                     "selected_complexity": s["selected_complexity"], "selected": json.dumps(s["selected"]), "input_dim": s["input_dim"], "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].copy()
    elig = ok[ok["kind"] == "method"].sort_values("test_mse")
    if elig.empty:
        logger("no regression method finished"); return None
    win = elig.iloc[0]["method"]
    e2w = (y - preds[win]) ** 2
    recs = []
    for m in ok["method"]:
        if m == win:
            recs.append({"method": m}); continue
        e2 = (y - preds[m]) ** 2
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
    dmb = dm_test((y - preds["baseline_mean"]) ** 2, e2w)
    lines = [f"WINNER (lowest test MSE of raw logerror): {win}",
             f"   test MSE {w['test_mse']:.6f} +- {w['test_mse_se']:.6f} | RMSE {w['test_rmse']:.5f} | {w['rel_mse_vs_mean_baseline']:.4f} x the predict-the-mean MSE | input dim {int(w['input_dim'])}",
             f"Not significantly worse than the winner (Holm-adjusted Diebold-Mariano p > {alpha}): {', '.join(ties)}",
             f"Winner vs predict-the-mean baseline: {'significantly better' if dmb[0] > 0 and dmb[2] < alpha else 'NOT significantly better'} (DM p = {dmb[2]:.4f})"]
    _dot_plot(ok, "test_mse", "test_mse_se", "test MSE of raw logerror (+- SE)", res / "comparison_test_mse.png", "Regression: kernel methods (test set)",
              ref=(float(np.mean((y - base) ** 2)), "predict train mean"))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xtr = np.r_[prep.inputs("x1", "train")[:, 0], prep.inputs("x1", "val")[:, 0]]; ytr = np.r_[prep.y("train"), prep.y("val")]
    e = np.unique(np.quantile(xtr, np.linspace(0, 1, 41))); bb = np.clip(np.searchsorted(e, xtr, side="right") - 1, 0, len(e) - 2)
    ax.errorbar([xtr[bb == i].mean() for i in range(len(e) - 1)], [ytr[bb == i].mean() for i in range(len(e) - 1)], yerr=[ytr[bb == i].std(ddof=1) / np.sqrt((bb == i).sum()) for i in range(len(e) - 1)], fmt=".", color="grey", label="binned means")
    for m in methods:
        if (res / m / "curve_grid.csv").exists():
            c = pd.read_csv(res / m / "curve_grid.csv"); ax.plot(c["x"], c["fhat"], lw=1.5, label=m)
    ax.set_xlabel(prep.meta.get("x1_label", "predictor")); ax.set_ylabel("logerror"); ax.legend(fontsize=7); ax.set_title("1-D kernel smoothers on the chosen predictor")
    fig.tight_layout(); fig.savefig(res / "comparison_fitted_curves.png", dpi=130); plt.close(fig)
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + ok.drop(columns=["selected"]).to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": win, "tied_with_winner": ties, "test_mse": w["test_mse"], "test_mse_se": w["test_mse_se"]}, indent=2))
    for l in lines:
        logger(l)
    return ok


def compare_classification(res, prep, methods, alpha, logger):
    y = prep.y("test").astype(int)
    P, T, rows = {}, {}, []
    pri = float(prep.y("train").mean())
    for m in methods:
        if not (res / m / "summary.json").exists():
            rows.append({"method": m, "status": "FAILED/missing"}); continue
        s = json.loads((res / m / "summary.json").read_text())
        P[m] = np.load(res / m / "test_pred.npy"); T[m] = s["test"]["threshold"]
        t = s["test"]
        rows.append({"method": m, "status": "ok", "test_log_loss": t["log_loss"], "test_log_loss_se": t["log_loss_se"], "test_error": t["error"], "test_error_se": t["error_se"],
                     "test_auc": t["auc"], "test_auc_se": t["auc_se"], "brier": t["brier"], "ece": t["ece"], "majority_class_error": t["majority_class_error"], "null_log_loss": t["null_log_loss"],
                     "threshold": t["threshold"], "sensitivity": t["sensitivity"], "specificity": t["specificity"], "cv_log_loss": s["cv"]["log_loss"], "input_dim": s["input_dim"],
                     "selected": json.dumps(s["selected"]), "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].sort_values("test_log_loss").reset_index(drop=True)
    if ok.empty:
        logger("no classification method finished"); return None
    win = ok.loc[0, "method"]
    llw = logloss_vec(y, P[win])
    recs = []
    for m in ok["method"]:
        if m == win:
            recs.append({"method": m}); continue
        ll = logloss_vec(y, P[m])
        ca, cb = (P[win] > T[win]) == y, (P[m] > T[m]) == y
        b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
        a1, a01, a10 = delong_parts(y, P[win]); a2, b01, b10 = delong_parts(y, P[m])
        c01, c10 = np.cov(a01, b01), np.cov(a10, b10)
        var = (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / len(a01) + (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / len(a10)
        recs.append({"method": m, "logloss_diff_winner_minus_other": float(np.mean(llw - ll)), "paired_t_p_logloss": paired_t(llw, ll)[2],
                     "mcnemar_p_error": 1.0 if b_ + c_ == 0 else float(binomtest(b_, b_ + c_, 0.5).pvalue), "auc_diff_winner_minus_other": float(a1 - a2),
                     "delong_p_auc": float(2 * (1 - norm.cdf(abs((a1 - a2) / np.sqrt(var))))) if var > 1e-18 else 1.0})
    pw = pd.DataFrame(recs)
    mk = pw["method"] != win
    for c in ("paired_t_p_logloss", "mcnemar_p_error", "delong_p_auc"):
        pw.loc[mk, c + "_holm"] = holm(pw.loc[mk, c].values)
    ok = ok.merge(pw, on="method", how="left")
    ok["not_significantly_worse_than_winner"] = (ok["method"] == win) | (ok["paired_t_p_logloss_holm"] > alpha)
    nullrow = {"method": "baseline_base_rate", "test_log_loss": float(ok["null_log_loss"].iloc[0]), "status": "ok"}
    pd.concat([ok, pd.DataFrame([nullrow]), tab[tab["status"] != "ok"]], ignore_index=True).to_csv(res / "comparison.csv", index=False)
    ties = ok.loc[ok["not_significantly_worse_than_winner"], "method"].tolist()
    w = ok.iloc[0]
    lines = [f"WINNER (lowest test log-loss): {win}",
             f"   log-loss {w['test_log_loss']:.5f} +- {w['test_log_loss_se']:.5f} (base rate {w['null_log_loss']:.5f}) | error {w['test_error']:.5f} +- {w['test_error_se']:.5f} "
             f"(always-majority {w['majority_class_error']:.5f}) | AUC {w['test_auc']:.5f} +- {w['test_auc_se']:.5f}",
             f"Not significantly worse than the winner (Holm-adjusted paired t on per-point log-loss > {alpha}): {', '.join(ties)}",
             f"Best AUC: {ok.sort_values('test_auc', ascending=False).iloc[0]['method']} ({ok['test_auc'].max():.5f}) | lowest error: {ok.sort_values('test_error').iloc[0]['method']} ({ok['test_error'].min():.5f})"]
    if w["test_log_loss"] >= w["null_log_loss"]:
        lines.append("NOTE: the winner does not beat the base-rate log-loss.")
    _dot_plot(ok, "test_log_loss", "test_log_loss_se", "test log-loss (+- SE)", res / "comparison_log_loss.png", "Classification: kernel methods (test set)", ref=(float(ok["null_log_loss"].iloc[0]), "base rate"))
    _dot_plot(ok, "test_auc", "test_auc_se", "test AUC (+- DeLong SE)", res / "comparison_auc.png", "AUC", ascending=False)
    _dot_plot(ok, "test_error", "test_error_se", "test error (+- SE)", res / "comparison_error.png", "Error at validation-tuned threshold", ref=(float(ok["majority_class_error"].iloc[0]), "always majority class"))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m in ok["method"]:
        o = np.argsort(-P[m], kind="stable"); ys = y[o]
        ax.plot(np.cumsum(1 - ys) / (1 - ys).sum(), np.cumsum(ys) / ys.sum(), lw=1.3, label=m)
    ax.plot([0, 1], [0, 1], "k:"); ax.legend(fontsize=7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Test ROC")
    fig.tight_layout(); fig.savefig(res / "comparison_roc.png", dpi=130); plt.close(fig)
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + ok.drop(columns=["selected"]).to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": win, "tied_with_winner": ties, "test_log_loss": w["test_log_loss"], "test_log_loss_se": w["test_log_loss_se"], "auc": w["test_auc"], "error": w["test_error"]}, indent=2))
    for l in lines:
        logger(l)
    return ok


def compare(results_dir, prep_dir, task, methods, alpha=0.05, logger=print):
    res, prep = Path(results_dir), Prepared(prep_dir)
    return (compare_regression if task == "regression" else compare_classification)(res, prep, methods, alpha, logger)
