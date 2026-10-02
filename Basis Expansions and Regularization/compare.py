"""
compare.py - cross-method comparison on one common test set and winner announcement.

Regression smoothers (raw logerror, MSE): test MSE +- SE, paired Diebold-Mariano test (Newey-West long-run variance; rows are in time order) and paired t-test
on the per-point squared errors against the winner, Holm-adjusted. Reference rows: predict the train mean, and a straight line in the chosen predictor.
Nonparametric logistic regression has a different target (large miss), so it is judged on test LOG-LOSS vs the base-rate model, separately.
The wavelet method's simulated-Doppler results (MSE against the true signal, SURE risk) are appended to the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import Prepared, dm_test, holm, logloss_vec, paired_t

REG = ["regression_splines", "natural_cubic_splines", "smoothing_splines", "b_splines", "thin_plate_splines", "rkhs", "wavelet_smoothing"]


def compare(results_dir, prep_dir, methods, alpha=0.05, logger=print):
    res = Path(results_dir)
    prep = Prepared(prep_dir)
    y = prep.y_test
    preds, rows = {}, []
    base = float(prep.y_train.mean())
    preds["baseline_mean"] = np.full(len(y), base)
    A = np.c_[np.ones(len(prep.x_train)), prep.x_train]
    b = np.linalg.lstsq(A, prep.y_fit_train, rcond=None)[0]
    preds["baseline_linear_in_predictor"] = b[0] + b[1] * prep.x_test
    for nm in ("baseline_mean", "baseline_linear_in_predictor"):
        e2 = (y - preds[nm]) ** 2
        rows.append({"method": nm, "kind": "baseline", "status": "ok", "test_mse": e2.mean(), "test_mse_se": e2.std(ddof=1) / np.sqrt(len(y)), "df": np.nan})
    for m in [m for m in methods if m in REG]:
        d = res / m
        if not (d / "summary.json").exists():
            rows.append({"method": m, "kind": "method", "status": "FAILED/missing"})
            continue
        s = json.loads((d / "summary.json").read_text())
        preds[m] = np.load(d / "test_pred.npy")
        t = s["test"]
        rows.append({"method": m, "kind": "method", "status": "ok", "test_mse": t["mse"], "test_mse_se": t["mse_se"], "test_rmse": t["rmse"], "test_mae": t["mae"],
                     "rel_mse_vs_mean_baseline": t["rel_mse_vs_mean_baseline"], "val_mse": s["validation_train_only_model"]["mse"], "cv_mse": s["cv"]["mse"],
                     "cv_se": s["cv"]["se"], "df": s["selected_df"], "selected": json.dumps(s["selected"]), "inputs": s["inputs"], "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].copy()
    elig = ok[ok["kind"] == "method"].sort_values("test_mse")
    lines = []
    if elig.empty:
        logger("no regression method finished")
    else:
        win = elig.iloc[0]["method"]
        e2w = (y - preds[win]) ** 2
        recs = []
        for m in ok["method"]:
            e2 = (y - preds[m]) ** 2
            if m == win:
                recs.append({"method": m}); continue
            dd, ds, dp = dm_test(e2w, e2)
            td, ts, tp = paired_t(e2w, e2)
            recs.append({"method": m, "mse_diff_winner_minus_other": dd, "dm_stat": ds, "dm_p": dp, "paired_t_p": tp})
        pw = pd.DataFrame(recs)
        mk = pw["method"] != win
        pw.loc[mk, "dm_p_holm"] = holm(pw.loc[mk, "dm_p"].values)
        pw.loc[mk, "paired_t_p_holm"] = holm(pw.loc[mk, "paired_t_p"].values)
        ok = ok.merge(pw, on="method", how="left")
        ok["not_significantly_worse_than_winner"] = (ok["method"] == win) | (ok["dm_p_holm"] > alpha)
        ok = ok.sort_values("test_mse").reset_index(drop=True)
        ok.to_csv(res / "comparison.csv", index=False)
        pd.concat([ok, tab[tab["status"] != "ok"]], ignore_index=True).to_csv(res / "comparison.csv", index=False)
        ties = ok.loc[(ok["kind"] == "method") & ok["not_significantly_worse_than_winner"], "method"].tolist()
        w = ok[ok["method"] == win].iloc[0]
        dm_b = dm_test((y - preds["baseline_mean"]) ** 2, e2w)
        beat = "significantly better" if (dm_b[0] > 0 and dm_b[2] < alpha) else "NOT significantly better"
        lines += [f"WINNER (lowest test MSE of raw logerror, regression smoothers): {win}",
                  f"   test MSE {w['test_mse']:.6f} +- {w['test_mse_se']:.6f} | RMSE {w['test_rmse']:.5f} | {w['rel_mse_vs_mean_baseline']:.4f} x the predict-the-mean MSE | df {w['df']:.1f}",
                  f"Not significantly worse than the winner (Holm-adjusted Diebold-Mariano p > {alpha}): {', '.join(ties)}",
                  f"Winner vs predict-the-mean baseline: {beat} (DM p = {dm_b[2]:.4f}, mean MSE diff baseline - winner = {dm_b[0]:.3e})"]
        lin = ok[ok["method"] == "baseline_linear_in_predictor"].iloc[0]
        lines.append(f"Straight-line baseline in the chosen predictor: test MSE {lin['test_mse']:.6f} +- {lin['test_mse_se']:.6f}")
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(ok) + 1.8))
        d = ok.sort_values("test_mse", ascending=False)
        ax.errorbar(d["test_mse"], range(len(d)), xerr=d["test_mse_se"], fmt="o", capsize=3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["method"]); ax.axvline(base_mse := float(np.mean((y - base) ** 2)), color="grey", ls="--", label="predict train mean")
        ax.scatter([w["test_mse"]], [list(d["method"]).index(win)], s=170, facecolors="none", edgecolors="gold", linewidths=2, label="winner")
        ax.grid(axis="x", alpha=0.3); ax.set_xlabel("test MSE of raw logerror (+- SE)"); ax.legend(); ax.set_title("Regression smoothers - test set (Aug 2017 onward)")
        fig.tight_layout(); fig.savefig(res / "comparison_test_mse.png", dpi=130); plt.close(fig)
    # fitted-curve overlay
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.r_[prep.x_train, prep.x_val]; yy = np.r_[prep.y_train, prep.y_val]
    xs, ms, ss = common._binned(x, yy, 40)
    ax.errorbar(xs, ms, yerr=ss, fmt=".", color="grey", alpha=0.8, label="binned means (train+val)")
    for m in [m for m in methods if m in REG and (res / m / "curve_grid.csv").exists()]:
        c = pd.read_csv(res / m / "curve_grid.csv"); ax.plot(c["x"], c["fhat"], lw=1.6, label=m)
    ax.set_xlabel(prep.x_label); ax.set_ylabel("logerror"); ax.legend(fontsize=7); ax.set_title(f"All 1-D smoothers on the chosen predictor ({prep.predictor})")
    fig.tight_layout(); fig.savefig(res / "comparison_fitted_curves.png", dpi=130); plt.close(fig)

    # logistic track
    lj = res / "nonparametric_logistic" / "summary.json"
    if "nonparametric_logistic" in methods and lj.exists():
        s = json.loads(lj.read_text()); t = s["test"]
        p = np.load(res / "nonparametric_logistic" / "test_pred.npy")
        yb = prep.b_test
        d_ll = logloss_vec(yb, np.full(len(yb), prep.b_train.mean())) - logloss_vec(yb, p)
        dm_ = dm_test(logloss_vec(yb, np.full(len(yb), prep.b_train.mean())), logloss_vec(yb, p))
        pt = paired_t(logloss_vec(yb, np.full(len(yb), prep.b_train.mean())), logloss_vec(yb, p))
        res_l = {"test_log_loss": t["log_loss"], "test_log_loss_se": t["log_loss_se"], "null_log_loss": t["null_log_loss"],
                 "improvement_vs_null": t["log_loss_improvement_vs_null"], "dm_p_vs_null": dm_[2], "paired_t_p_vs_null": pt[2], "auc": t["auc"], "auc_se": t["auc_se"],
                 "ece": t["ece"], "error_at_validation_youden_threshold": t["error"], "error_se": t["error_se"], "majority_class_error": t["majority_class_error"], "selected_df": s["selected_df"]}
        (res / "logistic_summary.json").write_text(json.dumps(common._jsonable(res_l), indent=2))
        lines.append(f"Nonparametric logistic (large miss) test log-loss {t['log_loss']:.5f} +- {t['log_loss_se']:.5f} vs base-rate {t['null_log_loss']:.5f} "
                     f"({'significantly better' if dm_[0] > 0 and dm_[2] < alpha else 'NOT significantly better'} than the base rate, DM p = {dm_[2]:.4f}); AUC {t['auc']:.4f}")
    # wavelet simulation
    wj = res / "wavelet_smoothing" / "summary.json"
    if wj.exists():
        s = json.loads(wj.read_text())
        if "simulation" in s:
            lines.append("Wavelet simulation (Doppler, MSE against the TRUE signal): " + ", ".join(f"{k} {v:.2e}" for k, v in s["simulation"]["mean_mse_vs_truth"].items()) +
                         f"; mean SURE risk estimate / actual loss = {s['simulation']['sure_estimate_over_actual_mse']:.2f}")
    (res / "winner.txt").write_text("\n".join(lines) + ("\n\n" + ok.drop(columns=[c for c in ("selected",) if c in ok]).to_string(index=False) if not elig.empty else ""))
    if not elig.empty:
        (res / "winner.json").write_text(json.dumps({"winner": win, "tied_with_winner": ties, "test_mse": w["test_mse"], "test_mse_se": w["test_mse_se"]}, indent=2))
    for l in lines:
        logger(l)
    return ok
