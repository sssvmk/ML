"""
hd_compare.py - cross-method comparison on the common test set.
regression     winner = lowest test MSE +- SE among the (non-baseline) methods; paired t-tests (and Diebold-Mariano, small-sample corrected) on the per-sample squared errors against the winner, Holm-adjusted;
               report: the winner, every entry not significantly worse, genes used by each entry, and whether the winner beats the ridge / mean baselines.
classification winner = lowest test ERROR +- SE (ties: lower log-loss, then fewer genes); exact McNemar tests against the winner, Holm-adjusted; paired t-tests on per-sample log-loss among the methods that give
               probabilities (against the best log-loss), Holm-adjusted; report: the winner, every method not significantly worse, the sparsest of those, the majority-class baseline.
With 12 (regression) or 20 (classification) test samples these tests have very little power: 'not significantly worse' mostly means 'cannot be told apart'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hd_lib as hl


def _load(res, methods):
    S, P = {}, {}
    for m in methods:
        f = Path(res) / m / "summary.json"
        if f.exists():
            S[m] = json.loads(f.read_text()); P[m] = pd.read_csv(Path(res) / m / "predictions_test.csv")
    return S, P


def compare(res_dir, task, methods, alpha=0.05, logger=print):
    res = Path(res_dir)
    S, P = _load(res, methods)
    if not S:
        logger("no method results to compare"); return None
    return (_reg if task == "regression" else _clf)(res, S, P, alpha, logger)


def _reg(res, S, P, alpha, log):
    rows = []
    for m, s in S.items():
        t, v = s["test"], s.get("validation") or {}
        rows.append({"method": m, "baseline": s["baseline"], "test_mse": t["mse"], "test_mse_se": t["se"], "test_rmse": t["rmse"], "test_mae": t["mae"], "test_r2_vs_mean": t["r2_vs_train_mean"],
                     "val_mse": v.get("mse"), "cv_mse": s["cv"]["selected_loss"], "genes_used": s["n_features_used"], "selected": json.dumps(s["selected"])})
    tab = pd.DataFrame(rows).sort_values("test_mse")
    elig = tab[~tab["baseline"]]
    win = elig.iloc[0]["method"]
    others = [m for m in tab["method"] if m != win]
    pt = [hl.paired_t(P[m]["sq_err"].values, P[win]["sq_err"].values) for m in others]
    dm = [hl.dm_test(P[m]["sq_err"].values, P[win]["sq_err"].values)[1] for m in others]
    adj = hl.holm(pt)
    cmp_ = pd.DataFrame({"method": others, "test_mse": [S[m]["test"]["mse"] for m in others], "p_paired_t": pt, "p_holm": adj, "p_DM_HLN": dm,
                         "significantly_worse": [(a < alpha) and (S[m]["test"]["mse"] > S[win]["test"]["mse"]) for m, a in zip(others, adj)]})
    ties = [win] + [m for m, w in zip(others, cmp_["significantly_worse"]) if not w and not S[m]["baseline"]]
    tab = tab.merge(cmp_[["method", "p_paired_t", "p_holm", "significantly_worse"]], on="method", how="left")
    tab.to_csv(res / "comparison.csv", index=False)
    w = S[win]["test"]
    base = {m: cmp_.loc[cmp_["method"] == m].iloc[0] for m in ("ridge", "mean") if m in cmp_["method"].values}
    lines = [f"WINNER (lowest test MSE, non-baseline methods): {win}", f"   test MSE {w['mse']:.4f} +- {w['se']:.4f} | RMSE {w['rmse']:.4f} | MAE {w['mae']:.4f} | R2 vs mean {w['r2_vs_train_mean']:.3f} | genes used {S[win]['n_features_used']}",
             f"Not significantly worse than the winner (Holm-adjusted paired t, alpha={alpha}; n_test={w['n']}): {', '.join(ties)}"]
    for b, r in base.items():
        lines.append(f"Winner vs {b} baseline: test MSE {S[win]['test']['mse']:.4f} vs {S[b]['test']['mse']:.4f}; {'significantly' if (r['p_holm'] < alpha and S[b]['test']['mse'] > S[win]['test']['mse']) else 'NOT significantly'} better (Holm p = {r['p_holm']:.3f})")
    lines.append("Genes used: " + ", ".join(f"{m} {S[m]['n_features_used']}" for m in tab["method"]))
    lines.append("CAUTION: 12 test samples - the test MSE standard errors are large and the tests have little power.")
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + tab.drop(columns=["selected"]).round(5).to_string(index=False))
    (res / "winner.json").write_text(json.dumps(hl.jsonable({"winner": win, "not_significantly_worse": ties, "test_mse": w["mse"], "se": w["se"]}), indent=2))
    for l in lines:
        log(l)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t2 = tab.sort_values("test_mse")
    ax.bar(t2["method"], t2["test_mse"], yerr=t2["test_mse_se"], color=["lightgray" if b else "tab:blue" for b in t2["baseline"]], capsize=3); ax.set_ylabel("test MSE +- SE"); ax.set_title("Riboflavin: test MSE (grey = baselines)"); plt.xticks(rotation=30)
    fig.tight_layout(); fig.savefig(res / "comparison_mse.png", dpi=120); plt.close(fig)
    return tab


def _clf(res, S, P, alpha, log):
    rows = []
    for m, s in S.items():
        t = s["test"]
        rows.append({"method": m, "baseline": s["baseline"], "test_error": t["error"], "test_error_se": t["se_error"], "n_errors": t["n_errors"], "log_loss": t.get("log_loss"), "log_loss_se": t.get("se_log_loss"),
                     "macro_ovr_auc": t.get("macro_ovr_auc"), "cv_error": s["cv"]["selected_loss"], "genes_used": s["n_features_used"], "selected": json.dumps(s["selected"])})
    tab = pd.DataFrame(rows)
    elig = tab[~tab["baseline"]].assign(_ll=lambda d: d["log_loss"].fillna(1e9)).sort_values(["test_error", "_ll", "genes_used"])
    win = elig.iloc[0]["method"]
    others = [m for m in tab["method"] if m != win]
    cw = P[win]["correct"].values.astype(bool)
    pm = [hl.mcnemar_exact(P[m]["correct"].values.astype(bool), cw) for m in others]
    adj = hl.holm(pm)
    cmp_ = pd.DataFrame({"method": others, "test_error": [S[m]["test"]["error"] for m in others], "p_mcnemar": pm, "p_holm": adj,
                         "significantly_worse": [(a < alpha) and (S[m]["test"]["error"] > S[win]["test"]["error"]) for m, a in zip(others, adj)]})
    ties = [win] + [m for m, w in zip(others, cmp_["significantly_worse"]) if not w and not S[m]["baseline"]]
    spars = min(ties, key=lambda m: (S[m]["n_features_used"], S[m]["test"]["error"]))
    # paired t on per-sample log-loss among the probabilistic methods, against the best log-loss
    pl = [m for m in tab["method"] if "log_loss" in P[m]]
    ll_tab = None
    if len(pl) > 1:
        bl = min(pl, key=lambda m: S[m]["test"]["log_loss"])
        o2 = [m for m in pl if m != bl]
        p2 = [hl.paired_t(P[m]["log_loss"].values, P[bl]["log_loss"].values) for m in o2]
        ll_tab = pd.DataFrame({"method": o2, "vs_best_log_loss": bl, "log_loss": [S[m]["test"]["log_loss"] for m in o2], "p_paired_t": p2, "p_holm": hl.holm(p2)})
        ll_tab.to_csv(res / "comparison_log_loss_tests.csv", index=False)
    tab = tab.merge(cmp_[["method", "p_mcnemar", "p_holm", "significantly_worse"]], on="method", how="left").sort_values(["test_error"])
    tab.to_csv(res / "comparison.csv", index=False)
    cmp_.to_csv(res / "comparison_mcnemar.csv", index=False)
    w = S[win]["test"]
    maj = S.get("majority")
    lines = [f"WINNER (lowest test error, non-baseline methods): {win}", f"   test error {w['error']:.4f} +- {w['se_error']:.4f} ({w['n_errors']} of {w['n']} test samples wrong)" + (f" | log-loss {w['log_loss']:.4f}" if "log_loss" in w else "") + f" | genes used {S[win]['n_features_used']}",
             f"Not significantly worse than the winner (Holm-adjusted exact McNemar, alpha={alpha}; n_test={w['n']}): {', '.join(ties)}", f"Sparsest of those: {spars} ({S[spars]['n_features_used']} genes, test error {S[spars]['test']['error']:.4f})"]
    if maj:
        lines.append(f"Majority-class baseline: test error {maj['test']['error']:.4f}" + (f", log-loss {maj['test']['log_loss']:.4f}" if "log_loss" in maj["test"] else ""))
    if ll_tab is not None:
        lines.append(f"Best log-loss: {ll_tab['vs_best_log_loss'].iloc[0]} ({S[ll_tab['vs_best_log_loss'].iloc[0]]['test']['log_loss']:.4f}); methods not significantly worse on log-loss (Holm paired t): " + ", ".join([ll_tab['vs_best_log_loss'].iloc[0]] + ll_tab.loc[ll_tab["p_holm"] >= alpha, "method"].tolist()))
    lines.append("Genes used: " + ", ".join(f"{m} {S[m]['n_features_used']}" for m in tab["method"]))
    lines.append("CAUTION: 20 test samples - one error is 5 percentage points; the tests have very little power. DeLong AUC tests are binary-only, so the four-class AUC is only reported (macro one-vs-rest).")
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + tab.drop(columns=["selected"]).round(5).to_string(index=False))
    (res / "winner.json").write_text(json.dumps(hl.jsonable({"winner": win, "not_significantly_worse": ties, "sparsest_of_those": spars, "test_error": w["error"]}), indent=2))
    for l in lines:
        log(l)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    t2 = tab.sort_values("test_error")
    ax.bar(t2["method"], t2["test_error"], yerr=t2["test_error_se"], color=["lightgray" if b else "tab:blue" for b in t2["baseline"]], capsize=3); ax.set_ylabel("test error +- SE"); ax.set_title("SRBCT: test error (grey = baseline)"); plt.xticks(rotation=30)
    fig.tight_layout(); fig.savefig(res / "comparison_error.png", dpi=120); plt.close(fig)
    return tab
