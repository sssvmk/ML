"""
compare.py - cross-method comparison on the SAME test rows and winner announcement.

Per (feature set, method): test misclassification error +- SE (validation-tuned Youden threshold), AUC +- SE (DeLong), log-loss +- SE
(probabilistic methods only).  Paired tests against the winner: McNemar (exact) for error, DeLong for AUC, paired mean difference for
log-loss; Holm-adjusted across the comparisons.  Also the effect of the feature set (rich - lean) per method.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from common import Prepared, delong_parts, log_loss_vec, sigmoid


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def delong_pair(y, sa, sb):
    a, a01, a10 = delong_parts(y, sa)
    b, b01, b10 = delong_parts(y, sb)
    m, n = len(a01), len(a10)
    c01, c10 = np.cov(a01, b01), np.cov(a10, b10)
    var = (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / m + (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / n
    diff = a - b
    z = diff / np.sqrt(var) if var > 1e-18 else 0.0
    return diff, float(np.sqrt(max(var, 0))), float(2 * (1 - norm.cdf(abs(z))))


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue))


def compare(results_dir, feature_sets, methods, winner_metric="auc", logger=print):
    res = Path(results_dir)
    y = Prepared(res / "prepared" / feature_sets[0]).y_test
    items, rows = {}, []
    for fs, m in product(feature_sets, methods):
        d = res / fs / m
        if not (d / "summary.json").exists():
            rows.append({"key": f"{fs}/{m}", "feature_set": fs, "method": m, "status": "FAILED/missing"})
            continue
        s = json.loads((d / "summary.json").read_text())
        t = s["test"]
        items[f"{fs}/{m}"] = {"score": np.load(d / "test_scores.npy"), "pred": np.load(d / "test_pred.npy").astype(int), "prob": s["probabilistic"]}
        rows.append({"key": f"{fs}/{m}", "feature_set": fs, "method": m, "status": "ok", "test_error": t["error"], "test_error_se": t["error_se"],
                     "majority_class_error": t["majority_class_error"], "auc": t["auc"], "auc_se": t["auc_se"],
                     "log_loss": t.get("log_loss"), "log_loss_se": t.get("log_loss_se"), "null_log_loss": t.get("null_log_loss"),
                     "threshold": t["threshold"], "sensitivity": t["sensitivity"], "specificity": t["specificity"],
                     "error_at_default_rule": t["error_at_default_rule"], "selected_hp": json.dumps(s["selected_hyperparameters"]),
                     "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].copy()
    if ok.empty:
        logger("no method finished - nothing to compare")
        return None
    asc = winner_metric == "error"
    ok = ok.sort_values("test_error" if asc else "auc", ascending=asc).reset_index(drop=True)
    win = ok.loc[0, "key"]
    wi = items[win]
    pw = []
    for k in ok["key"]:
        if k == win:
            pw.append({"key": k})
            continue
        b, c, p_mc = mcnemar(y, wi["pred"], items[k]["pred"])
        dauc, dse, p_dl = delong_pair(y, wi["score"], items[k]["score"])
        pw.append({"key": k, "mcnemar_winner_only_correct": b, "mcnemar_other_only_correct": c, "mcnemar_p": p_mc,
                   "auc_diff_winner_minus_other": dauc, "auc_diff_se": dse, "delong_p": p_dl})
    pw = pd.DataFrame(pw)
    mask = pw["key"] != win
    pw.loc[mask, "mcnemar_p_holm"] = holm(pw.loc[mask, "mcnemar_p"].values)
    pw.loc[mask, "delong_p_holm"] = holm(pw.loc[mask, "delong_p"].values)
    ok = ok.merge(pw, on="key", how="left")
    pcol = "mcnemar_p_holm" if asc else "delong_p_holm"
    ok["tied_with_winner"] = (ok["key"] == win) | (ok[pcol] > 0.05)

    # log-loss among probabilistic methods
    prob = ok[ok["log_loss"].notna()]
    ll_note = ""
    if len(prob):
        best_ll = prob.sort_values("log_loss").iloc[0]["key"]
        lb = log_loss_vec(y, sigmoid(items[best_ll]["score"]))
        ll_rows = []
        for k in prob["key"]:
            d = log_loss_vec(y, sigmoid(items[k]["score"])) - lb
            se = d.std(ddof=1) / np.sqrt(len(d))
            ll_rows.append({"key": k, "logloss_minus_best": d.mean(), "paired_se": se,
                            "p": 1.0 if k == best_ll else float(2 * (1 - norm.cdf(abs(d.mean() / max(se, 1e-300)))))})
        ll = pd.DataFrame(ll_rows)
        m2 = ll["key"] != best_ll
        ll.loc[m2, "p_holm"] = holm(ll.loc[m2, "p"].values)
        ll.to_csv(res / "logloss_vs_best.csv", index=False)
        ll_note = f"Best log-loss (probabilistic methods): {best_ll} ({prob.set_index('key').loc[best_ll, 'log_loss']:.5f})"

    # feature-set effect
    eff = []
    if len(feature_sets) == 2:
        a, b = feature_sets
        for m in methods:
            ka, kb = f"{a}/{m}", f"{b}/{m}"
            if ka in items and kb in items:
                d, se, p = delong_pair(y, items[kb]["score"], items[ka]["score"])
                _, _, pm = mcnemar(y, items[kb]["pred"], items[ka]["pred"])
                eff.append({"method": m, f"auc_{b}_minus_{a}": d, "se": se, "delong_p": p, "mcnemar_p_error": pm})
        pd.DataFrame(eff).to_csv(res / "feature_set_effect.csv", index=False)

    ok.to_csv(res / "comparison.csv", index=False)
    pd.concat([ok.drop(columns=[c for c in ok.columns if c in ()]), tab[tab["status"] != "ok"]], ignore_index=True).to_csv(res / "comparison.csv", index=False)

    # plots
    for col, se, lab, fname in (("auc", "auc_se", "test AUC (+- DeLong SE)", "comparison_auc.png"), ("test_error", "test_error_se", "test misclassification error (+- SE)", "comparison_error.png")):
        d = ok.sort_values(col, ascending=(col == "test_error"))
        fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(d) + 1.8))
        ax.errorbar(d[col][::-1], range(len(d)), xerr=d[se][::-1], fmt="o", capsize=3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["key"][::-1]); ax.grid(axis="x", alpha=0.3); ax.set_xlabel(lab)
        if col == "test_error":
            ax.axvline(d["majority_class_error"].iloc[0], color="grey", ls="--", label="always predict majority class"); ax.legend()
        fig.tight_layout(); fig.savefig(res / fname, dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for k in ok["key"].head(6):
        s = items[k]["score"]
        o = np.argsort(-s, kind="stable"); ys = y[o]
        ax.plot(np.cumsum(1 - ys) / (1 - ys).sum(), np.cumsum(ys) / ys.sum(), lw=1.3, label=k)
    ax.plot([0, 1], [0, 1], "k:"); ax.legend(fontsize=7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Test ROC - top 6")
    fig.tight_layout(); fig.savefig(res / "comparison_roc_top6.png", dpi=130); plt.close(fig)

    w = ok.loc[0]
    ties = ok.loc[ok["tied_with_winner"], "key"].tolist()
    best_err = ok.sort_values("test_error").iloc[0]
    lines = [f"WINNER (by test {'misclassification error' if asc else 'AUC'}): {win}",
             f"   AUC {w['auc']:.5f} +- {w['auc_se']:.5f} | error {w['test_error']:.5f} +- {w['test_error_se']:.5f} "
             f"(always-majority error {w['majority_class_error']:.5f}) | sensitivity {w['sensitivity']:.3f}, specificity {w['specificity']:.3f}",
             f"Not significantly different from the winner (Holm-adjusted {'McNemar' if asc else 'DeLong'} p > 0.05): {', '.join(ties)}",
             f"Lowest test error: {best_err['key']} ({best_err['test_error']:.5f} +- {best_err['test_error_se']:.5f})"]
    if ll_note:
        lines.append(ll_note)
    for fs in feature_sets:
        sub = ok[ok["feature_set"] == fs]
        if len(sub):
            r = sub.iloc[0]
            lines.append(f"Best within feature set '{fs}': {r['method']} (AUC {r['auc']:.5f}, error {r['test_error']:.5f})")
    if w["test_error"] >= w["majority_class_error"] * 0.999:
        lines.append("NOTE: the winner's error is not below the always-majority-class error; rely on AUC.")
    (res / "winner.txt").write_text("\n".join(lines) + "\n\n" + ok.drop(columns=["selected_hp"]).to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": win, "criterion": winner_metric, "tied_with_winner": ties,
                                                  **{k: (None if pd.isna(v) else v) for k, v in w.to_dict().items() if k != "selected_hp"}}, indent=2, default=str))
    for l in lines:
        logger(l)
    return ok
