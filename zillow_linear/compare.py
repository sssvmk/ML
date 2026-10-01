"""compare.py - cross-method comparison on the TEST set (MSE +- SE, paired differences) and winner announcement."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import Prepared


def compare(results_dir, prep_dir, method_names, logger=print):
    res = Path(results_dir)
    prep = Prepared(prep_dir)
    y = prep.y_test
    rows, preds = [], {}
    for m in method_names:
        sj = res / m / "summary.json"
        pj = res / m / "test_predictions.npy"
        if not sj.exists() or not pj.exists():
            rows.append({"method": m, "status": "FAILED/missing"})
            continue
        s = json.loads(sj.read_text())
        preds[m] = np.load(pj).astype(float)
        rows.append({"method": m, "status": "ok", "test_mse": s["test"]["mse"], "test_mse_se": s["test"]["mse_se"],
                     "test_rmse": s["test"]["rmse"], "test_mae": s["test"]["mae"], "test_r2": s["test"]["r2"],
                     "val_mse": s["validation_train_only_model"]["mse"], "cv_mse_at_choice": s["selection"]["cv_mse"],
                     "cv_se_at_choice": s["selection"]["cv_se"], "n_nonzero": s["selection"]["n_nonzero_final"],
                     "runtime_sec": s["runtime_sec"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"] == "ok"].sort_values("test_mse").reset_index(drop=True)
    if ok.empty:
        logger("no method finished successfully - nothing to compare")
        return None
    best = ok.loc[0, "method"]
    e2_best = (y - preds[best]) ** 2
    d_mean, d_se, signif = [], [], []
    for m in ok["method"]:
        d = (y - preds[m]) ** 2 - e2_best
        dm, ds = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        d_mean.append(dm); d_se.append(ds); signif.append(bool(m != best and dm / max(ds, 1e-300) > 2))
    ok["mse_minus_winner"], ok["paired_se"] = d_mean, d_se
    ok["significantly_worse_than_winner_(z>2)"] = signif
    base_mse = prep.baseline_test_mse
    ok["rel_mse_vs_train_mean_baseline"] = ok["test_mse"] / base_mse
    full = pd.concat([ok, tab[tab["status"] != "ok"]], ignore_index=True)
    full.to_csv(res / "comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(ok) + 1.8))
    ax.errorbar(ok["test_mse"][::-1], range(len(ok)), xerr=ok["test_mse_se"][::-1], fmt="o", color="tab:blue", capsize=3)
    ax.set_yticks(range(len(ok))); ax.set_yticklabels(ok["method"][::-1])
    ax.scatter([ok.loc[0, "test_mse"]], [len(ok) - 1], s=160, facecolors="none", edgecolors="gold", linewidths=2, label="winner", zorder=5)
    ax.axvline(base_mse, color="grey", ls="--", label="predict train mean")
    lo = min(ok["test_mse"].min(), base_mse) - 3 * ok["test_mse_se"].max()
    ax.set_xlim(max(lo, 0), max(ok["test_mse"].max(), base_mse) + 3 * ok["test_mse_se"].max())
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("test MSE of raw logerror (+- SE)"); ax.legend(); ax.set_title("Cross-method comparison (test set, Aug 2017 onward)")
    fig.tight_layout(); fig.savefig(res / "comparison_test_mse.png", dpi=130); plt.close(fig)

    ties = ok.loc[~ok["significantly_worse_than_winner_(z>2)"], "method"].tolist()
    w = ok.loc[0]
    msg = [f"WINNER: {best}  (test MSE {w['test_mse']:.6f} +- {w['test_mse_se']:.6f}, RMSE {w['test_rmse']:.5f}, "
           f"{w['rel_mse_vs_train_mean_baseline']:.4f}x the predict-the-mean baseline)",
           f"Not significantly different from the winner (paired z<=2): {', '.join(ties)}"]
    if w["rel_mse_vs_train_mean_baseline"] >= 1.0:
        msg.append("NOTE: the winner does not beat the predict-the-mean baseline on test - signal in logerror is weak for linear models.")
    (res / "winner.txt").write_text("\n".join(msg) + "\n\n" + full.to_string(index=False))
    (res / "winner.json").write_text(json.dumps({"winner": best, "tied_with_winner": ties, **w.to_dict()}, indent=2, default=str))
    for m_ in msg:
        logger(m_)
    return full
