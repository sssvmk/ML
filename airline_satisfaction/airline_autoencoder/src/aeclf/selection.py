"""Step 9: analyse the experiments and declare a winner, with a written decision trace.

Procedure (thresholds from config.selection):
  1. Candidates = every successful non-baseline model (the 'scratch' control is eligible too: if it wins, pretraining did not help).
  2. Best = highest VALIDATION primary metric. A paired, stratified bootstrap on the SAME validation rows gives a 95% CI for
     (best - other). 'other' ties with the best if the difference is below min_practical_delta OR its CI contains 0.
  3. Among the tie set choose: no large train-validation gap (soft flag) -> fewest trainable parameters -> lowest latency.
  4. Pretraining benefit: each AE-based candidate vs the scratch control (paired bootstrap), reported separately.
  5. The test set is NOT used here; it is evaluated once for the winner (final_evaluation.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve

from .metrics import fast_weighted_auc, primary_value
from .utils import dump_json, load_json


def load_candidates(run_dir: Path) -> list[dict]:
    return [load_json(p) for p in sorted((Path(run_dir) / "candidates").glob("*/result.json"))]


def _bootstrap(y, preds: dict, metric: str, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pos, neg, n = np.where(y == 1)[0], np.where(y == 0)[0], len(y)
    if metric == "roc_auc":
        prep = {k: (lambda u: (u[1], len(u[0])))(np.unique(p, return_inverse=True)) for k, p in preds.items()}
    else:
        B = min(B, 200)
    out = {k: np.empty(B) for k in preds}
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if metric == "roc_auc":
            w = np.bincount(idx, minlength=n).astype(float)
            for k in preds:
                out[k][b] = fast_weighted_auc(y, preds[k], prep[k][0], prep[k][1], w)
        else:
            for k in preds:
                out[k][b] = primary_value(metric, y[idx], preds[k][idx])
    return out


def _diff(boot, a, b, delta):
    d = boot[a] - boot[b]
    lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
    return float(np.nanmean(d)), lo, hi, bool(lo <= 0 <= hi), float(max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / len(d)))


def declare_winner(results: list[dict], cfg: dict, run_dir: Path) -> dict:
    sel, metric = cfg["selection"], cfg["metric"]["primary"]
    run_dir, trace = Path(run_dir), []
    ok = [r for r in results if r["status"] == "ok"]
    for r in results:
        if r["status"] != "ok":
            trace.append(f"EXCLUDED {r['name']}: {r['status']} - {r['message']}")
    cands = [r for r in ok if r["role"] != "baseline"]
    if not cands:
        raise RuntimeError("no successful candidate")
    y, preds = None, {}
    for r in ok:
        f = np.load(r["artifacts"]["val_predictions"])
        y = f["y"] if y is None else y
        preds[r["name"]] = f["p"]
    boot = _bootstrap(y, preds, metric, sel["bootstrap_samples"], cfg["project"]["seed"])
    table = []
    for r in ok:
        b = boot[r["name"]]
        table.append({"name": r["name"], "display_name": r["display_name"], "role": r["role"], "mode": r["mode"], "val_primary": r["val_primary"],
                      "ci95_low": float(np.nanpercentile(b, 2.5)), "ci95_high": float(np.nanpercentile(b, 97.5)), "train_primary": r["train_primary"],
                      "overfit_gap": r["overfit_gap"], "accuracy": r["val_metrics"]["accuracy"], "f1": r["val_metrics"]["f1"], "pr_auc": r["val_metrics"]["pr_auc"],
                      "log_loss": r["val_metrics"]["log_loss"], "brier": r["val_metrics"]["brier"], "ece": r["val_metrics"]["ece"],
                      "trainable_params": r["trainable_params"], "latency_ms_per_1000": r["latency_ms_per_1000_rows"], "flags": []})
    by = {t["name"]: t for t in table}
    base = next((r for r in ok if r["role"] == "baseline"), None)
    if base:
        trace.append(f"Baseline floor: {metric} = {by['majority_baseline']['val_primary']:.4f}.")
        for c in list(cands):
            if by[c["name"]]["val_primary"] - by["majority_baseline"]["val_primary"] <= sel["min_practical_delta"]:
                trace.append(f"EXCLUDED {c['name']}: does not beat the majority-class baseline.")
                cands.remove(c)
    if not cands:
        raise RuntimeError("no candidate beats the baseline")
    for c in cands:
        t = by[c["name"]]
        if t["overfit_gap"] > sel["max_overfit_gap"]:
            t["flags"].append(f"large train-validation gap {t['overfit_gap']:.4f} > {sel['max_overfit_gap']} (soft: loses ties)")
            trace.append(f"FLAG {c['name']}: {t['flags'][-1]}")
    best = max(cands, key=lambda r: r["val_primary"])
    trace.append(f"Best validation {metric}: {best['display_name']} = {best['val_primary']:.4f} (95% CI {by[best['name']]['ci95_low']:.4f}-{by[best['name']]['ci95_high']:.4f}).")
    pairwise, ties = [], [best]
    for c in cands:
        if c["name"] == best["name"]:
            continue
        diff, lo, hi, zero_in, p = _diff(boot, best["name"], c["name"], sel["min_practical_delta"])
        tie = (best["val_primary"] - c["val_primary"] <= sel["min_practical_delta"]) or zero_in
        pairwise.append({"vs": c["name"], "diff": best["val_primary"] - c["val_primary"], "ci95": [lo, hi], "bootstrap_p": p, "tie": bool(tie),
                         "mcnemar_p": _mcnemar(y, preds[best["name"]], preds[c["name"]], best, c)})
        trace.append(f"  {best['display_name']} - {c['display_name']}: diff {best['val_primary'] - c['val_primary']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] -> "
                     + ("TIE" if tie else "best is significantly better"))
        if tie:
            ties.append(c)
    keys = {"trainable_params": lambda r: by[r["name"]]["trainable_params"], "latency": lambda r: by[r["name"]]["latency_ms_per_1000"]}
    order = [lambda r: bool(by[r["name"]]["flags"])] + [keys[k] for k in sel["tie_breakers"]]
    winner = sorted(ties, key=lambda r: tuple(f(r) for f in order))[0]
    if len(ties) > 1:
        trace.append("Tie set: " + ", ".join(t["display_name"] for t in ties) + f". Tie-breakers {sel['tie_breakers']} -> {winner['display_name']}.")
    trace.append(f"Winner: {winner['display_name']}." if winner["name"] == best["name"] else
                 f"Winner differs from the raw best: {winner['display_name']} is statistically indistinguishable but smaller/faster.")
    if winner["role"] == "control":
        trace.append("NOTE: the no-pretraining control won or tied-and-won: in this experiment autoencoder pretraining did not help the classifier.")
    if cfg["metric"].get("target_value") is None:
        trace.append("No business target configured: the winner is the best AVAILABLE model, not proven good enough.")
    benefit = []
    scr = next((r for r in ok if r["mode"] == "scratch"), None)       # control for neural heads
    raw = next((r for r in ok if r["mode"] == "lgbm_raw"), None)      # control for LightGBM heads
    for r in ok:
        if r["role"] != "candidate":
            continue
        ctrl = raw if r["mode"].startswith("lgbm") else scr
        if ctrl:
            diff, lo, hi, zero_in, p = _diff(boot, r["name"], ctrl["name"], sel["min_practical_delta"])
            benefit.append({"candidate": r["name"], "control": ctrl["name"], "diff_vs_scratch": diff, "ci95": [lo, hi], "bootstrap_p": p,
                            "verdict": "encoder helps" if lo > 0 else ("encoder hurts" if hi < 0 else "no detectable difference")})
    out = {"primary_metric": metric, "winner": winner["name"], "raw_best": best["name"], "tie_set": [t["name"] for t in ties], "table": table,
           "pairwise_vs_best": pairwise, "pretraining_benefit": benefit, "decision_trace": trace, "validation_rows": int(len(y)),
           "bootstrap_samples": int(len(next(iter(boot.values())))), "rules": {k: sel[k] for k in ("min_practical_delta", "max_overfit_gap", "tie_breakers")}}
    dump_json(out, run_dir / "selection.json")
    _markdown(out, run_dir / "model_comparison.md")
    _plots(out, preds, y, run_dir)
    return out


def _mcnemar(y, pa, pb, ra, rb):
    ca = (pa >= ra["val_metrics"]["threshold"]).astype(int) == y
    cb = (pb >= rb["val_metrics"]["threshold"]).astype(int) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def _markdown(sel: dict, path: Path):
    m = sel["primary_metric"]
    L = [f"# Model comparison (validation, primary metric {m})", "", f"| Rank | Model | Val {m} | 95% CI | Train-val gap | Acc | F1 | Brier | Trainable params | ms/1k rows | Notes |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(sorted(sel["table"], key=lambda t: -t["val_primary"]), 1):
        note = ("**WINNER** " if t["name"] == sel["winner"] else "") + "; ".join(t["flags"]) + (" (control)" if t["role"] == "control" else "")
        L.append(f"| {i} | {t['display_name']} | {t['val_primary']:.4f} | {t['ci95_low']:.4f}-{t['ci95_high']:.4f} | {t['overfit_gap']:+.4f} | {t['accuracy']:.4f} | "
                 f"{t['f1']:.4f} | {t['brier']:.4f} | {t['trainable_params']:,} | {t['latency_ms_per_1000']:.1f} | {note.strip()} |")
    L += ["", "## Decision trace", ""] + [f"- {x}" for x in sel["decision_trace"]]
    if sel["pretraining_benefit"]:
        L += ["", "## Does the encoder help? (paired bootstrap vs the matching no-encoder control: scratch network for neural heads, raw-feature LightGBM for LightGBM heads)", "", "| Candidate | control | diff vs control | 95% CI | verdict |", "|---|---|---|---|---|"]
        for b in sel["pretraining_benefit"]:
            L.append(f"| {b['candidate']} | {b['control']} | {b['diff_vs_scratch']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['verdict']} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _plots(sel, preds, y, run_dir):
    d = Path(run_dir) / "reports"
    d.mkdir(exist_ok=True, parents=True)
    t = pd.DataFrame(sel["table"]).sort_values("val_primary")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(t) + 1.5))
    err = np.clip(np.array([t["val_primary"] - t["ci95_low"], t["ci95_high"] - t["val_primary"]]), 0, None)
    ax.barh(t["display_name"], t["val_primary"], xerr=err, color=["tab:green" if n == sel["winner"] else ("tab:gray" if r != "candidate" else "tab:blue") for n, r in zip(t["name"], t["role"])])
    if sel["primary_metric"] != "neg_log_loss":
        ax.set_xlim(max(0, float(t["ci95_low"].min()) - 0.05), min(1.0, float(t["ci95_high"].max()) + 0.01))
    ax.set(xlabel=f"validation {sel['primary_metric']} (95% bootstrap CI)", title="Model comparison (green = winner, grey = controls)")
    fig.tight_layout(); fig.savefig(d / "model_comparison.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=k)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC overlay (validation)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "roc_overlay.png", dpi=110); plt.close(fig)
