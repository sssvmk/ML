"""Step 12: analyse the experiments and declare a winner - with a written, auditable decision trail.

Decision procedure (all thresholds come from config.selection):
  1. Exclude algorithms that failed/were skipped. The baseline is a reference, never a candidate.
  2. Exclude candidates whose cross-validation std is both above max_cv_std and above 2x the candidates' median (unstable).
     A train-minus-validation gap above max_overfit_gap is a SOFT flag: such models lose ties to compliant ones.
     If the hard rule would exclude everyone, relax it and say so.
  3. Find the candidate with the best VALIDATION primary metric.
  4. A paired, stratified bootstrap on the SAME validation rows gives a CI for (best - other).
     'other' is a statistical tie if the CI contains 0 OR the difference is below min_practical_delta.
  5. Among the best and all its ties choose by tie-breakers (simplest -> fastest -> best calibrated).
     Rationale: with indistinguishable accuracy, the simpler/faster model generalises and operates more reliably.
  6. The TEST set is NOT used here. It is evaluated once, for the winner only (see final_evaluation.py).
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

from ..metrics import fast_weighted_auc, primary_value
from ..utils import dump_json, load_json


def load_results(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted((Path(run_dir) / "algorithms").glob("*/result.json")):
        out.append(load_json(p))
    return out


def _load_preds(res: dict):
    f = np.load(res["artifacts"]["val_predictions"])
    return f["y"], f["p"]


def _bootstrap_metric(y, p_dict: dict, metric: str, B: int, seed: int):
    """Return {name: array of B bootstrap values} using the SAME resamples for every model (paired)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = len(y)
    names = list(p_dict)
    out = {k: np.empty(B) for k in names}
    if metric == "roc_auc":
        prep = {}
        for k, p in p_dict.items():
            uniq, inv = np.unique(p, return_inverse=True)
            prep[k] = (inv, len(uniq))
    else:
        B = min(B, 200)  # generic path is slower
        out = {k: np.empty(B) for k in names}
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if metric == "roc_auc":
            w = np.bincount(idx, minlength=n).astype(float)
            for k in names:
                inv, g = prep[k]
                out[k][b] = fast_weighted_auc(y, p_dict[k], inv, g, w)
        else:
            for k in names:
                out[k][b] = primary_value(metric, y[idx], p_dict[k][idx])
    return out


def declare_winner(results: list[dict], cfg: dict, run_dir: Path) -> dict:
    sel, metric = cfg["selection"], cfg["metric"]["primary"]
    run_dir = Path(run_dir)
    trace: list[str] = []
    ok = [r for r in results if r["status"] == "ok"]
    for r in results:
        if r["status"] != "ok":
            trace.append(f"EXCLUDED {r['name']}: {r['status']} - {r['message']}")
    baseline = next((r for r in ok if r["family"] == "baseline"), None)
    cands = [r for r in ok if r["family"] != "baseline"]
    if not cands:
        raise RuntimeError("No successful candidate algorithm: nothing to select.")

    # bootstrap on validation predictions (all models share the same validation rows) -------------------------------
    y_ref = None
    preds = {}
    for r in ok:
        y, p = _load_preds(r)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise RuntimeError("Validation labels differ between algorithms: results are not comparable.")
        preds[r["name"]] = p
    boot = _bootstrap_metric(y_ref, preds, metric, sel["bootstrap_samples"], cfg["project"]["seed"])

    table = []
    for r in ok:
        b = boot[r["name"]]
        row = {"name": r["name"], "display_name": r["display_name"], "family": r["family"],
               "val_primary": r["val_primary"], "ci95_low": float(np.nanpercentile(b, 2.5)),
               "ci95_high": float(np.nanpercentile(b, 97.5)), "train_primary": r["train_primary"],
               "overfit_gap": r["overfit_gap"], "cv_mean": (r.get("cv") or {}).get("mean"),
               "cv_std": (r.get("cv") or {}).get("std"), "accuracy": r["val_metrics"]["accuracy"],
               "f1": r["val_metrics"]["f1"], "pr_auc": r["val_metrics"]["pr_auc"], "log_loss": r["val_metrics"]["log_loss"],
               "brier": r["val_metrics"]["brier"], "ece": r["val_metrics"]["ece"],
               "latency_ms_per_1000": r["latency_ms_per_1000_rows"], "fit_seconds": r["fit_seconds"],
               "complexity_rank": r["complexity_rank"], "diagnosis": r["diagnosis"].get("verdict"),
               "disqualified": [], "flags": [], "assumption_warnings": [a["name"] for a in r["assumptions"] if not a["passed"]]}
        table.append(row)
    by_name = {t["name"]: t for t in table}

    # 1. sanity vs baseline ------------------------------------------------------------------------------------------------
    if baseline:
        floor = by_name[baseline["name"]]["val_primary"]
        trace.append(f"Baseline floor ({baseline['display_name']}): {metric} = {floor:.4f}.")
        for c in cands:
            if by_name[c["name"]]["val_primary"] - floor <= sel["min_practical_delta"]:
                by_name[c["name"]]["disqualified"].append("does not beat the majority-class baseline")

    # 2. guard-rails ---------------------------------------------------------------------------------------------------------
    #    * instability is a hard exclusion (CV std above the absolute limit AND above 2x the median of the candidates, so that
    #      small-sample noise that affects everybody equally does not disqualify everybody)
    #    * a large train-validation gap is a SOFT rule: it is the first tie-breaker and is flagged. In-sample scores of bagged
    #      trees are inflated by construction, and validation scores are already held-out, so a gap alone is not grounds for exclusion.
    stds = [by_name[c["name"]]["cv_std"] for c in cands if by_name[c["name"]]["cv_std"] is not None]
    med_std = float(np.median(stds)) if stds else 0.0
    for c in cands:
        t = by_name[c["name"]]
        t["flags"] = []
        if t["cv_std"] is not None and t["cv_std"] > sel["max_cv_std"] and t["cv_std"] > 2 * med_std:
            t["disqualified"].append(f"unstable: CV std {t['cv_std']:.4f} > {sel['max_cv_std']} and > 2x median {med_std:.4f}")
        if t["overfit_gap"] is not None and t["overfit_gap"] > sel["max_overfit_gap"]:
            t["flags"].append(f"large train-validation gap {t['overfit_gap']:.4f} > {sel['max_overfit_gap']}")
    eligible = [c for c in cands if not by_name[c["name"]]["disqualified"]]
    relaxed = False
    if not eligible:
        eligible, relaxed = list(cands), True
        trace.append("WARNING: every candidate was excluded by a guard-rail; guard-rails relaxed so a winner can still be reported. Treat with caution.")
    for c in cands:
        for d in by_name[c["name"]]["disqualified"]:
            trace.append(f"DISQUALIFIED {c['name']}: {d}" + (" (ignored: relaxed)" if relaxed else ""))
        for f in by_name[c["name"]]["flags"]:
            trace.append(f"FLAG {c['name']}: {f} (soft rule: loses ties to compliant models)")

    # 3-4. best + paired bootstrap against it ------------------------------------------------------------------------------
    best = max(eligible, key=lambda r: r["val_primary"])
    trace.append(f"Best validation {metric}: {best['display_name']} = {best['val_primary']:.4f} "
                 f"(95% CI {by_name[best['name']]['ci95_low']:.4f}-{by_name[best['name']]['ci95_high']:.4f}).")
    pairwise = []
    ties = [best]
    for c in eligible:
        if c["name"] == best["name"]:
            continue
        d = boot[best["name"]] - boot[c["name"]]
        lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
        diff = best["val_primary"] - c["val_primary"]
        p_boot = float(2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        tie = (diff <= sel["min_practical_delta"]) or (lo <= 0 <= hi)
        pairwise.append({"vs": c["name"], "diff": diff, "ci95": [lo, hi], "bootstrap_p": max(p_boot, 1 / len(d)),
                         "tie": bool(tie), "mcnemar_p": _mcnemar(y_ref, preds[best["name"]], preds[c["name"]],
                                                                by_name[best["name"]], by_name[c["name"]], best, c)})
        trace.append(f"  {best['display_name']} - {c['display_name']}: diff {diff:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] -> "
                     + ("statistical/practical TIE" if tie else "best is significantly better"))
        if tie:
            ties.append(c)

    # 5. tie-break -----------------------------------------------------------------------------------------------------------
    keymap = {"complexity": lambda r: r["complexity_rank"], "latency": lambda r: by_name[r["name"]]["latency_ms_per_1000"],
              "brier": lambda r: by_name[r["name"]]["brier"]}
    order = [lambda r: bool(by_name[r["name"]]["flags"])] + [keymap[k] for k in sel["tie_breakers"]]   # compliant models first
    ties_sorted = sorted(ties, key=lambda r: tuple(f(r) for f in order))
    winner = ties_sorted[0]
    if len(ties) > 1:
        trace.append("Tie set: " + ", ".join(t["display_name"] for t in ties) + f". Tie-breakers {sel['tie_breakers']} -> {winner['display_name']}.")
    if winner["name"] != best["name"]:
        trace.append(f"Chosen winner differs from the raw best: {winner['display_name']} is simpler/faster and statistically indistinguishable.")
    else:
        trace.append(f"Winner: {winner['display_name']}.")
    target = cfg["metric"].get("target_value")
    if target is None:
        trace.append("No business target configured -> the winner is the best AVAILABLE model, not proven 'good enough' (see TODO).")
    elif winner["val_primary"] < target:
        trace.append(f"Winner is below the configured target {target}.")

    sel_out = {"primary_metric": metric, "winner": winner["name"], "raw_best": best["name"], "tie_set": [t["name"] for t in ties],
               "relaxed_guardrails": relaxed, "table": table, "pairwise_vs_best": pairwise, "decision_trace": trace,
               "rules": {k: sel[k] for k in ("min_practical_delta", "max_overfit_gap", "max_cv_std", "tie_breakers")},
               "validation_rows": int(len(y_ref)), "bootstrap_samples": int(len(next(iter(boot.values()))))}
    dump_json(sel_out, run_dir / "selection.json")
    _write_markdown(sel_out, run_dir / "model_comparison.md")
    _plots(sel_out, preds, y_ref, run_dir)
    return sel_out


def _mcnemar(y, p_a, p_b, ta, tb, ra, rb) -> float | None:
    thr_a, thr_b = ra["val_metrics"]["threshold"], rb["val_metrics"]["threshold"]
    ca, cb = ((p_a >= thr_a).astype(int) == y), ((p_b >= thr_b).astype(int) == y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return 1.0
    return float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def _write_markdown(sel: dict, path: Path):
    m = sel["primary_metric"]
    lines = [f"# Model comparison (validation set, primary metric: {m})", "",
             "| Rank | Algorithm | Family | Val " + m + " | 95% CI | Train-Val gap | CV mean±std | Acc | F1 | Brier | ms/1k rows | Notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(sel["table"], key=lambda t: -t["val_primary"])
    for i, t in enumerate(ranked, 1):
        cv = f"{t['cv_mean']:.4f}±{t['cv_std']:.4f}" if t["cv_mean"] is not None else "-"
        notes = "; ".join(t["disqualified"] + t.get("flags", [])) if (t["disqualified"] or t.get("flags")) else ""
        if t["name"] == sel["winner"]:
            notes = ("**WINNER** " + notes).strip()
        lines.append(f"| {i} | {t['display_name']} | {t['family']} | {t['val_primary']:.4f} | {t['ci95_low']:.4f}-{t['ci95_high']:.4f} | "
                     f"{t['overfit_gap']:+.4f} | {cv} | {t['accuracy']:.4f} | {t['f1']:.4f} | {t['brier']:.4f} | {t['latency_ms_per_1000']:.1f} | {notes} |")
    lines += ["", "## Decision trace", ""] + [f"- {t}" for t in sel["decision_trace"]]
    if sel["pairwise_vs_best"]:
        lines += ["", "## Paired bootstrap vs the best model", "", "| Competitor | diff | 95% CI of diff | bootstrap p | McNemar p | Tie? |", "|---|---|---|---|---|---|"]
        for p in sel["pairwise_vs_best"]:
            lines.append(f"| {p['vs']} | {p['diff']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['bootstrap_p']:.3g} | "
                         f"{'' if p['mcnemar_p'] is None else format(p['mcnemar_p'], '.3g')} | {'yes' if p['tie'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(sel: dict, preds: dict, y, run_dir: Path):
    d = Path(run_dir) / "reports"
    d.mkdir(exist_ok=True)
    t = pd.DataFrame(sel["table"]).sort_values("val_primary")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 1.5))
    err = np.array([t["val_primary"] - t["ci95_low"], t["ci95_high"] - t["val_primary"]])
    colors = ["tab:green" if n == sel["winner"] else "tab:blue" for n in t["name"]]
    ax.barh(t["display_name"], t["val_primary"], xerr=np.clip(err, 0, None), color=colors)
    lo = max(0.0, float(t["ci95_low"].min()) - 0.05) if sel["primary_metric"] != "neg_log_loss" else None
    if lo is not None:
        ax.set_xlim(lo, min(1.0, float(t["ci95_high"].max()) + 0.01))
    ax.set(xlabel=f"validation {sel['primary_metric']} (95% bootstrap CI)", title="Model comparison (green = winner)")
    fig.tight_layout(); fig.savefig(d / "model_comparison.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC overlay (validation)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "roc_overlay.png", dpi=110); plt.close(fig)
