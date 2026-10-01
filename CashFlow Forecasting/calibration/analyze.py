"""
Turns a calibration results file into PROPOSED Layer 2 parameters (significance level, seeds, series, margin rule) for approval (G-42).
Every number is computed from the data; the alternatives are shown so the approver sees what each choice would decide. Nothing here
approves anything: `config.json -> admission.calibration` stays null until a human fills it in.

    python -m calibration.analyze calibration/pilot_results.jsonl --out calibration/proposal
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import stats

ALPHA, POWER, MARGIN_SD = 0.05, 0.80, 1.0


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def n_for_noninferiority(sd_d: float, margin: float, alpha=ALPHA, power=POWER, true_diff: float = 0.0) -> int:
    """Seeds needed so that a one-sided paired t-test (H0: mean diff >= margin) has the stated power when the true mean difference is `true_diff`."""
    gap = margin - true_diff
    if gap <= 0 or not np.isfinite(sd_d):
        return -1
    n0 = max(2, math.ceil(((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * sd_d / gap) ** 2))
    for n in range(n0, 100_000):                                   # smallest n whose t-based requirement is met (t quantiles fall as n grows)
        need = ((stats.t.ppf(1 - alpha, n - 1) + stats.t.ppf(power, n - 1)) * sd_d / gap) ** 2
        if need <= n:
            return n
    return -1


def noninferiority(gap_ref: np.ndarray, gap_cus: np.ndarray, k: float = MARGIN_SD, alpha=ALPHA) -> dict:
    d = gap_cus - gap_ref
    n = len(d)
    sd_ref, sd_d = _sd(gap_ref), _sd(d)
    margin = k * sd_ref
    t = (d.mean() - margin) / (sd_d / math.sqrt(n)) if n > 1 and sd_d > 0 else float("nan")
    p = float(stats.t.cdf(t, n - 1)) if np.isfinite(t) else float("nan")
    upper = float(d.mean() + stats.t.ppf(1 - alpha, n - 1) * sd_d / math.sqrt(n)) if n > 1 else float("nan")
    # conservative sd of the paired difference: the upper 80% confidence bound (the pilot is small)
    sd_hi = sd_d * math.sqrt((n - 1) / stats.chi2.ppf(0.2, n - 1)) if n > 1 else float("nan")
    return {"n_seeds": n, "margin_multiple_of_reference_sd": k, "margin": margin, "reference_gap_mean": float(gap_ref.mean()), "reference_gap_sd": sd_ref,
            "custom_gap_mean": float(gap_cus.mean()), "custom_gap_sd": _sd(gap_cus), "paired_diff_mean": float(d.mean()), "paired_diff_sd": sd_d,
            "upper_confidence_bound_of_diff": upper, "p_value_noninferior": p, "noninferior_on_pilot": bool(p < alpha) if np.isfinite(p) else None,
            "seeds_needed_for_80pct_power_if_equal": n_for_noninferiority(sd_d, margin), "seeds_needed_conservative_sd": n_for_noninferiority(sd_hi, margin)}


def coverage(recs, who: str, level=0.8, alpha=ALPHA) -> dict:
    from calibration.metrics import kupiec_pvalue
    hits = np.array([r[who]["coverage80_hits"] for r in recs]); n = np.array([r[who]["coverage80_n"] for r in recs])
    per_seed = hits / n
    pooled_p = float(kupiec_pvalue(int(hits.sum()), int(n.sum()), level))
    binom_var = level * (1 - level) / n.mean()
    de = float(max(1.0, np.var(per_seed, ddof=1) / binom_var)) if len(per_seed) > 1 else float("nan")   # design effect: points within a seed are dependent
    eff_n = float(n.sum() / de) if np.isfinite(de) else float("nan")
    return {"pooled_coverage": float(hits.sum() / n.sum()), "points": int(n.sum()), "kupiec_p_pooled_as_if_independent": pooled_p,
            "per_seed_coverage_sd": _sd(per_seed), "design_effect": de, "effective_points": eff_n,
            "kupiec_p_with_effective_n": float(kupiec_pvalue(int(round(hits.sum() / de)), int(round(eff_n)), level)) if np.isfinite(de) else None}


def analyse(path: Path) -> dict:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    out = {"source": str(path), "records": len(recs), "assumptions": {"alpha": ALPHA, "power": POWER, "margin_rule": f"{MARGIN_SD} x the reference's own seed-to-seed SD of its gap to the oracle"}, "legs": {}}
    groups = {}
    for r in recs:
        groups.setdefault((r["leg"], r["setting"]), []).append(r)
    for (leg, setting), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        n = len(rs)
        metric = "crps" if leg == "deepstate" else "energy_score"
        orc = np.array([r["oracle"][metric] for r in rs]); ref = np.array([r["reference"][metric] for r in rs]); cus = np.array([r["custom"][metric] for r in rs])
        block = {"seeds": n, "cfg": rs[0]["cfg"], "score": metric, "mean_seconds_per_seed": {k: float(np.mean([r["seconds"][k] for r in rs])) for k in rs[0]["seconds"]},
                 "oracle_score_mean": float(orc.mean())}
        if n < 3:
            block["note"] = "fewer than 3 seeds: nothing can be estimated"
            out["legs"][f"{leg}/{setting}"] = block; continue
        block["accuracy"] = noninferiority(ref - orc, cus - orc)
        block["accuracy_alternative_margins"] = {f"{k}xSD": {kk: v for kk, v in noninferiority(ref - orc, cus - orc, k=k).items() if kk in ("margin", "p_value_noninferior", "noninferior_on_pilot", "seeds_needed_for_80pct_power_if_equal")} for k in (0.5, 1.0, 2.0)}
        block["coverage80"] = {"reference": coverage(rs, "reference"), "custom": coverage(rs, "custom")}
        if leg == "deepstate":
            tr = np.array([r["truth"]["step1_predictive_std_oracle"] for r in rs])
            lr = {w: np.log(np.array([r[w]["step1_predictive_std"] for r in rs]) / tr) for w in ("reference", "custom")}
            lo, hi = float(lr["reference"].mean() - stats.t.ppf(0.975, n - 1) * _sd(lr["reference"]) * math.sqrt(1 + 1 / n)), float(lr["reference"].mean() + stats.t.ppf(0.975, n - 1) * _sd(lr["reference"]) * math.sqrt(1 + 1 / n))
            block["parameter_recovery_step1_predictive_std_log_ratio"] = {
                "reference_mean": float(lr["reference"].mean()), "reference_sd": _sd(lr["reference"]), "custom_mean": float(lr["custom"].mean()), "custom_sd": _sd(lr["custom"]),
                "reference_95pct_prediction_interval": [lo, hi], "custom_mean_inside_reference_interval": bool(lo <= lr["custom"].mean() <= hi)}
        else:
            corr = rs[0]["corr"]
            c = {w: np.array([r[w]["step1_corr_dim01"] for r in rs]) for w in ("reference", "custom", "negative_control_diagonal")}
            block["parameter_recovery_step1_cross_correlation"] = {"true_innovation_corr": corr, **{f"{w}_mean": float(v.mean()) for w, v in c.items()}, **{f"{w}_sd": _sd(v) for w, v in c.items()}}
            dia = np.array([r["negative_control_diagonal"][metric] for r in rs])
            gain = (dia - orc) - (cus - orc)                       # positive = the full-covariance model beats the diagonal control
            sd_g = _sd(gain); t = gain.mean() / (sd_g / math.sqrt(n)) if sd_g > 0 else float("nan")
            block["negative_control_full_vs_diagonal"] = {"mean_gain_of_full_covariance": float(gain.mean()), "sd": sd_g, "p_full_better": float(1 - stats.t.cdf(t, n - 1)) if np.isfinite(t) else None,
                                                          "expected": "full covariance clearly better when corr > 0; no material difference when corr = 0",
                                                          "seeds_needed_for_80pct_power_at_observed_mean": n_for_noninferiority(sd_g, abs(gain.mean()) if gain.mean() != 0 else float("nan"), true_diff=0.0) if gain.mean() > 0 else -1}
        out["legs"][f"{leg}/{setting}"] = block
    return out


def render(res: dict) -> str:
    L = ["# Layer 2 calibration: pilot results and PROPOSED parameters (for approval)\n",
         f"Source: `{res['source']}` -- {res['records']} records. **Nothing here is approved**; `config.json -> admission.calibration` stays null until you fill it in.\n",
         f"Proposed conventions (need your approval): significance level alpha = {ALPHA}; target power = {POWER}; margin rule = {res['assumptions']['margin_rule']}.\n"]
    for name, b in res["legs"].items():
        L.append(f"\n## {name}  ({b['seeds']} seeds, score = {b['score']}, oracle mean {b['oracle_score_mean']:.3f})\n")
        L.append("Mean seconds per seed: " + ", ".join(f"{k} {v:.0f}s" for k, v in b["mean_seconds_per_seed"].items()) + "\n")
        if "accuracy" not in b:
            L.append(b.get("note", "")); continue
        a = b["accuracy"]
        L.append(f"- Reference gap to oracle: mean {a['reference_gap_mean']:.3f}, seed-to-seed SD {a['reference_gap_sd']:.3f}  ->  margin (1 SD) = {a['margin']:.3f}")
        L.append(f"- Custom gap to oracle: mean {a['custom_gap_mean']:.3f}, SD {a['custom_gap_sd']:.3f}; paired difference (custom - reference): mean {a['paired_diff_mean']:.3f}, SD {a['paired_diff_sd']:.3f}")
        L.append(f"- Non-inferiority on this pilot: p = {a['p_value_noninferior']:.3f} -> {'non-inferior' if a['noninferior_on_pilot'] else 'NOT shown non-inferior'} (upper bound of the difference {a['upper_confidence_bound_of_diff']:.3f} vs margin {a['margin']:.3f})")
        L.append(f"- Seeds needed for 80% power if the two are truly equal: {a['seeds_needed_for_80pct_power_if_equal']} (conservative, using the upper 80% bound of the SD: {a['seeds_needed_conservative_sd']})")
        L.append("- Alternative margins: " + "; ".join(f"{k}: margin {v['margin']:.3f}, p {v['p_value_noninferior']:.3f}, seeds {v['seeds_needed_for_80pct_power_if_equal']}" for k, v in b["accuracy_alternative_margins"].items()))
        for who, c in b["coverage80"].items():
            L.append(f"- 80% interval coverage, {who}: {c['pooled_coverage']:.3f} over {c['points']} points; Kupiec p (as if independent) {c['kupiec_p_pooled_as_if_independent']:.3f}; design effect {c['design_effect']:.1f} -> effective points {c['effective_points']:.0f}, p {c['kupiec_p_with_effective_n']:.3f}")
        if "parameter_recovery_step1_predictive_std_log_ratio" in b:
            p = b["parameter_recovery_step1_predictive_std_log_ratio"]
            L.append(f"- Parameter recovery (one-step predictive std / truth, log): reference {p['reference_mean']:+.3f} (SD {p['reference_sd']:.3f}), custom {p['custom_mean']:+.3f} (SD {p['custom_sd']:.3f}); custom inside the reference's 95% interval: {p['custom_mean_inside_reference_interval']}")
        if "parameter_recovery_step1_cross_correlation" in b:
            p = b["parameter_recovery_step1_cross_correlation"]
            L.append(f"- Cross-correlation recovery (true {p['true_innovation_corr']}): reference {p['reference_mean']:+.2f}, custom {p['custom_mean']:+.2f}, diagonal control {p['negative_control_diagonal_mean']:+.2f}")
            n = b["negative_control_full_vs_diagonal"]
            L.append(f"- Negative control (full vs diagonal covariance): mean gain {n['mean_gain_of_full_covariance']:.2f}, p(full better) {n['p_full_better']:.3f} -- expected: {n['expected']}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyse(Path(a.results))
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, indent=2, default=float)); Path(a.out + ".md").write_text(render(res))
    print(render(res))
