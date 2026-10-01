"""
Elimination & Ranking (PRD v10 §3.5), extracted so both the Orchestrator
(algorithm selection) and search.py (hyperparameter selection, §3.3.6:
"the same evaluation principles defined for model comparison") share one
implementation rather than two copies that could drift apart.

Steps, exactly as specified:
1. Eliminate against Baseline (MASE): MASE >= 1 means "not better than
   the naive/seasonal-naive benchmark" -- eliminated. MASE is undefined
   or an error also eliminates.
2. Eliminate Unacceptable Bias: |Bias %| > bias_threshold eliminated.
   bias_threshold is fully configurable (config.json), never hardcoded.
3. Rank Remaining Algorithms: lowest sMAPE/WAPE/MAE/RMSE/MASE gets the
   best (lowest) rank; bias closest to zero gets the best rank.
4. Combine Ranks: lowest sum of the six per-metric ranks wins.

Step 5 (Consistency Check) needs fold-level data, not just the averaged
metrics this module ranks on, so it lives in orchestrator.py, applied
to the winner this module returns before that winner is accepted.
"""

from __future__ import annotations
from dataclasses import dataclass, field

METRIC_KEYS = ["smape", "wape", "mae", "rmse", "mase", "bias_pct"]


@dataclass
class RankingResult:
    eliminated_mase: list[str] = field(default_factory=list)
    eliminated_bias: list[str] = field(default_factory=list)
    eliminated_error: list[str] = field(default_factory=list)
    ranked: list[tuple[str, float, dict]] = field(default_factory=list)  # (name, combined_rank, per_metric_rank)
    winner: str | None = None


def _rank_ascending(values: dict[str, float]) -> dict[str, float]:
    """Standard competition ranking (ties share the average rank), lowest value = rank 1."""
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    ranks: dict[str, float] = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        while j < n and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # average of positions i+1..j (1-indexed)
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def eliminate_and_rank(results: dict[str, dict], bias_threshold: float, eliminate_on_mase: bool = True) -> RankingResult:
    """
    `results`: name -> six-metric dict (as returned by
    AlgorithmModule.evaluate().metrics, averaged across backtest folds),
    or {"error": ...} for a candidate whose backtest produced no valid
    fold. Excludes the baseline itself -- callers pass only the
    candidates being compared against it.

    `eliminate_on_mase`: True for final algorithm selection (§3.5 step 1
    -- eliminate anything that doesn't beat the baseline). search.py
    passes False when using this function to rank one algorithm's OWN
    hyperparameter trials against each other (§3.3.6): those trials
    are being compared to find the best configuration for that
    algorithm, not screened against the baseline -- an algorithm whose
    every configuration happens to score MASE >= 1 on this segment
    should still get its least-bad configuration selected here, rather
    than falling back to un-searched defaults; §3.5's baseline
    elimination is applied once, at final selection, by the
    Orchestrator, not redundantly inside every trial comparison.
    """
    out = RankingResult()

    survivors: dict[str, dict] = {}
    for name, m in results.items():
        if "error" in m or m.get("mase") is None:
            out.eliminated_error.append(name)
            continue
        if eliminate_on_mase and m["mase"] >= 1.0:
            out.eliminated_mase.append(name)
            continue
        survivors[name] = m

    survivors2: dict[str, dict] = {}
    for name, m in survivors.items():
        bias_pct = m.get("bias_pct")
        if bias_pct is not None and abs(bias_pct) > bias_threshold:
            out.eliminated_bias.append(name)
            continue
        survivors2[name] = m

    if not survivors2:
        return out

    per_metric_ranks: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        if key == "bias_pct":
            values = {n: abs(m[key]) for n, m in survivors2.items() if m.get(key) is not None}
        else:
            values = {n: m[key] for n, m in survivors2.items() if m.get(key) is not None}
        per_metric_ranks[key] = _rank_ascending(values) if values else {}

    combined: dict[str, float] = {}
    per_name_ranks: dict[str, dict] = {}
    for name in survivors2:
        ranks = {key: per_metric_ranks[key].get(name) for key in METRIC_KEYS}
        per_name_ranks[name] = ranks
        combined[name] = sum(r for r in ranks.values() if r is not None)

    ordered_names = sorted(combined, key=lambda n: combined[n])
    out.ranked = [(n, combined[n], per_name_ranks[n]) for n in ordered_names]
    out.winner = ordered_names[0] if ordered_names else None
    return out
