"""
Production monitoring with human-reviewed rollback (PRD §4.4 last paragraph; G-01).

Rule (decision on G-01): on realised data, the CURRENT Production version and the PRIOR version
(the one a rollback would restore) each forecast the same periods from the same history; both are scored
with the six metrics and compared with the SAME rank-sum used for selection (ranking.eliminate_and_rank
with elimination off, so only the six per-metric ranks are combined). If the prior version's combined rank
is strictly lower (better), the monitor records a rollback PROPOSAL in the registry with the full evidence.
It never rolls back by itself: a named reviewer decides via registry.decide_rollback().

Ties keep the current Production version. The elimination rules of §3.5 (MASE >= 1, bias threshold) are
not part of this comparison; they are reported as information on the proposal for the reviewer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from algorithms.seasonal_naive import SeasonalNaiveModule
from algorithms.utils import endogenous_series
from ranking import eliminate_and_rank


@dataclass
class MonitorResult:
    segment_id: str
    action: str  # "ok" | "rollback_proposed" | "proposal_pending" | "proposal_rejected" | "no_prior_version" | "prior_unavailable" | "insufficient_data"
    detail: str = ""
    production: dict = field(default_factory=dict)  # six live metrics of the Production version
    prior: dict = field(default_factory=dict)  # six live metrics of the prior version
    combined_rank: dict = field(default_factory=dict)  # {"production": x, "prior": y}; lower is better
    production_version: int | None = None
    prior_version: int | None = None
    proposal_id: str | None = None
    info_flags: dict = field(default_factory=dict)


class ProductionMonitor:
    def __init__(self, registry, model_loader, min_actuals: int = 1, seasonal_period: int = 7,
                 bias_threshold: float | None = None):
        """`model_loader(entry) -> loaded AlgorithmModule` (supplied by the Orchestrator, which owns the candidates)."""
        self.registry = registry
        self.model_loader = model_loader
        self.min_actuals = min_actuals
        self.seasonal_period = seasonal_period
        self.bias_threshold = bias_threshold  # information only, never a rollback trigger

    def _live_metrics(self, entry, history_df, actuals_df, horizon) -> dict:
        try:
            mod = self.model_loader(entry)
            fc = mod.infer(history_df, horizon)
            ev = SeasonalNaiveModule({"season": self.seasonal_period}).evaluate(
                fc, actuals_df, insample_series=endogenous_series(history_df), seasonal_period=self.seasonal_period)
            return ev.metrics
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def check(self, segment_id: str, actuals_df: pd.DataFrame, history_df: pd.DataFrame) -> MonitorResult:
        """
        history_df: canonical rows up to the forecast origin (what both models see);
        actuals_df: [date, value] realised after the origin. Horizon = number of realised dates.
        """
        cur = self.registry.get_active(segment_id)
        if cur is None:
            raise RuntimeError(f"no Production version for segment {segment_id}")
        prior = self.registry.prior_entry(segment_id)
        horizon = int(actuals_df["date"].nunique())
        if horizon < self.min_actuals:
            return MonitorResult(segment_id, "insufficient_data", f"{horizon} realised periods < min_actuals {self.min_actuals}",
                                 production_version=cur.version)
        prod_m = self._live_metrics(cur, history_df, actuals_df, horizon)
        if "error" in prod_m or prod_m.get("mase") is None:
            return MonitorResult(segment_id, "insufficient_data",
                                 f"Production live metrics unavailable: {prod_m.get('error', 'undefined MASE scale')}",
                                 production=prod_m, production_version=cur.version)
        flags = {"production_mase_ge_1": prod_m["mase"] >= 1.0, "production_bias_pct": prod_m.get("bias_pct")}
        if self.bias_threshold is not None and prod_m.get("bias_pct") is not None:
            flags["production_bias_exceeds_threshold"] = abs(prod_m["bias_pct"]) > self.bias_threshold
        if prior is None:
            return MonitorResult(segment_id, "no_prior_version", "no earlier Production version exists to compare against or restore",
                                 production=prod_m, production_version=cur.version, info_flags=flags)
        prior_m = self._live_metrics(prior, history_df, actuals_df, horizon)
        if "error" in prior_m or prior_m.get("mase") is None:
            return MonitorResult(segment_id, "prior_unavailable",
                                 f"prior v{prior.version} could not be scored: {prior_m.get('error', 'undefined MASE scale')}",
                                 production=prod_m, prior=prior_m, production_version=cur.version,
                                 prior_version=prior.version, info_flags=flags)
        ranking = eliminate_and_rank({"production": prod_m, "prior": prior_m}, bias_threshold=float("inf"), eliminate_on_mase=False)
        combined = {n: r for n, r, _ in ranking.ranked}
        per_metric = {n: pm for n, _, pm in ranking.ranked}
        base = dict(production=prod_m, prior=prior_m, combined_rank=combined, production_version=cur.version,
                    prior_version=prior.version, info_flags=flags)
        if not (combined["prior"] < combined["production"]):
            return MonitorResult(segment_id, "ok", f"Production v{cur.version} rank-sum {combined['production']} "
                                 f"<= prior v{prior.version} {combined['prior']}", **base)
        reason = (f"prior v{prior.version} beats Production v{cur.version} on live data over {horizon} periods: "
                  f"six-metric rank-sum {combined['prior']} vs {combined['production']} (lower is better)")
        rejected = [p for p in self.registry.proposals(segment_id) if p["status"] == "rejected"
                    and p["production_version"] == cur.version and p["prior_version"] == prior.version]
        if rejected:
            d = rejected[-1]["decision"]
            return MonitorResult(segment_id, "proposal_rejected",
                                 reason + f" -- a proposal for this pair was rejected by {d['by']} on {d['at']}; not re-proposed", **base)
        already = [p for p in self.registry.pending_proposals(segment_id)
                   if p["production_version"] == cur.version and p["prior_version"] == prior.version]
        prop = self.registry.propose_rollback(segment_id, {
            "horizon": horizon, "production_metrics": prod_m, "prior_metrics": prior_m,
            "combined_rank": combined, "per_metric_rank": per_metric, "info_flags": flags,
            "production_algorithm": cur.algorithm_name, "prior_algorithm": prior.algorithm_name,
        }, reason)
        return MonitorResult(segment_id, "proposal_pending" if already else "rollback_proposed",
                             reason + " -- awaiting human review (registry.decide_rollback)", proposal_id=prop["id"], **base)
