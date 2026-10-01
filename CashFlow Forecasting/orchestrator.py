"""
Orchestrator (Design doc §5.3, §6.1; PRD v10 §3.2-§3.7). Runs the three
cadences and is the only thing that calls an algorithm module's ten
interfaces -- it never touches Data Module internals (it only ever sees
canonical contract rows) and never touches an algorithm module's
internals (it only ever calls the interfaces in algorithms/base.py).

Per full_train/weekly_revalidate call, for each segment:
  1. Reserve a held-out tail (holdout_periods, config) untouched by
     everything below (§3.2).
  2. For each candidate: with hyperparameter search enabled, search the
     training window JOINTLY with the algorithm's hyperparameters (G-04;
     lower bound = the sampled configuration's own required_observations(),
     upper bound = min(available history, backtest_years_of_history));
     otherwise (or for an algorithm with no declared space) pick the
     window from candidate windows. Then rolling-backtest the chosen
     (window, hyperparameters) for the six-metric aggregate, per-fold
     metrics, train-vs-validation gap and diagnostics.
  3. Also backtest Seasonal-Naive-7 (the elimination baseline) and, as
     informational challengers only, Naive-1 and Seasonal-Naive-30 (§3.5).
  4. Eliminate & rank candidates (ranking.py, §3.5 steps 1-4).
  5. Consistency check (§3.5 step 5): the ranked winner must have
     MASE < 1 in at least `consistency_min_pass_rate` of its own folds;
     otherwise move to the next-ranked candidate.
  6. If nothing survives, fall back to rolling mean/median (§3.6).
  7. Refit the winner on all pre-holdout history at its chosen window,
     score on the held-out tail (reported, not used for selection), then
     refit again on the FULL history (holdout included) as the artifact
     that actually gets registered for production inference.
  8. Register it as a NEW VERSION and promote it to Production (previous
     Production is Archived) -- registry.py, mirrored to the MLflow Model
     Registry when enabled (§4.4, G-01).

Candidate backtests run in parallel via parallel.py (Ray if available,
else a thread pool, §3.7) -- not one after another. Charts are drawn
afterwards from the main thread (matplotlib is not thread-safe).
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import shutil
import pandas as pd

from algorithms.seasonal_naive import SeasonalNaiveModule
from algorithms.naive_1 import Naive1Module
from algorithms.rolling_fallback import RollingFallbackModule
from algorithms.utils import endogenous_series, future_known_datasets
from contract import validate_contract
from registry import ModelRegistry, RegistryEntry
from backtest import rolling_backtest, pooled_rolling_backtest
from pooling import slice_pool
from ranking import eliminate_and_rank
from parallel import parallel_run
from search import random_then_adaptive_search
from mlflow_logging import MLflowLogger
from monitor import ProductionMonitor
import charts as chart_lib


def _endog_dates(df: pd.DataFrame) -> list:
    """Sorted dates on which the TARGET is observed. Windows, holdouts and horizons are counted on these only: a
    future-known covariate extends the frame's date range beyond the last observation and must not shift them."""
    return sorted(df.loc[df["series_role"] == "endogenous", "date"].unique())


def _n_exog(df: pd.DataFrame) -> int:
    return df[df["series_role"] == "exogenous"]["dataset"].nunique()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


class Orchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        output_dir: str | Path,
        candidates: dict,
        max_parallel_workers: int = 8,
        seasonal_period: int = 7,
        bias_threshold: float = 20.0,
        consistency_min_pass_rate: float = 0.6,
        holdout_periods: int = 0,
        window_search_enabled: bool = True,
        backtest_years_of_history: int = 4,
        backtest_step: int | None = None,
        hyperparameter_search: dict | None = None,
        mlflow_config: dict | None = None,
        monitoring: dict | None = None,
        baselines: dict | None = None,
    ):
        bl = baselines or {}
        eg = bl.get("elimination_gate", {"season": 7})
        self.baseline_factory = lambda eg=dict(eg): SeasonalNaiveModule(dict(eg))
        chal = bl.get("challengers", {"naive_1": {"enabled": True}, "seasonal_naive_30": {"enabled": True, "season": 30}})
        self.challenger_factories = {}
        if (chal.get("naive_1") or {}).get("enabled", True):
            self.challenger_factories["naive_1_challenger"] = lambda: Naive1Module()
        sn30 = chal.get("seasonal_naive_30") or {}
        if sn30.get("enabled", True):
            self.challenger_factories["seasonal_naive_30_challenger"] = lambda season=sn30.get("season", 30): SeasonalNaiveModule({"season": season})
        fb = bl.get("fallback", {"combine": "rolling_mean_median", "lookback": 14})
        self._fallback_hp = dict(fb)
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.candidates = candidates  # name -> zero-arg factory returning AlgorithmModule
        self.max_parallel_workers = max_parallel_workers
        self.seasonal_period = seasonal_period
        self.bias_threshold = bias_threshold
        self.consistency_min_pass_rate = consistency_min_pass_rate
        self.holdout_periods = holdout_periods
        self.window_search_enabled = window_search_enabled
        self.max_window_days = backtest_years_of_history * 365
        self.backtest_step = backtest_step  # None -> rolling_backtest defaults to step=horizon (PRD §3.2: "Step size between windows remains configurable")
        hp_cfg = hyperparameter_search or {}
        self.hp_search_enabled = bool(hp_cfg.get("enabled", False))
        self.hp_search_budget = int(hp_cfg.get("budget", 16))
        self.hp_search_seed = hp_cfg.get("seed")
        self.hp_pruner_cfg = hp_cfg.get("pruner") or {}
        self.hp_prune_margin = float(hp_cfg.get("prune_margin", 2.0))
        self.mlflow = MLflowLogger({"mlflow": mlflow_config} if mlflow_config else None)
        if self.mlflow.active and getattr(self.registry, "backend", None) is None:
            self.registry.backend = self.mlflow  # mirror lifecycle transitions onto MLflow's Model Registry
        mon = monitoring or {}
        self.monitor = ProductionMonitor(
            self.registry, self._load_registered_model,
            min_actuals=int(mon.get("min_actuals", 1)), seasonal_period=seasonal_period, bias_threshold=bias_threshold,
        )

    def _validated(self, segment_df: pd.DataFrame) -> pd.DataFrame:
        result = validate_contract(segment_df)
        if not result.ok:
            raise ValueError(f"contract validation failed, segment/batch blocked: {result.errors}")
        return segment_df

    # ---- per-candidate: joint window+HPO search, final backtest -----------
    def _prepare_and_backtest(self, factory, segment_df: pd.DataFrame, horizon: int, ctx: dict) -> dict:
        probe = factory()
        cls = type(probe)
        n_exog = _n_exog(segment_df)
        req_obs = probe.required_observations(n_exog=n_exog)
        available = len(segment_df["date"].unique())
        max_window = min(available - horizon, self.max_window_days)

        if max_window < req_obs:
            return {
                "error": f"available history {available} (usable window <= {max_window}) "
                         f"below required observations {req_obs}",
                "aggregate": {"error": "ineligible: insufficient history"},
                "folds": [], "window": None, "hyperparameters": probe.hyperparameters,
            }

        fk = ctx.get("future_known")
        base_hp = dict(probe.hyperparameters)
        search_summary = None
        searched_window = None
        if self.hp_search_enabled:
            sr = random_then_adaptive_search(
                cls, base_hp, segment_df, window=max_window, horizon=horizon,
                budget=self.hp_search_budget, seasonal_period=self.seasonal_period,
                prune_margin=self.hp_prune_margin, available_observations=available,
                seed=self.hp_search_seed, window_bounds=(1, max_window), n_exog=n_exog,
                step=self.backtest_step, pruner_cfg=self.hp_pruner_cfg,
                tracker=self.mlflow, context=ctx, future_known=fk,
            )
            if sr.searched:
                base_hp = sr.best_hyperparameters
                searched_window = sr.best_window
                search_summary = {
                    "search_space_version": sr.search_space_version,
                    "joint_window_search": True,
                    "selected_window": sr.best_window,
                    "n_trials": len(sr.trials),
                    "n_completed": sum(1 for t in sr.trials if t.status == "completed"),
                    "n_pruned": sum(1 for t in sr.trials if t.status == "pruned"),
                    "n_failed": sum(1 for t in sr.trials if t.status == "failed"),
                    "trials": [
                        {"hyperparameters": t.hyperparameters, "window": t.window, "status": t.status, "reason": t.reason,
                         "aggregate_mase": t.aggregate_metrics.get("mase"), "duration_s": round(t.duration_s, 3),
                         "folds_run": t.n_folds_run, "selected": t.selected, "mlflow_run_id": t.mlflow_run_id}
                        for t in sr.trials
                    ],
                }

        if searched_window is not None:
            candidate_windows = [searched_window]  # window already optimised jointly with the hyperparameters
        elif self.window_search_enabled:
            candidate_windows = sorted(set(
                w for w in [req_obs, min(max_window, int(req_obs * 1.5)), max_window] if w >= req_obs
            ))
        else:
            candidate_windows = [max_window]

        best = None  # (window, BacktestResult)
        last_reasons: list[str] = []
        for w in candidate_windows:
            result = rolling_backtest(
                lambda hp=base_hp: cls(dict(hp)), segment_df, window=w, horizon=horizon,
                seasonal_period=self.seasonal_period, step=self.backtest_step, future_known=fk,
            )
            last_reasons = result.eligibility_reasons or last_reasons
            if result.error:
                continue
            current_mase = result.aggregate.get("mase")
            best_mase = best[1].aggregate.get("mase") if best else None
            if best is None or (current_mase is not None and (best_mase is None or current_mase < best_mase)):
                best = (w, result)

        if best is None:
            return {
                "error": "no candidate window produced a valid backtest fold",
                "aggregate": {"error": "no valid window"}, "folds": [], "window": None,
                "hyperparameters": base_hp, "search_summary": search_summary,
                "eligibility_reasons": last_reasons,
            }
        w, result = best
        return {
            "aggregate": result.aggregate, "folds": result.folds, "window": w,
            "hyperparameters": base_hp, "search_summary": search_summary,
            "fold_details": result.fold_details, "overfit": result.overfit,
            "eligibility_reasons": result.eligibility_reasons,
            "backtest_duration_s": round(result.duration_s, 3),
        }

    def _select_winner(self, segment_df: pd.DataFrame, horizon: int, ctx: dict,
                       pooled_details: dict | None = None) -> tuple[str, dict, dict, str]:
        """Returns (winner_name_or_'fallback', winner_detail, all_candidate_detail, backend).
        `pooled_details` (G-03): name -> this segment's result from the shared pooled backtest; it REPLACES the
        single-segment result of that pooling-capable algorithm, so the pooled model competes for this segment
        on identical terms (same folds' six metrics, same elimination, ranking and consistency rules)."""
        pooled_details = pooled_details or {}
        jobs = {name: (lambda factory=factory: self._prepare_and_backtest(factory, segment_df, horizon, ctx))
                for name, factory in self.candidates.items() if name not in pooled_details}
        all_detail, backend = parallel_run(jobs, max_workers=self.max_parallel_workers)
        all_detail.update(pooled_details)

        aggregate_for_ranking = {
            name: detail.get("aggregate", {"error": detail.get("error", "no result")})
            for name, detail in all_detail.items()
        }
        ranking = eliminate_and_rank(aggregate_for_ranking, bias_threshold=self.bias_threshold)

        winner_name = None
        consistency_rejected = []
        for name, combined_rank, per_metric in ranking.ranked:
            folds = all_detail[name]["folds"]
            ok_folds = [f for f in folds if "error" not in f and f.get("mase") is not None]
            if not ok_folds:
                consistency_rejected.append(name)
                continue
            pass_rate = sum(1 for f in ok_folds if f["mase"] < 1.0) / len(ok_folds)
            if pass_rate >= self.consistency_min_pass_rate:
                winner_name = name
                break
            consistency_rejected.append(name)

        elimination_log = {
            "eliminated_mase": ranking.eliminated_mase,
            "eliminated_bias": ranking.eliminated_bias,
            "eliminated_error": ranking.eliminated_error,
            "consistency_rejected": consistency_rejected,
            "ranked": [(n, r) for n, r, _ in ranking.ranked],
        }

        if winner_name is None:
            return "fallback", {}, {"all_candidates": all_detail, "elimination_log": {**elimination_log, "winner": None}}, backend
        return winner_name, all_detail[winner_name], {"all_candidates": all_detail, "elimination_log": {**elimination_log, "winner": winner_name}}, backend

    # ---- per-candidate logs and charts (G-07, G-08, G-10), main thread ----
    def _write_candidate_artifacts(self, segment_id: str, run_detail: dict, stamp: str) -> None:
        for name, detail in run_detail["all_candidates"].items():
            if detail.get("window") is None and not detail.get("folds"):
                # still log WHY it produced nothing (eligibility outcome) -- §5 item 8
                payload = {"algorithm": name, "error": detail.get("error"),
                           "eligibility_reasons": detail.get("eligibility_reasons", [])}
                out = self.output_dir / segment_id / "backtest" / name / "no_window"
                out.mkdir(parents=True, exist_ok=True)
                (out / "log.json").write_text(__import__("json").dumps(payload, indent=2, default=str))
                continue
            folds = detail.get("folds", [])
            fd = detail.get("fold_details", [])
            window_tag = f"w{detail.get('window')}"
            out_dir = self.output_dir / segment_id / "backtest" / name / window_tag
            charts: dict = {}
            p = chart_lib.metric_by_fold_chart(folds, out_dir / "charts", f"{name}_{window_tag}_metric_by_fold",
                                               f"{name} (window {detail.get('window')}): metric by fold")
            if p:
                charts["metric_by_fold"] = str(p)
            last_ok = next((d for d in reversed(fd) if "forecast" in d and "actual" in d), None)
            if last_ok:
                dts = [d for d, _ in last_ok["actual"]]
                act = [v for _, v in last_ok["actual"]]
                fc = [dict(last_ok["forecast"]).get(d) for d in dts]
                charts["forecast_vs_actual"] = str(chart_lib.forecast_vs_actual_chart(
                    dts, act, fc, out_dir / "charts", f"{name}_{window_tag}_forecast_vs_actual",
                    f"{name}: last backtest fold, forecast vs actual"))
                charts["residuals"] = str(chart_lib.residual_chart(
                    dts, [f - a for f, a in zip(fc, act)], out_dir / "charts", f"{name}_{window_tag}_residuals",
                    f"{name}: last backtest fold residuals"))
            detail["charts"] = charts
            diag = next((d["diagnostics"] for d in reversed(fd) if "diagnostics" in d), None)
            payload = {
                "algorithm": name, "window": detail.get("window"), "hyperparameters": detail.get("hyperparameters"),
                "aggregate": detail.get("aggregate"), "folds": folds, "overfit_train_vs_validation": detail.get("overfit"),
                "diagnostics_last_fold": diag, "eligibility_reasons": detail.get("eligibility_reasons", []),
                "early_stopping": next((d["early_stopping"] for d in reversed(fd) if "early_stopping" in d), None),
                "search": detail.get("search_summary"), "charts": charts,
            }
            probe = self.candidates[name]()
            probe.log(self.output_dir, segment_id, "backtest", f"{window_tag}_{stamp}", payload)

        # Cross-model comparison: one file per segment, all candidates side by side
        self._write_cross_model_comparison(segment_id, run_detail, stamp)

    def _write_cross_model_comparison(self, segment_id: str, run_detail: dict, stamp: str) -> None:
        """One file per segment run showing every algorithm's aggregate metrics, hyperparameters,
        window, eligibility outcome and rank — the single place a reviewer can compare all models."""
        import json
        rows = []
        elim = run_detail.get("elimination_log", {})
        raw_ranked = elim.get("ranked") or []
        # ranked can be [(name, score), ...] (normal path) or [name, ...] (forced path)
        ranked_names = [r[0] if isinstance(r, (list, tuple)) else r for r in raw_ranked]
        elim_mase  = set(elim.get("eliminated_mase")  or [])
        elim_bias  = set(elim.get("eliminated_bias")   or [])
        elim_err   = set(elim.get("eliminated_error")  or [])
        cons_rej   = set(elim.get("consistency_rejected") or [])
        for name, detail in sorted(run_detail["all_candidates"].items()):
            agg = detail.get("aggregate") or {}
            outcome = ("winner" if name == elim.get("winner") else
                       "ranked"              if name in ranked_names else
                       "consistency_rejected" if name in cons_rej else
                       "eliminated_bias"     if name in elim_bias else
                       "eliminated_mase"     if name in elim_mase else
                       "eliminated_error"    if name in elim_err else
                       "error"               if detail.get("error") else "unknown")
            rank = ranked_names.index(name) + 1 if name in ranked_names else None
            rows.append({"algorithm": name, "outcome": outcome, "rank": rank,
                         "window": detail.get("window"), "hyperparameters": detail.get("hyperparameters"),
                         "mase": agg.get("mase"), "smape": agg.get("smape"), "wape": agg.get("wape"),
                         "rmse": agg.get("rmse"), "mae": agg.get("mae"), "bias_pct": agg.get("bias_pct"),
                         "n_folds": len(detail.get("folds") or []),
                         "error": detail.get("error"), "overfit": detail.get("overfit")})
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 9999))
        out_dir = self.output_dir / segment_id
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"segment_id": segment_id, "stamp": stamp,
                   "elimination_log": elim, "models": rows}
        (out_dir / "cross_model_comparison.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- holdout / final fit ----------------------------------------------
    def _split_holdout(self, segment_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Reserves the last `holdout_periods` dates, untouched by selection/HPO (§3.2)."""
        if not self.holdout_periods:
            return segment_df, None
        dates = _endog_dates(segment_df)
        if len(dates) <= self.holdout_periods + 30:  # not enough history to spare a holdout meaningfully
            return segment_df, None
        cutoff = dates[-self.holdout_periods]
        selection_df = segment_df[segment_df["date"] < cutoff]
        return selection_df, segment_df  # full segment_df carries the holdout actuals too

    def _forced_algorithm(self) -> str | None:
        """The single algorithm id with force_enabled=true in config.json, or None. G-42 override: the orchestrator picks and
        runs this model directly -- it is trained (or, for a purely pretrained model, just resolved/loaded) and installed as
        the Production winner unconditionally, bypassing candidate competition/ranking and, if applicable, G-42 admission
        (config.py already includes it in self.candidates despite a failing or missing admission report)."""
        names = [n for n, f in self.candidates.items() if getattr(f, "force_enabled", False)]
        if len(names) > 1:
            raise ValueError(f"more than one algorithm has force_enabled=true in config.json: {names}; only one forced winner is supported at a time")
        return names[0] if names else None

    def _forced_elimination_log(self, name: str) -> dict:
        cls = type(self.candidates[name]())
        reason = f"config algorithms.{name}.force_enabled=true: competition and ranking bypassed, this algorithm is always the winner"
        if getattr(cls, "requires_admission", False):
            reason += "; G-42 admission bypassed too"
        return {"eliminated_mase": [], "eliminated_bias": [], "ranked": [name], "forced_override": True, "reason": reason}

    def _select_and_fit(self, segment_id: str, segment_df: pd.DataFrame, horizon: int, rule_version: str,
                        pooled: dict | None = None) -> dict:
        """Steps 1-7. Saves the fitted artifact under the NEXT registry version's directory but does
        NOT register it -- the caller decides (full_train always does; weekly_revalidate only on change)."""
        segment_df = self._validated(segment_df)
        fk = future_known_datasets(segment_df)     # inferred from structure on the FULL segment, before the holdout slice hides it
        selection_df, full_df = self._split_holdout(segment_df)
        sel_window = min(len(_endog_dates(selection_df)) - horizon, self.max_window_days)
        stamp = _stamp()
        ctx = {"segment_id": segment_id, "rule_version": rule_version, "future_known": fk}

        baseline_result = rolling_backtest(self.baseline_factory, selection_df, window=sel_window, horizon=horizon, seasonal_period=self.seasonal_period, step=self.backtest_step, collect_details=False)
        challenger_results = {
            name: rolling_backtest(factory, selection_df, window=sel_window, horizon=horizon, seasonal_period=self.seasonal_period, step=self.backtest_step, collect_details=False).aggregate
            for name, factory in self.challenger_factories.items()
        }

        pooled_map = pooled or {}
        pooled_details = {n: p["per_segment"][segment_id] for n, p in pooled_map.items() if segment_id in p["per_segment"]}
        forced = self._forced_algorithm()
        if forced:
            cls = type(self.candidates[forced]())
            if getattr(cls, "pooling_capable", False):
                if pooled is None or forced not in pooled:
                    raise ValueError(f"'{forced}' has force_enabled=true and is pooling-capable: run it through "
                                     f"full_train_pooled / weekly_revalidate_pooled (>= 2 segments), not full_train")
                detail = pooled[forced]["per_segment"].get(segment_id)
                if not pooled[forced].get("artifact_path") or detail is None or detail.get("error"):
                    raise RuntimeError(f"'{forced}' has force_enabled=true but failed to produce a usable pooled forecast for "
                                       f"segment {segment_id}: {(detail or {}).get('error') or 'no pooled artifact was produced'}")
                winner_name, winner_detail = forced, detail
            else:
                fac = self.candidates[forced]
                bt = rolling_backtest(fac, selection_df, window=sel_window, horizon=horizon, seasonal_period=self.seasonal_period,
                                      step=self.backtest_step, collect_details=False)
                winner_name = forced
                winner_detail = {"hyperparameters": fac().hyperparameters, "window": sel_window, "aggregate": bt.aggregate}
            run_detail = {"all_candidates": {forced: winner_detail}, "elimination_log": self._forced_elimination_log(forced)}
            backend = "forced"
        else:
            winner_name, winner_detail, run_detail, backend = self._select_winner(selection_df, horizon, ctx, pooled_details)
        run_detail["baseline"] = baseline_result.aggregate
        run_detail["challengers"] = challenger_results
        run_detail["parallel_backend"] = backend
        # ensure the winner name is in elimination_log so _write_cross_model_comparison can mark it
        run_detail["elimination_log"].setdefault("winner", winner_name if winner_name != "fallback" else None)
        self._write_candidate_artifacts(segment_id, run_detail, stamp)

        version = self.registry.next_version(segment_id)
        holdout_metrics = None
        diagnostics = None
        eval_charts: dict = {}
        shared = None
        if winner_name == "fallback":
            mod = RollingFallbackModule(dict(self._fallback_hp))
            mod.train(selection_df)
            path = self.registry.artifact_dir(segment_id, "fallback", version) / "model.json"
            entry_window = None
            entry_hp = mod.hyperparameters
        elif winner_name in pooled_map:
            # G-03: the winner is the POOLED model -- one artifact shared by every segment it serves. It was
            # already fitted (history incl. holdout) and saved once by _fit_pooled_candidates; nothing to refit.
            shared = pooled_map[winner_name]
            cls = type(self.candidates[winner_name]())
            entry_hp, entry_window = winner_detail["hyperparameters"], winner_detail["window"]
            mod = cls(dict(entry_hp))
            path = Path(shared["artifact_path"])
            mod.load(path)
            holdout_metrics = shared["holdout"].get(segment_id)
        else:
            cls = type(self.candidates[winner_name]())
            hp = winner_detail["hyperparameters"]
            window = winner_detail["window"]
            entry_window = window
            entry_hp = hp
            mod = cls(dict(hp))
            path = self.registry.artifact_dir(segment_id, winner_name, version) / "model.pkl"

            if full_df is not None:
                # Score on the untouched holdout tail before touching it for training (§3.2).
                dates = _endog_dates(selection_df)
                train_window_dates = set(dates[-window:]) if window else set(dates)
                holdout_train_df = selection_df[
                    selection_df["date"].isin(train_window_dates) | (selection_df["series_role"] == "exogenous")
                ]
                holdout_dates = sorted(set(_endog_dates(full_df)) - set(dates))
                if fk:   # the covariates a real forecast origin would already know for the held-out dates
                    fk_rows = full_df[(full_df["series_role"] == "exogenous") & full_df["dataset"].isin(fk)
                                      & (full_df["date"] > dates[-1]) & (full_df["date"] <= holdout_dates[-1])]
                    holdout_train_df = pd.concat([holdout_train_df, fk_rows])
                holdout_actuals = full_df[
                    (full_df["date"].isin(holdout_dates)) & (full_df["series_role"] == "endogenous")
                ].groupby("date")["value"].sum().reset_index()
                try:
                    probe_mod = cls(dict(hp))
                    probe_mod.train(holdout_train_df)
                    fc = probe_mod.infer(holdout_train_df, len(holdout_dates))
                    holdout_eval = probe_mod.evaluate(
                        fc, holdout_actuals, insample_series=endogenous_series(holdout_train_df),
                        seasonal_period=self.seasonal_period,
                        chart_dir=self.output_dir / segment_id / "full_training" / winner_name / "charts", chart_tag=f"holdout_{stamp}",
                    )
                    holdout_metrics = holdout_eval.metrics
                    eval_charts = holdout_eval.charts
                except Exception as exc:
                    holdout_metrics = {"error": f"holdout evaluation failed: {exc}"}

            # Final artifact is refit on ALL history (holdout included), sized to the same
            # chosen window, before being registered for production inference.
            final_df = full_df if full_df is not None else selection_df
            final_dates = _endog_dates(final_df)
            final_window_dates = set(final_dates[-window:]) if window else set(final_dates)
            final_train_df = final_df[
                final_df["date"].isin(final_window_dates) | (final_df["series_role"] == "exogenous")
            ]
            mod.train(final_train_df)
            diag = mod.diagnose()
            diagnostics = diag.details if diag.ran else {"ran": False, **diag.details}

        if shared is None:
            mod.save(path)
        entry = RegistryEntry(
            segment_id=segment_id, algorithm_name=winner_name, is_fallback=(winner_name == "fallback"),
            hyperparameters=entry_hp, artifact_path=str(path), rule_version=rule_version,
            metrics=(winner_detail.get("aggregate", {}) if winner_name != "fallback" else {}),
            fallback_combine="rolling_mean_median" if winner_name == "fallback" else None,
            window=entry_window, holdout_metrics=holdout_metrics,
            elimination_log=run_detail["elimination_log"],
            pooled_group=shared["group_id"] if shared else None,
            pooled_segments=list(shared["segments"]) if shared else None,
        )
        return {"entry": entry, "module": mod, "version": version, "run_detail": run_detail, "stamp": stamp,
                "diagnostics": diagnostics, "eval_charts": eval_charts, "winner_detail": winner_detail}

    def _register(self, segment_id: str, fit: dict, rule_version: str) -> RegistryEntry:
        """Log the winning run (tags/params/metrics/charts/log JSON), register the version, promote it."""
        entry: RegistryEntry = fit["entry"]
        mod = fit["module"]
        run_detail = fit["run_detail"]
        winner_detail = fit["winner_detail"]
        numeric_metrics = {k: v for k, v in (entry.metrics or {}).items() if isinstance(v, (int, float))}
        numeric_metrics.update({f"holdout_{k}": v for k, v in (entry.holdout_metrics or {}).items() if isinstance(v, (int, float))})
        numeric_metrics.update({f"overfit_{k}": v for k, v in (winner_detail.get("overfit") or {}).items() if isinstance(v, (int, float))})
        diag = fit.get("diagnostics") or {}
        for lag, vals in (diag.get("ljung_box") or {}).items():
            numeric_metrics[f"ljung_box_p_lag{lag}"] = vals["p_value"]
        log_payload = {"winner": entry.algorithm_name, "window": entry.window, "holdout_metrics": entry.holdout_metrics,
                       "diagnostics": diag, "registry_version": fit["version"], **{k: v for k, v in run_detail.items() if k != "all_candidates"},
                       "candidates_summary": {n: {"window": d.get("window"), "aggregate": d.get("aggregate"), "error": d.get("error")}
                                              for n, d in run_detail["all_candidates"].items()}}
        log_file = mod.log(self.output_dir, entry.segment_id, "full_training", fit["stamp"], log_payload)
        with self.mlflow.run(
            segment_id=segment_id, process="full_training", algorithm=entry.algorithm_name, rule_version=rule_version,
            hyperparameters=entry.hyperparameters, metrics=numeric_metrics,
            extra_tags={"registry_version": fit["version"], "window": entry.window},
        ) as run:
            run_id = run.info.run_id if run is not None else None
            # G-07: attach charts and the JSON log to the MLflow run
            for p in list((winner_detail.get("charts") or {}).values()) + list((fit.get("eval_charts") or {}).values()):
                self.mlflow.log_artifact_in_run(p, "charts")
            self.mlflow.log_artifact_in_run(log_file, "logs")
            self.registry.record_winner(entry, run_id)
        return entry

    def full_train(self, segment_id: str, segment_df: pd.DataFrame, horizon: int = 4, rule_version: str = "unknown") -> RegistryEntry:
        fit = self._select_and_fit(segment_id, segment_df, horizon, rule_version)
        return self._register(segment_id, fit, rule_version)

    def weekly_revalidate(self, segment_id: str, segment_df: pd.DataFrame, horizon: int = 4, rule_version: str = "unknown") -> RegistryEntry:
        """Re-runs the backtest/ranking; registers a new Production version only if a new winner is
        found (PRD §4.3). A rule_version change vs. the currently-registered entry always counts as a
        new winner (§4.2), and so does a segment flagged after an approved rollback (§4.4), even if the
        same algorithm is re-selected."""
        current = self.registry.get_active(segment_id)
        flagged = self.registry.needs_revalidation(segment_id)
        fit = self._select_and_fit(segment_id, segment_df, horizon, rule_version)
        return self._weekly_decision(segment_id, fit, rule_version, current, flagged)

    def _weekly_decision(self, segment_id: str, fit: dict, rule_version: str, current, flagged: bool) -> RegistryEntry:
        new_entry: RegistryEntry = fit["entry"]
        same_algorithm = (
            current is not None
            and current.algorithm_name == new_entry.algorithm_name
            and current.is_fallback == new_entry.is_fallback
            and current.rule_version == new_entry.rule_version
        )
        if same_algorithm and not flagged:
            if not new_entry.pooled_group:  # a pooled artifact may serve other segments: cleaned up per run instead
                shutil.rmtree(Path(new_entry.artifact_path).parent, ignore_errors=True)  # discard the unregistered refit
            return current
        return self._register(segment_id, fit, rule_version)

    # ---- pooled training across segments (G-03) -----------------------------------------------
    def _fit_pooled_candidates(self, segments: dict, horizon: int, group_id: str) -> dict:
        """
        For every pooling-capable candidate: rolling-backtest ONE model over all segments at once, score the
        untouched holdout per segment, then refit on the full history of all segments and save ONE shared
        artifact. Returns name -> {per_segment: {segment_id: detail}, holdout, artifact_path, group_id, segments}.
        Hyperparameters are the configured ones (joint window/HPO search is not run for pooled candidates).
        """
        splits = {sid: self._split_holdout(df) for sid, df in segments.items()}
        selection = {sid: sel for sid, (sel, _full) in splits.items()}
        max_window = min(min(len(sel["date"].unique()) for sel in selection.values()) - horizon, self.max_window_days)
        sel_dates = sorted(set().union(*(set(_endog_dates(sel)) for sel in selection.values())))
        fks = {sid: future_known_datasets(df) for sid, df in segments.items()}   # structural, from the full segments
        out: dict = {}
        for name, factory in self.candidates.items():
            probe = factory()
            if not probe.pooling_capable:
                continue
            hp = dict(probe.hyperparameters)
            # Window: ONE window per pooled model (one model instance serves all series). Lower bound = the
            # algorithm's own required-observations floor expressed per series: pooled N / n_series for
            # "pooled_total" rules, N itself for "per_series" rules (§3.2, generalised to a pool).
            req = int(probe.required_observations())
            lo = -(-req // len(selection)) if probe.pooled_n_basis == "pooled_total" else req
            if self.window_search_enabled:
                cand_windows = sorted({w for w in (lo, int(lo * 1.5), max_window) if lo <= w <= max_window})
            else:
                cand_windows = [max_window] if lo <= max_window else []
            best = None  # (mean MASE across segments, window, results, reasons)
            reasons: list[str] = []
            for w in cand_windows:
                res, rs = pooled_rolling_backtest(factory, selection, w, horizon,
                                                  seasonal_period=self.seasonal_period, step=self.backtest_step,
                                                  future_known=fks)
                reasons = rs or reasons
                mases = [r.aggregate.get("mase") for r in res.values() if not r.error and r.aggregate.get("mase") is not None]
                if not mases:
                    continue
                score = sum(mases) / len(mases)
                if best is None or score < best[0]:
                    best = (score, w, res, rs)
            if best is None:
                err = (f"available history ({max_window} usable periods per series) below the pooled minimum window {lo}"
                       if not cand_windows else "no candidate window produced a valid pooled backtest fold"
                       + (f"; ineligible: {reasons[0]}" if reasons else ""))
                from backtest import BacktestResult
                results, window, reasons = {sid: BacktestResult(error=err) for sid in selection}, None, reasons
            else:
                _, window, results, reasons = best
            per_segment = {}
            for sid, r in results.items():
                per_segment[sid] = {
                    "aggregate": r.aggregate if not r.error else {"error": r.error}, "folds": r.folds,
                    "window": window if not r.error else None, "hyperparameters": hp, "error": r.error,
                    "eligibility_reasons": reasons, "pooled_group": group_id,
                    "backtest_duration_s": round(r.duration_s, 3),
                }
            info = {"per_segment": per_segment, "holdout": {}, "artifact_path": None, "group_id": group_id, "segments": []}
            out[name] = info
            if all(r.error for r in results.values()):
                continue
            try:
                # 1) holdout scoring (§3.2): fit on the selection history only, forecast the reserved tail per segment
                if any(full is not sel for sel, full in splits.values()):
                    w_dates = set(sel_dates[-window:])
                    tb0 = slice_pool(selection, w_dates, sel_dates[-1])
                    aug = {}
                    for sid, seg in tb0.segments.items():   # add each series' future-known values for its held-out dates
                        full = splits[sid][1]
                        last = seg.loc[seg["series_role"] == "endogenous", "date"].max()
                        extra = full[(full["series_role"] == "exogenous") & full["dataset"].isin(fks[sid]) & (full["date"] > last)] \
                            if full is not splits[sid][0] else full.iloc[0:0]
                        aug[sid] = pd.concat([seg, extra]) if len(extra) else seg
                    from pooling import assemble_pool
                    tb = assemble_pool(aug)
                    m = factory()
                    if m.check_pooled_eligibility(tb).eligible:
                        m.train_pooled(tb)
                        n_hold = max((len(set(f["date"].unique()) - set(s_["date"].unique())) for s_, f in splits.values() if f is not s_), default=0)
                        joint_fc = m.infer_pool(tb.segments, n_hold) if getattr(m, "joint_inference", False) else None
                        for sid, (sel, full) in splits.items():
                            if full is sel or sid not in tb.segments:
                                continue
                            h_dates = sorted(set(full["date"].unique()) - set(sel["date"].unique()))
                            act = full[(full["date"].isin(h_dates)) & (full["series_role"] == "endogenous")]
                            act = act.groupby("date")["value"].sum().reset_index()
                            fc = joint_fc[sid].head(len(h_dates)) if joint_fc is not None else m.infer(tb.segments[sid], len(h_dates))
                            info["holdout"][sid] = m.evaluate(fc, act, insample_series=endogenous_series(tb.segments[sid]),
                                                              seasonal_period=self.seasonal_period).metrics
                # 2) final refit on ALL history (holdout included), one shared artifact
                finals = {sid: (full if full is not None else sel) for sid, (sel, full) in splits.items()}
                f_dates = sorted(set().union(*(set(_endog_dates(d)) for d in finals.values())))
                all_max = max(d["date"].max() for d in finals.values())
                fb = slice_pool(finals, set(f_dates[-window:]), f_dates[-1], future_known=fks, through=all_max)
                m = factory()
                elig = m.check_pooled_eligibility(fb)
                if not elig.eligible:
                    raise RuntimeError(f"final pool ineligible: {elig.reason}")
                m.train_pooled(fb)
                path = self.registry.pooled_artifact_dir(group_id, name) / "model.pkl"
                m.save(path)
                info["artifact_path"], info["segments"] = str(path), list(fb.segment_ids)
            except Exception as exc:
                for d in per_segment.values():  # cannot serve a production artifact -> cannot win
                    d["aggregate"], d["error"], d["window"] = {"error": f"pooled fit failed: {exc}"}, f"pooled fit failed: {exc}", None
        return out  # candidates whose pooled fit failed stay in: their per-segment details rank as errors

    def _run_pooled(self, segments: dict, horizon: int, rule_version: str, weekly: bool) -> dict:
        segments = {sid: self._validated(df) for sid, df in segments.items()}
        if len(segments) < 2:
            raise ValueError("pooled training needs >= 2 segments; use full_train / weekly_revalidate for one")
        group_id = f"pool-{_stamp()}"
        pooled = self._fit_pooled_candidates(segments, horizon, group_id)
        # a pooled candidate with no artifact cannot win: strip it so it is neither ranked nor loaded
        winners_ok = {n: p for n, p in pooled.items() if p["artifact_path"]}
        entries = {}
        for sid, df in segments.items():
            current = self.registry.get_active(sid)
            flagged = self.registry.needs_revalidation(sid)
            fit = self._select_and_fit(sid, df, horizon, rule_version, pooled=pooled)
            entries[sid] = (self._weekly_decision(sid, fit, rule_version, current, flagged) if weekly
                            else self._register(sid, fit, rule_version))
        for p in winners_ok.values():  # discard shared artifacts that no registered version references
            if not self.registry.is_referenced(p["artifact_path"]):
                shutil.rmtree(Path(p["artifact_path"]).parent, ignore_errors=True)
        gdir = self.registry.root / "_pooled" / group_id
        if gdir.exists() and not any(gdir.iterdir()):
            gdir.rmdir()
        return entries

    def full_train_pooled(self, segments: dict, horizon: int = 4, rule_version: str = "unknown") -> dict:
        """
        Full training over MANY segments at once (G-03): pooling-capable algorithms train ONE model on all
        series (per-series scaling, no FX conversion) and compete, per segment, against every other candidate.
        Returns {segment_id: RegistryEntry}; segments won by a pooled model share one artifact_path.
        A cross-segment summary file is written to output_dir/cross_segment_summary.json.
        """
        entries = self._run_pooled(segments, horizon, rule_version, weekly=False)
        self._write_cross_segment_summary(entries, horizon, rule_version)
        return entries

    def _write_cross_segment_summary(self, entries: dict, horizon: int, rule_version: str) -> None:
        """One file after a pooled run: winner algorithm and metrics for every segment side by side."""
        import json
        from collections import Counter
        rows = []
        for sid, e in sorted(entries.items()):
            rows.append({"segment_id": sid, "algorithm": e.algorithm_name, "is_fallback": e.is_fallback,
                         "window": e.window, "pooled_group": getattr(e, "pooled_group", None),
                         "metrics": e.metrics, "holdout_metrics": e.holdout_metrics})
        winner_counts = Counter(r["algorithm"] for r in rows)
        payload = {"rule_version": rule_version, "horizon": horizon, "n_segments": len(rows),
                   "winner_distribution": dict(winner_counts.most_common()),
                   "segments": rows}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "cross_segment_summary.json").write_text(
            json.dumps(payload, indent=2, default=str))
        print(f"[orchestrator] cross-segment summary → {self.output_dir / 'cross_segment_summary.json'}")

    def weekly_revalidate_pooled(self, segments: dict, horizon: int = 4, rule_version: str = "unknown") -> dict:
        """Weekly revalidation over many segments: registers a new Production version per segment only on change."""
        return self._run_pooled(segments, horizon, rule_version, weekly=True)

    def daily_infer(self, segment_id: str, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Reads the registry only; fits nothing new -- loads the active (Production) model and calls infer()."""
        segment_df = self._validated(segment_df)
        entry = self.registry.get_active(segment_id)
        if entry is None:
            raise RuntimeError(f"no Production version for segment {segment_id}; run full_train first")

        if entry.is_fallback:
            mod = RollingFallbackModule()
        else:
            mod = self.candidates[entry.algorithm_name]()
        mod.load(Path(entry.artifact_path))
        if getattr(mod, "joint_inference", False):
            raise RuntimeError(f"{entry.algorithm_name} forecasts the pooled series jointly and needs every one of them "
                               f"({entry.pooled_segments}); use daily_infer_pool(segments, horizon)")
        forecast = mod.infer(segment_df, horizon)
        mod.log(
            self.output_dir, segment_id, "daily_inference", _stamp(),
            {"algorithm": entry.algorithm_name, "registry_version": entry.version, "is_fallback": entry.is_fallback,
             "horizon": horizon, "forecast": forecast.to_dict(orient="records")},
        )
        with self.mlflow.run(
            segment_id=segment_id, process="daily_inference", algorithm=entry.algorithm_name,
            rule_version=entry.rule_version, hyperparameters=entry.hyperparameters,
            extra_tags={"registry_version": entry.version},
        ):
            pass
        return forecast

    def daily_infer_pool(self, segments: dict, horizon: int) -> dict:
        """
        Pool-aware daily inference. `segments` = {segment_id: current contract rows}. A segment whose Production model is a
        JOINT model (DeepVAR) is forecast together with the whole pool it was trained on: every pooled series must be supplied,
        and the shared artifact is loaded and run once. Every other segment is forecast exactly as `daily_infer` does.
        Returns {segment_id: [date, forecast]} for the segments supplied.
        """
        segments = {sid: self._validated(df) for sid, df in segments.items()}
        out: dict = {}
        joint_done: dict = {}
        for sid, df in segments.items():
            entry = self.registry.get_active(sid)
            if entry is None:
                raise RuntimeError(f"no Production version for segment {sid}; run full_train first")
            probe = None if entry.is_fallback else self.candidates[entry.algorithm_name]()   # candidates hold FACTORIES
            if probe is None or not getattr(probe, "joint_inference", False):
                out[sid] = self.daily_infer(sid, df, horizon)
                continue
            key = entry.artifact_path
            if key not in joint_done:
                needed = list(entry.pooled_segments or [])
                missing = [x for x in needed if x not in segments]
                if missing:
                    raise ValueError(f"{entry.algorithm_name} for {sid} forecasts {needed} jointly; missing rows for {missing}")
                mod = self.candidates[entry.algorithm_name]()
                mod.load(Path(entry.artifact_path))
                joint_done[key] = mod.infer_pool({x: segments[x] for x in needed}, horizon)
                mod.log(self.output_dir, sid, "daily_inference_pool", _stamp(),
                        {"algorithm": entry.algorithm_name, "registry_version": entry.version, "horizon": horizon, "pool": needed})
            out[sid] = joint_done[key][sid]
        return out

    def _load_registered_model(self, entry: RegistryEntry):
        mod = RollingFallbackModule() if entry.is_fallback else self.candidates[entry.algorithm_name]()
        mod.load(Path(entry.artifact_path))
        return mod

    def monitor_production(self, segment_id: str, actuals_df: pd.DataFrame, history_df: pd.DataFrame):
        """
        Production monitoring (§4.4, G-01): live six-metric rank-sum of Production vs the prior version.
        Only PROPOSES a rollback (recorded in the registry); a human decides via `decide_rollback`.
        `history_df` = canonical rows up to the forecast origin; `actuals_df` = [date, value] realised since.
        """
        result = self.monitor.check(segment_id, actuals_df, self._validated(history_df))
        cur = self.registry.get_active(segment_id)
        with self.mlflow.run(segment_id=segment_id, process="production_monitoring",
                             algorithm=cur.algorithm_name if cur else "none", rule_version=cur.rule_version if cur else "",
                             metrics={f"live_production_{k}": v for k, v in result.production.items() if isinstance(v, (int, float))}
                             | {f"live_prior_{k}": v for k, v in result.prior.items() if isinstance(v, (int, float))},
                             extra_tags={"monitor_action": result.action, "proposal_id": result.proposal_id or "",
                                         "production_version": result.production_version or "", "prior_version": result.prior_version or ""}):
            pass
        return result

    def pending_rollback_proposals(self, segment_id: str | None = None) -> list[dict]:
        return self.registry.pending_proposals(segment_id)

    def decide_rollback(self, segment_id: str, proposal_id: str, approve: bool, reviewer: str, note: str = ""):
        """Human decision on a rollback proposal (approve executes the rollback, reject keeps Production)."""
        restored = self.registry.decide_rollback(segment_id, proposal_id, approve, reviewer, note)
        with self.mlflow.run(segment_id=segment_id, process="rollback_decision", algorithm="registry", rule_version="",
                             extra_tags={"proposal_id": proposal_id, "approved": str(bool(approve)), "reviewer": reviewer,
                                         "note": note, "restored_version": restored.version if restored else ""}):
            pass
        return restored
