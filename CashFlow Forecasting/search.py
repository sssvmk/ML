"""
Hyperparameter search (PRD v10 §3.3). For an algorithm with a declared
`hyperparameter_search_space()` (§3.3.1), runs:

  §3.3.2 Initial Search  -- randomized sampling of the declared space.
  §3.3.3 Adaptive Optimization -- the remaining budget resamples around the
    best trials (narrowing / local search; a lighter-weight stand-in for a
    full Bayesian optimizer, documented as such).
  §3.3.4 every trial is evaluated with rolling_backtest (the same rolling-
    origin mechanism as model evaluation).
  §3.3.5 Pruning -- FOLD-LEVEL (G-05): after every backtest fold the running
    mean MASE is reported to a median pruner (compared with the completed
    trials' running mean at the same fold count); a clearly inferior trial
    is aborted mid-backtest and recorded as `pruned` with its reason. A
    trial-level check against `prune_margin` x best-so-far is kept as well.
  §3.3.6 Selection -- trials are ranked with the same six-metric rank-sum as
    final algorithm selection (ranking.eliminate_and_rank).
  §3.3.8 Data-dependent space -- a sampled configuration is only evaluated if
    the module itself accepts it for the available history
    (AlgorithmModule.is_valid_config, e.g. SARIMAX N >= max(100,
    10*(p+q+P+Q+k+1))).
  §3.2 JOINT window + hyperparameter optimisation (G-04): when
    `window_bounds` is given the training-window length is one more searched
    dimension, bounded below by the sampled configuration's own
    required_observations(), so the search returns a (window, hyperparameters)
    pair rather than tuning hyperparameters at one fixed window.
  §3.3.10 -- with an MLflow tracker, one NESTED RUN PER TRIAL under a parent
    search run (G-02), with algorithm name+version, config, search-space
    version, backtesting config, fold-level and aggregate six metrics, trial
    status, pruning reason, duration, dataset version id and code version id;
    the selected trial is tagged `selected=true`.
"""

from __future__ import annotations
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field

from backtest import rolling_backtest, METRICS
from ranking import eliminate_and_rank
from versioning import code_version_id, dataset_version_id, search_space_version

MAX_DRAW_ATTEMPTS = 60


@dataclass
class SearchTrial:
    hyperparameters: dict
    status: str  # "completed" | "pruned" | "failed"
    aggregate_metrics: dict = field(default_factory=dict)
    reason: str | None = None
    window: int | None = None
    fold_metrics: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    n_folds_run: int = 0
    overfit: dict = field(default_factory=dict)
    mlflow_run_id: str | None = None
    selected: bool = False


@dataclass
class SearchResult:
    best_hyperparameters: dict
    trials: list[SearchTrial] = field(default_factory=list)
    searched: bool = False  # False when the algorithm declared no search space
    best_window: int | None = None  # set when the window was searched jointly (G-04)
    search_space_version: str | None = None
    parent_run_id: str | None = None


def _sample_param(rng: random.Random, name: str, spec: dict, available_observations: int) -> object:
    ptype = spec.get("type", "float")
    if ptype == "choice":
        return rng.choice(spec["choices"])
    low, high = spec["low"], spec["high"]
    if name == "n_lags" and available_observations:
        high = min(high, max(low, available_observations - 1))
    if ptype == "int":
        return rng.randint(int(low), int(high))
    if spec.get("log"):
        return math.exp(rng.uniform(math.log(max(low, 1e-9)), math.log(max(high, 1e-9))))
    return rng.uniform(low, high)


def _sample_config(rng: random.Random, space: dict, available_observations: int) -> dict:
    return {name: _sample_param(rng, name, spec, available_observations) for name, spec in space.items()}


def _narrow(space: dict, center: dict, shrink: float = 0.35) -> dict:
    """Shrinks each param's range to `shrink` of its original width, centered on `center`'s value."""
    narrowed = {}
    for name, spec in space.items():
        if spec.get("type") == "choice" or name not in center:
            narrowed[name] = spec
            continue
        low, high = spec["low"], spec["high"]
        width = (high - low) * shrink
        c = center[name]
        new_low = max(low, c - width / 2)
        new_high = min(high, c + width / 2)
        if new_low >= new_high:
            new_low, new_high = low, high
        narrowed[name] = {**spec, "low": new_low, "high": new_high}
    return narrowed


class _MedianPruner:
    """Fold-level median pruner (G-05). Thread-confined: one instance per search call."""

    def __init__(self, n_startup_trials: int = 3, n_warmup_folds: int = 2, margin: float = 1.5):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_folds = n_warmup_folds
        self.margin = margin
        self._by_fold: dict[int, list[float]] = {}
        self._n_completed = 0

    def make_callback(self):
        curve: list[tuple[int, float]] = []

        def cb(n_ok_folds: int, running_mean_mase: float):
            curve.append((n_ok_folds, running_mean_mase))
            if n_ok_folds < self.n_warmup_folds or self._n_completed < self.n_startup_trials:
                return None
            ref = self._by_fold.get(n_ok_folds)
            if not ref or len(ref) < self.n_startup_trials:
                return None
            median = statistics.median(ref)
            if running_mean_mase > self.margin * median:
                return (f"fold-level median pruning: running MASE {running_mean_mase:.3f} after {n_ok_folds} folds "
                        f"> {self.margin}x median {median:.3f} of completed trials at the same fold")
            return None

        return cb, curve

    def record_completed(self, curve: list[tuple[int, float]]) -> None:
        self._n_completed += 1
        for k, v in curve:
            self._by_fold.setdefault(k, []).append(v)


def random_then_adaptive_search(
    module_cls,
    base_hyperparameters: dict,
    segment_df,
    window: int,
    horizon: int,
    budget: int = 16,
    seasonal_period: int = 7,
    prune_margin: float = 2.0,
    available_observations: int | None = None,
    seed: int | None = None,
    *,
    window_bounds: tuple[int, int] | None = None,
    n_exog: int = 0,
    step: int | None = None,
    pruner_cfg: dict | None = None,
    tracker=None,
    context: dict | None = None,
    future_known: list | None = None,
) -> SearchResult:
    """
    `window`: the fixed training window used when `window_bounds` is None.
    `window_bounds=(lo, hi)`: search the window jointly (G-04); each sampled
    configuration's own required_observations() raises the lower bound.
    `tracker`: an MLflowLogger (nested per-trial runs, G-02) or None.
    `context`: extra tags for tracking -- segment_id, rule_version, process.
    """
    # A local Random instance, NOT the global `random` module: multiple candidates' searches run
    # concurrently (parallel.py, §3.7); a shared global generator would let one thread's draws
    # interleave with another's, making runs irreproducible even with a fixed seed.
    rng = random.Random(seed)
    probe = module_cls(dict(base_hyperparameters))
    space = probe.hyperparameter_search_space()
    if not space:
        return SearchResult(best_hyperparameters=base_hyperparameters, searched=False, best_window=None)

    ctx = context or {}
    sp_version = search_space_version(space)
    available_observations = available_observations or len(segment_df["date"].unique())
    pruner = _MedianPruner(**(pruner_cfg or {}))
    trials: list[SearchTrial] = []
    best_mase = float("inf")

    # --- parent run for nested per-trial runs (G-02) ---
    parent_id = None
    if tracker is not None and tracker.active:
        parent_id = tracker.start_run_explicit(
            f"{ctx.get('segment_id', 'segment')}/hyperparameter_search/{probe.name}",
            tags={"segment_id": ctx.get("segment_id", ""), "process": "hyperparameter_search",
                  "algorithm": probe.name, "rule_version": ctx.get("rule_version", "")},
        )
        tracker.log_params_explicit(parent_id, {
            "algorithm": probe.name, "algorithm_version": getattr(probe, "algorithm_version", "1"),
            "budget": budget, "seed": seed, "search_space_version": sp_version,
            "search_space": json.dumps(space, default=str), "joint_window_search": window_bounds is not None,
            "dataset_version_id": dataset_version_id(segment_df), "code_version_id": code_version_id(),
        })

    def draw(space_: dict, win_center: int | None = None, win_shrink: float = 0.35):
        """Draws a configuration (and window) the module itself accepts; None if none found."""
        for _ in range(MAX_DRAW_ATTEMPTS):
            hp = _sample_config(rng, space_, available_observations)
            cand = module_cls({**base_hyperparameters, **hp})
            if window_bounds is None:
                ok, _why = cand.is_valid_config(window, n_exog)
                if ok:
                    return hp, None
                continue
            lo, hi = window_bounds
            req = int(cand.required_observations(n_exog=n_exog))
            lo_cfg = max(lo, req)
            if lo_cfg > hi:
                continue
            if win_center is not None:
                w_lo = max(lo_cfg, int(win_center - (hi - lo_cfg) * win_shrink / 2))
                w_hi = min(hi, int(win_center + (hi - lo_cfg) * win_shrink / 2))
                if w_lo > w_hi:
                    w_lo, w_hi = lo_cfg, hi
            else:
                w_lo, w_hi = lo_cfg, hi
            w = rng.randint(w_lo, w_hi)
            ok, _why = cand.is_valid_config(w, n_exog)
            if ok:
                return hp, w
        return None, None

    def run_trial(hp: dict, w: int | None, index: int) -> SearchTrial:
        nonlocal best_mase
        merged = {**base_hyperparameters, **hp}
        use_window = w if w is not None else window
        factory = lambda merged=merged: module_cls(dict(merged))
        cb, curve = pruner.make_callback()
        t0 = time.time()
        result = rolling_backtest(factory, segment_df, use_window, horizon, seasonal_period=seasonal_period,
                                  step=step, pruner=cb, future_known=future_known)
        dur = time.time() - t0
        common = dict(window=use_window if window_bounds is not None else None, fold_metrics=result.folds,
                      duration_s=dur, n_folds_run=len(result.folds), overfit=result.overfit)
        if result.error or (not result.pruned and not result.aggregate.get("mase")):
            trial = SearchTrial(hp, status="failed", reason=result.error or "no MASE produced", **common)
        elif result.pruned:
            trial = SearchTrial(hp, status="pruned", aggregate_metrics=result.aggregate, reason=result.prune_reason, **common)
        else:
            mase = result.aggregate["mase"]
            if mase > prune_margin * best_mase:
                trial = SearchTrial(hp, status="pruned", aggregate_metrics=result.aggregate,
                                    reason=f"MASE {mase:.3f} > {prune_margin}x current best {best_mase:.3f}", **common)
            else:
                best_mase = min(best_mase, mase)
                trial = SearchTrial(hp, status="completed", aggregate_metrics=result.aggregate, **common)
                pruner.record_completed(curve)
        _log_trial(trial, index, result)
        return trial

    def _log_trial(trial: SearchTrial, index: int, result) -> None:
        if parent_id is None:
            return
        rid = tracker.start_run_explicit(
            f"trial_{index:03d}", parent_run_id=parent_id,
            tags={"segment_id": ctx.get("segment_id", ""), "process": "hyperparameter_trial",
                  "algorithm": probe.name, "rule_version": ctx.get("rule_version", ""),
                  "trial_status": trial.status, "pruning_reason": trial.reason or "", "selected": "false"},
        )
        trial.mlflow_run_id = rid
        tracker.log_params_explicit(rid, {
            "algorithm": probe.name, "algorithm_version": getattr(probe, "algorithm_version", "1"),
            **{f"hp.{k}": v for k, v in trial.hyperparameters.items()},
            "window": trial.window if trial.window is not None else window,
            "search_space_version": sp_version,
            "bt.horizon": horizon, "bt.step": step or horizon, "bt.seasonal_period": seasonal_period,
            "dataset_version_id": dataset_version_id(segment_df), "code_version_id": code_version_id(),
            "trial_status": trial.status, "pruning_reason": trial.reason or "",
        })
        ok_folds = [f for f in trial.fold_metrics if "error" not in f]
        for i, f in enumerate(ok_folds, start=1):
            tracker.log_metrics_explicit(rid, {f"fold_{k}": f.get(k) for k in METRICS}, step=i)
        tracker.log_metrics_explicit(rid, {f"agg_{k}": v for k, v in trial.aggregate_metrics.items()})
        tracker.log_metrics_explicit(rid, {"training_duration_s": trial.duration_s, "n_folds_run": trial.n_folds_run})
        tracker.log_metrics_explicit(rid, {f"overfit_{k}": v for k, v in trial.overfit.items()})
        tracker.end_run_explicit(rid, "FAILED" if trial.status == "failed" else "FINISHED")

    # §3.3.2 initial randomized search: ~60% of budget
    initial_n = max(1, int(budget * 0.6))
    for _ in range(initial_n):
        hp, w = draw(space)
        if hp is None:
            trials.append(SearchTrial({}, status="failed", reason="no admissible configuration for the available history"))
            continue
        trials.append(run_trial(hp, w, len(trials)))

    # §3.3.3 adaptive optimization: narrow around the top completed trials, resample the rest
    completed = [t for t in trials if t.status == "completed"]
    remaining = budget - len(trials)
    if completed and remaining > 0:
        ranking = eliminate_and_rank(
            {f"trial_{i}": t.aggregate_metrics for i, t in enumerate(completed)},
            bias_threshold=float("inf"), eliminate_on_mase=False,
        )
        top_idx = [int(name.split("_")[1]) for name, _, _ in ranking.ranked[: max(1, len(completed) // 4)]] or list(range(len(completed)))
        top = [completed[i] for i in top_idx]
        for _ in range(remaining):
            center = rng.choice(top)
            hp, w = draw(_narrow(space, center.hyperparameters), win_center=center.window)
            if hp is None:
                trials.append(SearchTrial({}, status="failed", reason="no admissible configuration for the available history"))
                continue
            trials.append(run_trial(hp, w, len(trials)))

    completed = [t for t in trials if t.status == "completed"]
    result = SearchResult(best_hyperparameters=base_hyperparameters, trials=trials, searched=True,
                          search_space_version=sp_version, parent_run_id=parent_id)
    if completed:
        final_ranking = eliminate_and_rank(
            {f"trial_{i}": t.aggregate_metrics for i, t in enumerate(completed)},
            bias_threshold=float("inf"), eliminate_on_mase=False,
        )
        if final_ranking.winner is not None:
            best = completed[int(final_ranking.winner.split("_")[1])]
            best.selected = True
            result.best_hyperparameters = {**base_hyperparameters, **best.hyperparameters}
            result.best_window = best.window
            if parent_id is not None:
                if best.mlflow_run_id:
                    tracker.set_tags_explicit(best.mlflow_run_id, {"selected": "true"})
                tracker.set_tags_explicit(parent_id, {"selected_trial_run_id": best.mlflow_run_id or ""})
                tracker.log_params_explicit(parent_id, {"selected_hyperparameters": json.dumps(result.best_hyperparameters, default=str),
                                                        "selected_window": best.window})
    if parent_id is not None:
        tracker.log_metrics_explicit(parent_id, {
            "n_trials": len(trials), "n_completed": len(completed),
            "n_pruned": sum(1 for t in trials if t.status == "pruned"),
            "n_failed": sum(1 for t in trials if t.status == "failed"),
        })
        tracker.end_run_explicit(parent_id)
    return result
