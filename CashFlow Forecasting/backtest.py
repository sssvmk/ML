"""
Rolling-origin backtest (PRD v10 §3.2: "Use rolling backtesting so that
every algorithm predicts the same historical periods that are already
known.") Shared by the orchestrator and by search.py's hyperparameter
tuning (§3.3.4), so there is one mechanism, not two copies that could drift.

Beyond the six metrics per fold this also records, per fold (all optional
for callers that only read `.aggregate` / `.folds`):
  * the forecast and actuals (for charts, G-07);
  * training-window metrics and the train-vs-validation gap (overfitting
    signal, G-10);
  * post-fit residual diagnostics (G-08);
  * eligibility reasons when a fold was skipped.
A `pruner` callback can abort the backtest mid-way when the candidate is
demonstrably inferior (fold-level pruning, G-05).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import pandas as pd

from algorithms.utils import endogenous_series, future_known_datasets

METRICS = ("smape", "wape", "mae", "rmse", "mase", "bias_pct")


@dataclass
class BacktestResult:
    aggregate: dict = field(default_factory=dict)  # mean of each metric across ok folds
    folds: list[dict] = field(default_factory=list)  # per-fold metric dicts (ok and error alike)
    n_folds: int = 0
    error: str | None = None
    fold_details: list[dict] = field(default_factory=list)  # aligned with `folds`
    overfit: dict = field(default_factory=dict)  # aggregate train/validation/gap (G-10)
    eligibility_reasons: list[str] = field(default_factory=list)
    pruned: bool = False
    prune_reason: str | None = None
    duration_s: float = 0.0


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def rolling_backtest(
    module_factory,
    segment_df: pd.DataFrame,
    window: int,
    horizon: int,
    seasonal_period: int = 7,
    step: int | None = None,
    pruner=None,
    collect_details: bool = True,
    future_known: list[str] | None = None,
) -> BacktestResult:
    """
    Roll a window/horizon split across the segment's history, retraining
    a fresh module instance each fold (no leakage across folds -- every
    fold's `insample_series` for MASE scaling is that fold's OWN training
    window, per PRD's Hyndman & Koehler scaling requirement, not the
    full segment history).

    `pruner(n_ok_folds, running_mean_mase) -> reason | None`: called after
    each successful fold; a non-None reason aborts the backtest and the
    result comes back with pruned=True and the partial aggregate.
    """
    t_start = time.time()
    dates = sorted(segment_df.loc[segment_df["series_role"] == "endogenous", "date"].unique())   # target dates only
    n = len(dates)
    step = step or horizon
    # Future-known covariates (inferred from structure, G-28/G-29): exogenous series that are known ahead of time are
    # handed to the module through the fold's TEST window as well, exactly as they would be at a real forecast origin.
    # The caller passes the names when `segment_df` is already truncated (the orchestrator's selection slice); otherwise
    # they are detected here. Every other exogenous series is still cut at the training end (no leakage).
    fk = list(future_known) if future_known is not None else future_known_datasets(segment_df)
    if n < window + horizon:
        return BacktestResult(error=f"only {n} dates, need >= {window + horizon} for one backtest fold")

    fold_metrics: list[dict] = []
    fold_details: list[dict] = []
    reasons: list[str] = []
    start = 0
    fold_no = 0
    pruned_reason = None
    while start + window + horizon <= n:
        fold_no += 1
        train_dates = set(dates[start: start + window])
        test_dates = set(dates[start + window: start + window + horizon])
        train_end = dates[start + window - 1]
        # Exogenous rows are limited to dates on or before the training window's end: the original
        # construction passed ALL exogenous rows (including those AFTER the forecast origin), so the
        # carried-forward exogenous value at forecast time came from the future -- leakage.
        test_end = dates[start + window + horizon - 1]
        is_exog = segment_df["series_role"] == "exogenous"
        is_fk = is_exog & segment_df["dataset"].isin(fk)
        train_df = segment_df[
            segment_df["date"].isin(train_dates)
            | (is_exog & ~is_fk & (segment_df["date"] <= train_end))
            | (is_fk & (segment_df["date"] <= test_end))
        ]
        test_actuals = segment_df[
            (segment_df["date"].isin(test_dates)) & (segment_df["series_role"] == "endogenous")
        ][["date", "value"]]
        # the same endogenous-sum convention the modules use, so a multi-endogenous segment lines up
        test_actuals = test_actuals.groupby("date")["value"].sum().reset_index()
        detail: dict = {"fold": fold_no, "train_start": str(dates[start])[:10],
                        "train_end": str(dates[start + window - 1])[:10], "test_end": str(dates[start + window + horizon - 1])[:10]}
        try:
            mod = module_factory()
            elig = mod.check_eligibility(train_df)
            if not elig.eligible:
                if elig.reason not in reasons:
                    reasons.append(elig.reason)
                start += step
                continue
            insample_series = endogenous_series(train_df)
            mod.train(train_df)
            fc = mod.infer(train_df, horizon)
            result = mod.evaluate(fc, test_actuals, insample_series=insample_series, seasonal_period=seasonal_period)
            if "error" not in result.metrics:
                fold_metrics.append(result.metrics)
                if collect_details:
                    detail["forecast"] = [(str(d)[:10], float(v)) for d, v in zip(fc["date"], fc["forecast"])]
                    detail["actual"] = [(str(d)[:10], float(v)) for d, v in zip(test_actuals["date"], test_actuals["value"])]
                    train_metrics = mod.in_sample_forecast_check(train_df, horizon, seasonal_period)
                    if train_metrics:
                        detail["train_metrics"] = train_metrics
                        detail["gap"] = {k: (result.metrics[k] - train_metrics[k])
                                         for k in METRICS if result.metrics.get(k) is not None and train_metrics.get(k) is not None}
                    diag = mod.diagnose()
                    if diag.ran:
                        detail["diagnostics"] = diag.details
                    es = (getattr(mod, "_fitted_model", None) or {}).get("early_stopping") if isinstance(getattr(mod, "_fitted_model", None), dict) else None
                    if es:
                        detail["early_stopping"] = es
            else:
                fold_metrics.append({"error": result.metrics["error"]})
        except Exception as exc:  # a candidate that fails to fit this fold is skipped, not fatal
            fold_metrics.append({"error": str(exc)})
            detail["error"] = str(exc)
        fold_details.append(detail)
        start += step

        if pruner is not None:
            ok_now = [m for m in fold_metrics if "error" not in m]
            mases = [m.get("mase") for m in ok_now if m.get("mase") is not None]
            if mases:
                pruned_reason = pruner(len(ok_now), sum(mases) / len(mases))
                if pruned_reason:
                    break

    ok_folds = [m for m in fold_metrics if "error" not in m]
    if not ok_folds:
        return BacktestResult(
            folds=fold_metrics, n_folds=0, fold_details=fold_details, eligibility_reasons=reasons,
            error=(f"no fold produced a valid forecast ({len(fold_metrics)} attempted"
                   + (f"; ineligible: {reasons[0]}" if reasons else "") + ")"),
            duration_s=time.time() - t_start,
        )

    aggregate = {k: _mean([f.get(k) for f in ok_folds]) for k in ok_folds[0].keys()}

    overfit: dict = {}
    with_gap = [d for d in fold_details if "train_metrics" in d and "gap" in d]
    if with_gap:
        overfit = {
            "folds_with_train_metrics": len(with_gap),
            **{f"train_{k}": _mean([d["train_metrics"].get(k) for d in with_gap]) for k in METRICS},
            **{f"validation_{k}": _mean([d["gap"] and (d["train_metrics"].get(k) + d["gap"][k]) if k in d["gap"] else None for d in with_gap]) for k in METRICS},
            **{f"gap_{k}": _mean([d["gap"].get(k) for d in with_gap]) for k in METRICS},
        }

    return BacktestResult(
        aggregate=aggregate, folds=fold_metrics, n_folds=len(ok_folds), fold_details=fold_details,
        overfit=overfit, eligibility_reasons=reasons, pruned=bool(pruned_reason), prune_reason=pruned_reason,
        duration_s=time.time() - t_start,
    )


def pooled_rolling_backtest(
    module_factory,
    segments: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    seasonal_period: int = 7,
    step: int | None = None,
    future_known: dict | None = None,
) -> tuple[dict[str, BacktestResult], list[str]]:
    """
    Rolling-origin backtest of ONE pooled model over many segments (G-03). Every fold trains a fresh instance on
    the training slice of ALL segments at once (`pooling.slice_pool`), then forecasts and scores EACH segment
    separately -- MASE is scaled by that segment's own training window, exactly as in the single-segment
    backtest, so a pooled model's per-segment results are directly comparable with every other candidate's.

    Returns ({segment_id: BacktestResult}, eligibility_reasons). Folds where the pool is ineligible are skipped.
    """
    from pooling import slice_pool
    t_start = time.time()
    dates = sorted(set().union(*(set(df.loc[df["series_role"] == "endogenous", "date"].unique()) for df in segments.values())))
    n = len(dates)
    step = step or horizon
    reasons: list[str] = []
    per_seg_folds: dict[str, list[dict]] = {sid: [] for sid in segments}
    if n < window + horizon:
        err = f"only {n} dates, need >= {window + horizon} for one backtest fold"
        return {sid: BacktestResult(error=err) for sid in segments}, reasons

    start = 0
    while start + window + horizon <= n:
        train_dates = set(dates[start: start + window])
        test_dates = set(dates[start + window: start + window + horizon])
        train_end = dates[start + window - 1]
        batch = slice_pool(segments, train_dates, train_end, future_known=future_known,
                           through=dates[start + window + horizon - 1])
        try:
            mod = module_factory()
            elig = mod.check_pooled_eligibility(batch)
            if not elig.eligible:
                if elig.reason not in reasons:
                    reasons.append(elig.reason)
                start += step
                continue
            mod.train_pooled(batch)
            joint_fc = mod.infer_pool(batch.segments, horizon) if getattr(mod, "joint_inference", False) else None
            for sid in batch.segment_ids:
                actual = segments[sid][(segments[sid]["date"].isin(test_dates)) & (segments[sid]["series_role"] == "endogenous")]
                actual = actual.groupby("date")["value"].sum().reset_index()
                if len(actual) < horizon:
                    continue  # this series has no complete test period in this fold
                train_df = batch.segments[sid]
                fc = joint_fc[sid] if joint_fc is not None else mod.infer(train_df, horizon)
                res = mod.evaluate(fc, actual, insample_series=endogenous_series(train_df), seasonal_period=seasonal_period)
                per_seg_folds[sid].append(res.metrics if "error" not in res.metrics else {"error": res.metrics["error"]})
        except Exception as exc:  # a fold that fails to fit is recorded for every series in it, not fatal
            for sid in batch.segment_ids:
                per_seg_folds[sid].append({"error": f"{type(exc).__name__}: {exc}"})
        start += step

    out: dict[str, BacktestResult] = {}
    dur = time.time() - t_start
    for sid, folds in per_seg_folds.items():
        ok = [m for m in folds if "error" not in m]
        if not ok:
            out[sid] = BacktestResult(folds=folds, n_folds=0, eligibility_reasons=reasons, duration_s=dur,
                                      error="no fold produced a valid forecast" + (f"; ineligible: {reasons[0]}" if reasons else ""))
        else:
            out[sid] = BacktestResult(aggregate={k: _mean([f.get(k) for f in ok]) for k in ok[0].keys()}, folds=folds,
                                      n_folds=len(ok), eligibility_reasons=reasons, duration_s=dur)
    return out, reasons
