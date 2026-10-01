"""
Algorithm module base interface (PRD v10 §5.1: ten independently
callable interfaces -- describe_contract, validate, check_eligibility,
train, save, load, infer, diagnose, evaluate, log).

Every one of the 38 algorithm modules implements this interface and
nothing else -- it is the only thing the Orchestrator is allowed to call
on a module, and the only thing a module is allowed to assume about the
data it receives (the canonical contract rows, Section 4.2).

The ten interfaces are independently callable: the Orchestrator can ask
for eligibility without training, or evaluate a stored forecast without
re-running inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json
import pandas as pd


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str


@dataclass
class DiagnosticsResult:
    ran: bool
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metrics: dict
    charts: dict = field(default_factory=dict)  # chart_name -> file path


class AlgorithmModule(ABC):
    """Base class every algorithm module (§6) extends."""

    #: short machine name, e.g. "arimax" -- used for registry/log keys
    name: str = "base"

    @property
    def frequency(self) -> str:
        """Pandas frequency alias this instance operates at (this request): configured via
        `hyperparameters["frequency"]` (config.py injects config.json -> orchestration.frequency for every
        module), defaulting to "D" so nothing changes unless it's set. Every date-generating call in the
        codebase reads this instead of hard-coding daily."""
        return str(self.hyperparameters.get("frequency", "D"))

    #: version of this module's implementation, recorded on every MLflow trial (§3.3.10).
    #: Bump when a module's modelling behaviour changes.
    algorithm_version: str = "1"

    #: True for the 13 of 38 with a genuine necessary-and-sufficient
    #: eligibility condition (PRD v10 §3.2 row-level table, algorithms
    #: #1-4, 8-9, 23-29); False for the other 25, which always proceed to
    #: training once required observations are met. SVR (#18) and KNN (#19)
    #: are in the False group by decision G-15 (the row-level table
    #: governs: "None" means no necessary test, NOT no implementation --
    #: their required-observations floors still gate them). WaveNet,
    #: DeepState and TiDE inherit True from TCN/DeepAR/TFT respectively
    #: rather than restating it.
    has_eligibility_condition: bool = False

    #: True for the algorithms that can train ONE model instance across many segments (PRD Overview §1:
    #: DeepAR, DeepState, DeepVAR, TFT, TiDE). They are ineligible in the single-segment path (one series is
    #: not a pool) and compete through Orchestrator.full_train_pooled(), where eligibility is derived from the
    #: series actually supplied (G-03).
    pooling_capable: bool = False
    #: which N the required-observations rule is compared with in a pooled run:
    #: "pooled_total" = endogenous observations summed over all series (PRD "Pooled N >= ..."),
    #: "per_series" = the shortest series in the pool.
    pooled_n_basis: str = "pooled_total"
    #: True for a model that forecasts SEVERAL series jointly (DeepVAR): to forecast one series it needs the recent history of
    #: all the others, so it is served through `infer_pool(segments, horizon)` -- one call for the whole pool -- and its
    #: single-segment `infer` refuses. The pooled backtest, the pooled holdout and `Orchestrator.daily_infer_pool` all use it.
    joint_inference: bool = False

    def __init__(self, hyperparameters: dict | None = None):
        self.hyperparameters = hyperparameters or {}
        self._fitted_model = None

    # 1. input data contract -------------------------------------------------
    def describe_contract(self) -> dict:
        """
        Data expectations beyond the canonical fields (Section 5). Default:
        one endogenous series, any number of exogenous series, no extra
        columns required. Override to publish additional requirements.
        """
        return {
            "requires_endogenous": True,
            "requires_exogenous": False,
            "extra_columns": [],
        }

    # 2. universal validation + eligibility ----------------------------------
    def validate(self, segment_df: pd.DataFrame):
        """
        PRD v10 §5.1 `validate()`: pass/fail against the universal
        pre-fit checks (§3.2 step 1), exposed as this algorithm's own
        callable interface. Delegates to the single shared implementation
        in contract.py (the same gate the Orchestrator applies to a whole
        segment/batch before any algorithm sees it) rather than
        reimplementing the checks per module -- there is exactly one
        universal-validation rule set, applied identically to all 38.
        """
        from contract import validate_contract  # local import: avoids a module-load cycle with contract.py
        return validate_contract(segment_df)

    def check_eligibility(self, segment_df: pd.DataFrame) -> EligibilityResult:
        """
        Evaluate this algorithm's necessary-and-sufficient pre-fit
        condition, if it has one (PRD §3.2 table). An algorithm with no
        such condition (25 of 38) always proceeds to training.
        """
        if not self.has_eligibility_condition:
            return EligibilityResult(eligible=True, reason="no eligibility condition defined")
        return self._check_eligibility_impl(segment_df)

    def _check_eligibility_impl(self, segment_df: pd.DataFrame) -> EligibilityResult:
        # Overridden by the 13 algorithms with a genuine condition.
        return EligibilityResult(eligible=True, reason="not overridden")

    # pooled path (G-03) -------------------------------------------------------------
    def check_pooled_eligibility(self, batch) -> EligibilityResult:
        """Eligibility for a POOLED run, derived from the batch actually supplied (never a manual flag)."""
        if not self.pooling_capable:
            return EligibilityResult(False, f"{self.name} is not a pooling-capable algorithm")
        if batch.n_series < 2:
            return EligibilityResult(False, f"pool has {batch.n_series} series, need >= 2")
        req = int(self.required_observations())
        have = batch.total_obs if self.pooled_n_basis == "pooled_total" else batch.min_series_obs
        if have < req:
            return EligibilityResult(False, f"{self.pooled_n_basis} observations {have} < required {req}")
        return self._check_pooled_eligibility_impl(batch)

    def _check_pooled_eligibility_impl(self, batch) -> EligibilityResult:
        return EligibilityResult(True, f"{batch.n_series} series, {batch.total_obs} pooled observations")

    def train_pooled(self, batch) -> None:
        """Fit ONE model on every series of a pooling.PooledBatch. Only pooling-capable modules implement it."""
        raise NotImplementedError(f"{self.name} does not implement pooled training")

    def required_observations(self, n_exog: int = 0) -> int:
        """
        This algorithm's required-observation floor (PRD v10 §3.2
        row-level table, rightmost column), given `n_exog` exogenous
        columns available. Drives both the pre-fit gate and the
        per-algorithm searchable rolling-window sizing (§3.2: "the
        training window length is a per-algorithm searchable
        hyperparameter, sized to meet or exceed that algorithm's own
        required-observations threshold ... not a single fixed value
        applied uniformly across all 38 algorithms").

        Default here is a conservative floor read from hyperparameters
        (or 50) for the algorithms that have not yet overridden this
        with their PRD-table formula; overriding modules replace this
        with their actual N-formula (e.g. arimax overrides with
        max(50, 10*(p+q+k+1))).
        """
        return int(self.hyperparameters.get("min_observations", 50))

    # hyperparameter search space (§3.3) --------------------------------------
    def hyperparameter_search_space(self) -> dict:
        """
        This algorithm's tunable hyperparameters, valid ranges/choices,
        and any data-dependent limits (§3.3.1, §3.3.8), as
        {param_name: {"type": "int"|"float"|"choice", "low":..,
        "high":.., "choices":[...], "log": bool}}.

        Default: {} -- no declared search space, meaning this
        algorithm's configured hyperparameters (config.json) are used
        as-is and it is skipped by the hyperparameter-search stage
        (search.py). Algorithms with a declared space are tuned via
        search.random_then_adaptive_search(); the remainder are a
        tracked backlog item (see ALIGNMENT_v10.md) -- §3.3 requires an
        algorithm-specific space for all 38, and not all 38 have one
        defined yet.
        """
        return {}

    def is_valid_config(self, available_observations: int, n_exog: int = 0) -> tuple[bool, str]:
        """
        Data-dependent validity of THIS instance's hyperparameters
        (§3.3.8, G-04/G-06): a sampled configuration is admissible only if
        the history it would be trained on meets its own PRD
        required-observations threshold, e.g. SARIMAX
        N >= max(100, 10*(p+q+P+Q+k+1)). Used by search.py so it never
        evaluates a configuration the module itself would refuse.
        """
        req = self.required_observations(n_exog=n_exog)
        if available_observations < req:
            return False, f"needs {req} observations, only {available_observations} available"
        return True, "ok"

    # 3. train ----------------------------------------------------------------
    @abstractmethod
    def train(self, segment_df: pd.DataFrame) -> None:
        """Fit the model on this segment's canonical contract rows."""
        raise NotImplementedError

    # 4. post-fit diagnostics (model-comparison stage) -----------------------
    def residuals(self):
        """
        In-sample one-step residuals of the fitted model as a 1-D numpy
        array (chronological), or None when this module cannot supply
        them. Modules store them in `self._resid` at train() time (see
        the helpers below); diagnose() is built on this.
        """
        return getattr(self, "_resid", None)

    def _store_lag_residuals(self, X, y) -> None:
        """Regression family: in-sample residuals of the just-fitted `self._model`."""
        import numpy as np
        try:
            self._resid = np.asarray(y, dtype=float) - np.asarray(self._model.predict(X), dtype=float).ravel()
        except Exception:
            self._resid = None

    def _store_statespace_residuals(self, results, burn: int = 0) -> None:
        """Statistical family: one-step residuals of a fitted statsmodels result."""
        import numpy as np
        try:
            r = results.resid
            r = r.iloc[:, 0] if hasattr(r, "iloc") and getattr(r, "ndim", 1) == 2 else r
            r = np.asarray(r, dtype=float)
            self._resid = r[burn:]
        except Exception:
            self._resid = None

    def diagnose(self) -> DiagnosticsResult:
        """
        Post-fit residual diagnostics (G-08; PRD §5 item 4 / §5.1). Runs
        where the module can supply residuals: residual mean/std, lag-1
        autocorrelation, Ljung-Box (lags 10 and 20, model-df aware is not
        applied: reported as the plain statistic/p-value), and a
        Jarque-Bera normality check as a calibration signal. Never feeds
        back into eligibility (PRD §6, item 4).
        """
        res = self.residuals()
        if res is None:
            return DiagnosticsResult(ran=False, details={"reason": "module supplies no residuals"})
        import numpy as np
        r = np.asarray(res, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 12:
            return DiagnosticsResult(ran=False, details={"reason": f"only {len(r)} finite residuals"})
        details: dict = {
            "n_residuals": int(len(r)), "mean": float(r.mean()), "std": float(r.std(ddof=1)),
        }
        if details["std"] > 0:
            details["lag1_autocorrelation"] = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.stats.stattools import jarque_bera
            lags = [l for l in (10, 20) if l < len(r) // 2] or [max(1, len(r) // 4)]
            lb = acorr_ljungbox(r, lags=lags, return_df=True)
            details["ljung_box"] = {
                int(l): {"statistic": float(lb.loc[l, "lb_stat"]), "p_value": float(lb.loc[l, "lb_pvalue"])}
                for l in lags
            }
            jb_stat, jb_p, skew, kurt = jarque_bera(r)
            details["jarque_bera"] = {"statistic": float(jb_stat), "p_value": float(jb_p), "skew": float(skew), "kurtosis": float(kurt)}
        except Exception as exc:  # diagnostics must never break a run
            details["diagnostic_error"] = str(exc)
        return DiagnosticsResult(ran=True, details=details)

    def in_sample_forecast_check(self, train_df: pd.DataFrame, horizon: int, seasonal_period: int = 7):
        """
        Training-set metrics for the overfitting signal (G-10): the fitted
        module forecasts the LAST `horizon` periods of its own training
        window from the history that precedes them and is scored against
        those (in-sample) actuals with the same six metrics. Modules whose
        infer() ignores the supplied history (or that cannot do this)
        return None; the comparison is then skipped for them, never faked.
        """
        try:
            endog_rows = train_df[train_df["series_role"] == "endogenous"]
            dates = sorted(endog_rows["date"].unique())
            if len(dates) <= horizon + 10:
                return None
            cut = dates[-horizon]
            from .utils import future_known_datasets
            fk = future_known_datasets(train_df)
            is_fk = (train_df["series_role"] == "exogenous") & train_df["dataset"].isin(fk)
            # endogenous AND historical exogenous are truncated at the cut (no leakage); future-known series are kept
            # through the last training date, i.e. exactly the `horizon` periods being forecast
            history = train_df[(train_df["date"] < cut) | (is_fk & (train_df["date"] <= dates[-1]))]
            actual = train_df[(train_df["date"] >= cut) & (train_df["series_role"] == "endogenous")]
            actual = actual.groupby("date")["value"].sum().reset_index()
            fc = self.infer(history, horizon)
            from .utils import endogenous_series
            ev = self.evaluate(fc, actual, insample_series=endogenous_series(history), seasonal_period=seasonal_period)
            return None if "error" in ev.metrics else ev.metrics
        except Exception:
            return None

    # 5 & 6. save / load -------------------------------------------------------
    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model artifact and hyperparameters used."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Reconstitute a saved model. Independent of training: this does NOT
        carry data forward -- every infer() call supplies current data.
        """
        raise NotImplementedError

    # 7. inference --------------------------------------------------------------
    def infer_pool(self, segments: dict, horizon: int) -> dict:
        """
        Forecast every segment in `segments` ({segment_id: contract rows}). The default is one `infer` per segment; a
        `joint_inference` model overrides it to forecast all series together and returns {segment_id: [date, forecast]}.
        """
        return {sid: self.infer(df, horizon) for sid, df in segments.items()}

    @abstractmethod
    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Produce a forecast for exactly `horizon` future periods, given the
        loaded model plus the current canonical contract rows (endogenous
        history and any exogenous). Returns a DataFrame with columns
        [date, forecast].
        """
        raise NotImplementedError

    # 8. evaluation ------------------------------------------------------------
    def evaluate(
        self,
        forecast_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        insample_series: pd.Series | None = None,
        seasonal_period: int = 7,
        chart_dir: Path | None = None,
        chart_tag: str = "eval",
    ) -> EvaluationResult:
        """
        Compute all six PRD §3.4 metrics: sMAPE, WAPE, MAE, RMSE, MASE,
        Bias %.

        `insample_series` is the endogenous training-window history
        (pre-forecast) used to scale MASE, per Hyndman & Koehler: MASE's
        denominator is the in-sample mean absolute error of a
        seasonal-naive(seasonal_period) forecast on the training data
        the model actually saw -- NOT the held-out fold. `seasonal_period`
        should match the baseline's season (default 7, i.e.
        Seasonal-Naive-7, PRD §3.5's default baseline) so that MASE < 1
        means "beat the baseline" exactly as §3.5 step 1 requires.

        If `insample_series` is omitted (e.g. a direct unit-level call),
        MASE falls back to scaling by the naive-1 in-sample error of the
        actuals themselves -- a documented approximation, not the
        specified in-sample scale.
        """
        merged = forecast_df.merge(actuals_df, on="date", suffixes=("_fcst", "_actual"))
        if merged.empty:
            return EvaluationResult(metrics={"error": "no overlapping dates"})

        f = merged["forecast"].to_numpy(dtype=float)
        a = merged["value"].to_numpy(dtype=float)
        err = f - a
        abs_err = abs(err)

        mae = float(abs_err.mean())
        rmse = float((err ** 2).mean() ** 0.5)

        # sMAPE (symmetric, 0-200 scale as commonly defined; guards 0/0 -> 0)
        smape_denom = (abs(f) + abs(a))
        smape_terms = [200.0 * ae / d if d > 0 else 0.0 for ae, d in zip(abs_err, smape_denom)]
        smape = float(sum(smape_terms) / len(smape_terms)) if smape_terms else None

        # WAPE: sum of absolute error over sum of absolute actuals
        sum_abs_a = abs(a).sum()
        wape = float(abs_err.sum() / sum_abs_a * 100) if sum_abs_a > 0 else None

        # Bias %: signed error as a share of total actuals (sign shows
        # over- vs under-forecasting; magnitude is what the configured
        # bias_threshold gates in the elimination step, §3.5 step 2)
        bias_pct = float(err.sum() / sum_abs_a * 100) if sum_abs_a > 0 else None

        # MASE: in-sample seasonal-naive MAE is the scale
        if insample_series is not None and len(insample_series) > seasonal_period:
            naive_diffs = insample_series.diff(seasonal_period).dropna().abs()
            scale = float(naive_diffs.mean()) if len(naive_diffs) else float("nan")
        else:
            naive_diffs = pd.Series(a).diff(1).dropna().abs()
            scale = float(naive_diffs.mean()) if len(naive_diffs) else float("nan")
        mase = float(mae / scale) if scale and scale > 0 else None

        charts: dict = {}
        if chart_dir is not None:
            # G-07 (PRD §5 item 7): forecast-vs-actual and residual charts,
            # persisted in the config-defined log folder.
            from charts import forecast_vs_actual_chart, residual_chart
            charts["forecast_vs_actual"] = str(forecast_vs_actual_chart(
                merged["date"], a, f, Path(chart_dir), f"{self.name}_{chart_tag}_forecast_vs_actual", f"{self.name}: forecast vs actual"))
            charts["residuals"] = str(residual_chart(
                merged["date"], err, Path(chart_dir), f"{self.name}_{chart_tag}_residuals", f"{self.name}: residuals (forecast - actual)"))

        return EvaluationResult(
            metrics={
                "smape": smape,
                "wape": wape,
                "mae": mae,
                "rmse": rmse,
                "mase": mase,
                "bias_pct": bias_pct,
            },
            charts=charts,
        )

    # 9. logging ------------------------------------------------------------------
    def log(self, output_dir: Path, segment_id: str, process: str, window: str, payload: dict) -> Path:
        """
        Write this stage's outputs to the config-defined output location,
        keyed by segment/process/algorithm/window (PRD §6, item 8).
        """
        out = Path(output_dir) / segment_id / process / self.name / window
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / "log.json"
        with open(out_file, "w") as f:
            json.dump(payload, f, default=str, indent=2)
        return out_file
