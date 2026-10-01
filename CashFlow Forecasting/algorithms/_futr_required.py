"""
Shared base for the models whose eligibility rests on FUTURE-KNOWN covariates (TFT #28, TiDE #29; PRD §3.2: "future-known covariate
values must exist for the full forecast horizon").

Decision (G-28/G-29): a covariate is future-known when it is an exogenous series with values dated after the last endogenous
observation -- inferred from structure, never from a manual flag and never from what the series is called or where it came from.
Calendar features (derived from the date alone) are an optional extra and do NOT satisfy the requirement: eligibility needs data.

Eligibility here is derived, and the reason names what is missing. The model works on one segment or on a pool: in a pool every
series must supply the same number of such series, mapped positionally to shared channels.
"""
from __future__ import annotations

from .base import EligibilityResult
from .nf_adapter import NeuralForecastModule
from .nf_pooled import PooledNeuralForecastModule
from .utils import future_known_datasets, future_length


class FutureKnownRequiredModule(PooledNeuralForecastModule):
    has_eligibility_condition = True
    fk_required = True
    context_family_floor = 1000            # PRD v10 §3.2: N >= max(1,000, context + horizon + 500)
    #: PROVISIONAL (kept from the earlier code, still unconfirmed with the PRD owner): the PRD row says "N", not "pooled N",
    #: so in a pooled run the shortest series must meet it.
    pooled_n_basis = "per_series"
    default_n_lags = 28
    supports_futr_exog = True

    # single series uses NeuralForecastModule's path (no static covariates, no pool)
    def train(self, segment_df) -> None:
        NeuralForecastModule.train(self, segment_df)

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return NeuralForecastModule.required_observations(self, n_exog, horizon)

    def _fk_problem(self, df) -> str | None:
        usable = future_known_datasets(df, min_future=self.horizon)
        if usable:
            return None
        seen = {n: future_length(df, n) for n in future_known_datasets(df)}
        return (f"no future-known covariate series with >= {self.horizon} values after the last observation "
                f"(series with any future values: {seen or 'none'}); calendar features do not count")

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        problem = self._fk_problem(segment_df)
        if problem:
            return EligibilityResult(False, problem)
        names = future_known_datasets(segment_df, min_future=self.horizon)
        return EligibilityResult(True, f"{len(names)} future-known covariate series cover the {self.horizon}-period horizon")

    def _check_pooled_eligibility_impl(self, batch) -> EligibilityResult:
        counts = {}
        for sid, df in batch.segments.items():
            problem = self._fk_problem(df)
            if problem:
                return EligibilityResult(False, f"series {sid}: {problem}")
            counts[sid] = len(future_known_datasets(df, min_future=self.horizon))
        if len(set(counts.values())) != 1:
            return EligibilityResult(False, f"series carry different numbers of future-known covariate series: {counts} "
                                            "(they are mapped positionally, so every series needs the same number)")
        return EligibilityResult(True, f"{batch.n_series} series, each with {next(iter(counts.values()))} future-known covariate series")

    def _model_extra_kwargs(self) -> dict:
        return {}

    def _model_kwargs(self) -> dict:
        kw = super()._model_kwargs()
        kw.update(self._model_extra_kwargs())
        return kw
