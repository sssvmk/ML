"""
Shared plumbing for ARIMA-family algorithms built on
statsmodels.tsa.statespace.sarimax.SARIMAX (PRD §3.2 #2, #4). ARIMAX
(#1) was written before this template existed and is left as its own
file; SARIMAX and Dynamic Regression subclass this one. Each subclass
still owns its own order/seasonal_order defaults and its own eligibility
condition text -- only the fit/forecast plumbing is shared.
"""

from __future__ import annotations
from pathlib import Path
import pickle
import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import AlgorithmModule, EligibilityResult
from .utils import endogenous_series, exogenous_frame, future_dates


def stationary_by_adf_and_kpss(series: pd.Series, alpha: float = 0.05) -> tuple[bool, str]:
    if series.nunique() <= 1:
        return False, "series is constant after differencing"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(series, autolag="AIC")[1]
        try:
            kpss_p = kpss(series, regression="c", nlags="auto")[1]
        except Exception:
            kpss_p = 0.0
    ok = (adf_p < alpha) and (kpss_p > alpha)
    return ok, f"ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f} (alpha={alpha})"


class StatsmodelsARIMALikeModule(AlgorithmModule):
    """Base for SARIMAX-backed algorithms. Subclasses set `name`,
    `default_order`, `default_seasonal_order`, and `has_eligibility_condition`."""

    name = "arima_like_base"
    has_eligibility_condition = False
    default_order = (1, 1, 1)
    default_seasonal_order = (0, 0, 0, 0)
    #: PRD v10 §3.2 table floor: 100 for SARIMAX (#2), 50 for Dynamic
    #: Regression (#4) -- each subclass sets its own.
    min_observations_floor = 50

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        p = self.hyperparameters.get("p")
        d = self.hyperparameters.get("d")
        q = self.hyperparameters.get("q")
        if p is not None and d is not None and q is not None:
            self.order = (int(p), int(d), int(q))  # search.py samples p/d/q separately
        else:
            self.order = tuple(self.hyperparameters.get("order", self.default_order))
        self.seasonal_order = tuple(self.hyperparameters.get("seasonal_order", self.default_seasonal_order))
        # search.py samples the seasonal terms separately as P/D/Q/m (G-06)
        P, D, Q, m = (self.hyperparameters.get(k) for k in ("P", "D", "Q", "m"))
        if None not in (P, D, Q, m):
            # conditional: with no seasonal AR/MA/differencing the period is meaningless -> seasonality off
            self.seasonal_order = (int(P), int(D), int(Q), int(m)) if (int(P) + int(D) + int(Q)) > 0 else (0, 0, 0, 0)
        self._results = None
        self._exog_cols: list[str] = []

    #: candidate seasonal periods for the searched `m` (daily data: weekly, monthly)
    seasonal_period_choices = (7, 30)

    def hyperparameter_search_space(self) -> dict:
        # G-06 (PRD §3.2/§3.3.1): the seasonal terms P, D, Q and the seasonal
        # period m are searched together with p, d, q. Conditional/data-dependent
        # limits are enforced by is_valid_config() below, not by the ranges here.
        return {
            "p": {"type": "int", "low": 0, "high": 3},
            "d": {"type": "int", "low": 0, "high": 2},
            "q": {"type": "int", "low": 0, "high": 3},
            "P": {"type": "int", "low": 0, "high": 2},
            "D": {"type": "int", "low": 0, "high": 1},
            "Q": {"type": "int", "low": 0, "high": 2},
            "m": {"type": "choice", "choices": list(self.seasonal_period_choices)},
        }

    def is_valid_config(self, available_observations: int, n_exog: int = 0):
        ok, why = super().is_valid_config(available_observations, n_exog)
        if not ok:
            return ok, why
        P, D, Q, m = self.seasonal_order
        if (P + D + Q) > 0:
            # a seasonal model needs at least three full seasonal cycles of history
            need = 3 * m + self.order[1] + D * m
            if available_observations < need:
                return False, f"seasonal period {m} with D={D} needs >= {need} observations, only {available_observations}"
        return True, "ok"

    def _differenced(self, series: pd.Series) -> pd.Series:
        d = self.order[1]
        s = series.copy()
        for _ in range(d):
            s = s.diff().dropna()
        return s

    def required_observations(self, n_exog: int = 0) -> int:
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        return max(self.min_observations_floor, 10 * (p + q + P + Q + n_exog + 1))

    def _check_eligibility_impl(self, segment_df: pd.DataFrame) -> EligibilityResult:
        series = endogenous_series(segment_df)
        k = len(exogenous_frame(segment_df).columns)
        min_n = self.required_observations(n_exog=k)
        if len(series) < min_n:
            return EligibilityResult(False, f"only {len(series)} observations, need >= {min_n} for order {self.order}/{self.seasonal_order}")
        ok, reason = stationary_by_adf_and_kpss(self._differenced(series))
        return EligibilityResult(ok, reason)

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        exog_aligned = exog.reindex(series.index) if not exog.empty else None
        self._exog_cols = list(exog.columns) if not exog.empty else []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series, exog=exog_aligned, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._results = model.fit(disp=False)
        d_burn = self.order[1] + self.seasonal_order[1] * max(self.seasonal_order[3], 1)
        self._store_statespace_residuals(self._results, burn=int(d_burn))
        self._fitted_model = {"order": self.order, "seasonal_order": self.seasonal_order, "aic": float(self._results.aic)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"results": self._results, "order": self.order, "seasonal_order": self.seasonal_order, "exog_cols": self._exog_cols, "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._results = state["results"]
        self.order = state["order"]
        self.seasonal_order = state["seasonal_order"]
        self._exog_cols = state["exog_cols"]
        self.hyperparameters = state["hyperparameters"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("load() or train() must run before infer()")
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        future_exog = None
        if self._exog_cols:
            last_row = exog.iloc[-1] if not exog.empty else pd.Series(0, index=self._exog_cols)
            future_exog = pd.DataFrame([last_row.reindex(self._exog_cols)] * horizon, index=dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = self._results.apply(series, exog=exog.reindex(series.index) if self._exog_cols else None)
            fc = refit.get_forecast(steps=horizon, exog=future_exog)
        return pd.DataFrame({"date": dates, "forecast": fc.predicted_mean.to_numpy()})
