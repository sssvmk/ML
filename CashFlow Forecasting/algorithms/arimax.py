from __future__ import annotations
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import AlgorithmModule, EligibilityResult
from .utils import endogenous_series, exogenous_frame, future_dates


def _stationary_by_adf_and_kpss(series: pd.Series, alpha: float = 0.05) -> tuple[bool, str]:
    """ADF null = unit root (non-stationary); KPSS null = stationary.
    Agreement = ADF rejects its null AND KPSS fails to reject its null."""
    if series.nunique() <= 1:
        return False, "series is constant after differencing"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(series, autolag="AIC")[1]
        try:
            kpss_p = kpss(series, regression="c", nlags="auto")[1]
        except Exception:
            kpss_p = 0.0
    adf_says_stationary = adf_p < alpha
    kpss_says_stationary = kpss_p > alpha
    if adf_says_stationary and kpss_says_stationary:
        return True, f"ADF p={adf_p:.3f} (<{alpha}), KPSS p={kpss_p:.3f} (>{alpha}) -- both agree: stationary"
    return False, f"ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f} -- do not agree on stationarity"


class ARIMAXModule(AlgorithmModule):
    """
    ARIMAX (PRD §3.2 #1). Eligibility: the series must be stationary at
    differencing order `d` -- ADF and KPSS must agree on the differenced
    series. This is one of the 10 of 38 with a genuine
    necessary-and-sufficient pre-fit condition.
    """

    name = "arimax"
    has_eligibility_condition = True

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        p = self.hyperparameters.get("p")
        d = self.hyperparameters.get("d")
        q = self.hyperparameters.get("q")
        if p is not None and d is not None and q is not None:
            self.order = (int(p), int(d), int(q))  # search.py samples p/d/q separately (see hyperparameter_search_space)
        else:
            self.order = tuple(self.hyperparameters.get("order", (1, 1, 1)))  # (p, d, q)
        self._results = None
        self._exog_cols: list[str] = []

    def hyperparameter_search_space(self) -> dict:
        return {
            "p": {"type": "int", "low": 0, "high": 3},
            "d": {"type": "int", "low": 0, "high": 2},
            "q": {"type": "int", "low": 0, "high": 3},
        }

    def _differenced(self, series: pd.Series) -> pd.Series:
        d = self.order[1]
        s = series.copy()
        for _ in range(d):
            s = s.diff().dropna()
        return s

    def required_observations(self, n_exog: int = 0) -> int:
        p, d, q = self.order
        return max(50, 10 * (p + q + n_exog + 1))  # PRD v10 §3.2 row #1

    def _check_eligibility_impl(self, segment_df: pd.DataFrame) -> EligibilityResult:
        series = endogenous_series(segment_df)
        min_n = self.required_observations(n_exog=len(exogenous_frame(segment_df).columns))
        if len(series) < min_n:
            return EligibilityResult(False, f"only {len(series)} observations, need >= {min_n} for order {self.order}")
        diffed = self._differenced(series)
        ok, reason = _stationary_by_adf_and_kpss(diffed)
        return EligibilityResult(ok, reason)

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        exog_aligned = exog.reindex(series.index) if not exog.empty else None
        self._exog_cols = list(exog.columns) if not exog.empty else []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                exog=exog_aligned,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._results = model.fit(disp=False)
        self._store_statespace_residuals(self._results, burn=int(self.order[1]))
        self._fitted_model = {"order": self.order, "aic": float(self._results.aic)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"results": self._results, "order": self.order, "exog_cols": self._exog_cols, "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._results = state["results"]
        self.order = state["order"]
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
            # carry-forward the last known exogenous row for the horizon
            # (Section 4.3: a real future-known-exogenous feed is still open)
            last_row = exog.iloc[-1] if not exog.empty else pd.Series(0, index=self._exog_cols)
            future_exog = pd.DataFrame([last_row.reindex(self._exog_cols)] * horizon, index=dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = self._results.apply(series, exog=exog.reindex(series.index) if self._exog_cols else None)
            fc = refit.get_forecast(steps=horizon, exog=future_exog)
        return pd.DataFrame({"date": dates, "forecast": fc.predicted_mean.to_numpy()})

