"""
Prophet (PRD §3.2 #7) -- the real `prophet` library (gap G-12; approved
route in the matrix's 'Reuse Options' tab: "Use the real prophet library
(replaces the stand-in)").

Piecewise-linear trend with automatically placed changepoints and a sparse
(Laplace) prior on rate changes, Fourier weekly/yearly seasonality, optional
holiday effects, exogenous regressors, MAP fit. The fitted model exposes its
`changepoints` and `changepoint_prior_scale` (acceptance check for G-12).

Prophet fits through cmdstan; if the deployment image cannot run cmdstan the
module fails at train() with the library's own error -- it never silently
degrades to a different model. (Matrix note: "confirm in your deployment image".)
"""
from __future__ import annotations
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

from .base import AlgorithmModule
from .utils import endogenous_series, exogenous_frame, future_dates

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


class ProphetModule(AlgorithmModule):
    name = "prophet"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        hp = self.hyperparameters
        self.changepoint_prior_scale = float(hp.get("changepoint_prior_scale", 0.05))
        self.n_changepoints = int(hp.get("n_changepoints", 25))
        self.seasonality_prior_scale = float(hp.get("seasonality_prior_scale", 10.0))
        self.seasonality_mode = str(hp.get("seasonality_mode", "additive"))
        self.growth = str(hp.get("growth", "linear"))
        self.country_holidays = hp.get("country_holidays")  # e.g. "IN"; None = no holiday effects
        self._prophet = None
        self._exog_cols: list[str] = []

    def required_observations(self, n_exog: int = 0) -> int:
        return 50  # PRD v10 §3.2 row #7

    def hyperparameter_search_space(self) -> dict:
        return {
            "changepoint_prior_scale": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
            "seasonality_prior_scale": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            "n_changepoints": {"type": "int", "low": 5, "high": 50},
            "seasonality_mode": {"type": "choice", "choices": ["additive", "multiplicative"]},
        }

    def _frame(self, series: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
        for c in self._exog_cols:
            df[c] = exog[c].reindex(series.index).ffill().bfill().fillna(0.0).to_numpy() if not exog.empty else 0.0
        return df

    def train(self, segment_df: pd.DataFrame) -> None:
        from prophet import Prophet
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        self._exog_cols = [str(c) for c in exog.columns] if not exog.empty else []
        m = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            n_changepoints=min(self.n_changepoints, max(1, len(series) // 4)),
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=True,
            yearly_seasonality="auto",
            daily_seasonality=False,
        )
        if self.country_holidays:
            m.add_country_holidays(country_name=str(self.country_holidays))
        for c in self._exog_cols:
            m.add_regressor(c)
        train_df = self._frame(series, exog)
        if self.growth == "logistic":
            train_df["cap"] = float(self.hyperparameters.get("cap", train_df["y"].max() * 1.5))
        m.fit(train_df)  # MAP estimate (Prophet's default L-BFGS)
        self._prophet = m
        self._resid = (train_df["y"].to_numpy() - m.predict(train_df.drop(columns=["y"]))["yhat"].to_numpy())
        self._fitted_model = {
            "changepoint_prior_scale": m.changepoint_prior_scale,
            "n_changepoints_placed": int(len(m.changepoints)),
            "changepoints": [str(c.date()) for c in m.changepoints],
            "trend": "piecewise-" + self.growth,
        }

    @property
    def changepoints(self):
        return None if self._prophet is None else list(self._prophet.changepoints)

    def save(self, path: Path) -> None:
        from prophet.serialize import model_to_json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"model": model_to_json(self._prophet), "exog_cols": self._exog_cols,
                       "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        from prophet.serialize import model_from_json
        state = json.loads(Path(path).read_text())
        self._prophet = model_from_json(state["model"])
        self._exog_cols = state["exog_cols"]
        self.hyperparameters = state["hyperparameters"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._prophet is None:
            raise RuntimeError("load() or train() must run before infer()")
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        fut = pd.DataFrame({"ds": dates})
        for c in self._exog_cols:
            # exogenous regressors are carried forward from the last observed value (as in the
            # statsmodels modules) until a genuine future-known feed exists (PRD §4.3 item 3).
            last = float(exog[c].dropna().iloc[-1]) if (not exog.empty and c in exog.columns and exog[c].notna().any()) else 0.0
            fut[c] = last
        if self.growth == "logistic":
            fut["cap"] = float(self.hyperparameters.get("cap", 1e18))
        yhat = self._prophet.predict(fut)["yhat"].to_numpy()
        return pd.DataFrame({"date": dates, "forecast": yhat})
