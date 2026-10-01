from __future__ import annotations
from pathlib import Path
import pickle
import warnings
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .base import AlgorithmModule
from .utils import endogenous_series, exogenous_frame, future_dates


class StateSpaceModule(AlgorithmModule):
    """
    State Space Models (PRD §3.2 #5): a local-level model via
    statsmodels UnobservedComponents. No eligibility condition --
    state-dimension adequacy is a specification choice, not a testable
    data property; gated only by required observations
    (N >= 10 x state_dimension), which the contract validator's
    min_observations already enforces generically.
    """

    name = "state_space"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.level = self.hyperparameters.get("level", "local level")
        self._results = None
        self._exog_cols: list[str] = []

    def required_observations(self, n_exog: int = 0) -> int:
        # PRD v10 §3.2 row #5: N >= max(50, 10*state_dimension). state
        # dimension is a configuration choice (config.json ->
        # hyperparameters.state_dimension); default 1 for a local-level
        # model (single state: the level itself).
        state_dimension = int(self.hyperparameters.get("state_dimension", 1))
        return max(50, 10 * state_dimension)

    def hyperparameter_search_space(self) -> dict:
        return {"level": {"type": "choice", "choices": ["local level", "local linear trend"]}}

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        exog_aligned = exog.reindex(series.index) if not exog.empty else None
        self._exog_cols = list(exog.columns) if not exog.empty else []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(series, level=self.level, exog=exog_aligned)
            self._results = model.fit(disp=False)
        self._store_statespace_residuals(self._results, burn=1)
        self._fitted_model = {"level": self.level, "aic": float(self._results.aic)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"results": self._results, "level": self.level, "exog_cols": self._exog_cols, "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._results = state["results"]
        self.level = state["level"]
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
