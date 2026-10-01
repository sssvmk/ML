from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .base import AlgorithmModule
from .utils import endogenous_series, future_dates


class RollingFallbackModule(AlgorithmModule):
    """
    Rolling mean/median fallback (PRD §4.2 #14): used only when no
    algorithm survives elimination against the baseline for a given
    segment/window -- never evaluated as a peer candidate every window.
    """

    name = "fallback"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.combine = self.hyperparameters.get("combine", "rolling_mean_median")
        self.lookback = int(self.hyperparameters.get("lookback", 14))

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        window = series.tail(self.lookback)
        self._fitted_model = {"mean": float(window.mean()), "median": float(window.median())}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"hyperparameters": self.hyperparameters, "model": self._fitted_model}, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            state = json.load(f)
        self.hyperparameters = state["hyperparameters"]
        self.combine = self.hyperparameters.get("combine", "rolling_mean_median")
        self.lookback = int(self.hyperparameters.get("lookback", 14))
        self._fitted_model = state["model"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        series = endogenous_series(segment_df)
        window = series.tail(self.lookback)
        value = (window.mean() + window.median()) / 2 if self.combine == "rolling_mean_median" else window.mean()
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        return pd.DataFrame({"date": dates, "forecast": [float(value)] * horizon})
