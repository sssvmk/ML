from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .base import AlgorithmModule
from .utils import endogenous_series, future_dates


class SeasonalNaiveModule(AlgorithmModule):
    """
    Seasonal-Naive-k baseline (PRD §4.2 #8: Seasonal-Naive-7 is the
    default baseline every candidate must beat; Naive-1 and
    Seasonal-Naive-30 are challengers -- set via hyperparameters).
    No eligibility condition: this is the elimination-gate baseline, not
    a candidate algorithm, but it implements the same interface so the
    Orchestrator treats it uniformly.
    """

    name = "seasonal_naive"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.season = int(self.hyperparameters.get("season", 7))
        self._series: pd.Series | None = None

    def train(self, segment_df: pd.DataFrame) -> None:
        self._series = endogenous_series(segment_df)
        self._fitted_model = {"season": self.season, "last_values": self._series.tail(self.season).tolist()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"hyperparameters": self.hyperparameters, "model": self._fitted_model}, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            state = json.load(f)
        self.hyperparameters = state["hyperparameters"]
        self.season = int(self.hyperparameters.get("season", 7))
        self._fitted_model = state["model"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        series = endogenous_series(segment_df)
        last_season = series.tail(self.season)
        if last_season.empty:
            raise ValueError("not enough history to seasonal-naive forecast")
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        values = [last_season.iloc[i % len(last_season)] for i in range(horizon)]
        return pd.DataFrame({"date": dates, "forecast": values})
