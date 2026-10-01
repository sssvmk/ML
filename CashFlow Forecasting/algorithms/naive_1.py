from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .base import AlgorithmModule
from .utils import endogenous_series, future_dates


class Naive1Module(AlgorithmModule):
    """
    Naive-1 (PRD §3.5: "Naive-1 and Seasonal-Naive-30 run alongside [the
    Seasonal-Naive-7 baseline] as challengers"). Repeats the last
    observed value for the whole horizon. Not the elimination gate
    (Seasonal-Naive-7 is) -- computed and logged for comparison only.
    """

    name = "naive_1"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._last_value: float | None = None

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        self._last_value = float(series.iloc[-1])
        self._fitted_model = {"last_value": self._last_value}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"hyperparameters": self.hyperparameters, "model": self._fitted_model}, f)

    def load(self, path: Path) -> None:
        with open(path) as f:
            state = json.load(f)
        self.hyperparameters = state["hyperparameters"]
        self._fitted_model = state["model"]
        self._last_value = self._fitted_model["last_value"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        series = endogenous_series(segment_df)
        last_value = float(series.iloc[-1])
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        return pd.DataFrame({"date": dates, "forecast": [last_value] * horizon})
