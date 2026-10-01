from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import AlgorithmModule
from .utils import endogenous_series, exogenous_frame, future_dates
from .linear_regression import _build_features


class RidgeModule(AlgorithmModule):
    """
    Ridge regression (PRD §3.2 #10). No eligibility condition --
    regularization guarantees invertibility regardless of feature-matrix
    rank, so this always proceeds to training (28-of-38 pattern).
    """

    name = "ridge"
    has_eligibility_condition = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.n_lags = int(self.hyperparameters.get("n_lags", 3))
        self.alpha = float(self.hyperparameters.get("alpha", 1.0))
        self._model: Ridge | None = None
        self._feature_cols: list[str] = []

    def required_observations(self, n_exog: int = 0) -> int:
        return max(50, self.n_lags + n_exog)  # PRD v10 §3.2 row #10

    def hyperparameter_search_space(self) -> dict:
        return {"alpha": {"type": "float", "low": 1e-3, "high": 100.0, "log": True}, "n_lags": {"type": "int", "low": 1, "high": 14}}

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = _build_features(series, exog, self.n_lags).dropna()
        self._feature_cols = [c for c in feats.columns if c != "y"]
        X = feats[self._feature_cols].to_numpy()
        y = feats["y"].to_numpy()
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        self._store_lag_residuals(X, y)
        self._fitted_model = {"coef": self._model.coef_.tolist(), "intercept": float(self._model.intercept_), "alpha": self.alpha}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "feature_cols": self._feature_cols, "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._feature_cols = state["feature_cols"]
        self.hyperparameters = state["hyperparameters"]
        self.n_lags = int(self.hyperparameters.get("n_lags", 3))
        self.alpha = float(self.hyperparameters.get("alpha", 1.0))

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("load() or train() must run before infer()")
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        history = series.copy()
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        preds = []
        for d in dates:
            lags = [history.iloc[-lag] for lag in range(1, self.n_lags + 1)]
            row = list(lags)
            if not exog.empty:
                row += exog.iloc[-1].reindex(self._feature_cols[self.n_lags:]).fillna(0).tolist()
            x = np.array(row[: len(self._feature_cols)]).reshape(1, -1)
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat)
            history.loc[d] = yhat
        return pd.DataFrame({"date": dates, "forecast": preds})
