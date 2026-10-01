from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import AlgorithmModule, EligibilityResult
from .utils import endogenous_series, exogenous_frame, future_dates


def _build_features(series: pd.Series, exog: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    if not exog.empty:
        df = df.join(exog, how="left")
    return df


class LinearRegressionModule(AlgorithmModule):
    """
    Linear Regression on lagged endogenous + exogenous features (PRD
    §3.2 #8). Eligibility: the feature matrix (lags + exog) must have
    full column rank -- otherwise the least-squares fit is not unique.
    """

    name = "linear_regression"
    has_eligibility_condition = True

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.n_lags = int(self.hyperparameters.get("n_lags", 3))
        self._model: LinearRegression | None = None
        self._feature_cols: list[str] = []
        self._last_series: pd.Series | None = None
        self._last_exog: pd.DataFrame | None = None

    def required_observations(self, n_exog: int = 0) -> int:
        k = self.n_lags + n_exog
        return max(50, 10 * (k + 1))  # PRD v10 §3.2 row #8

    def hyperparameter_search_space(self) -> dict:
        return {"n_lags": {"type": "int", "low": 1, "high": 14}}

    def _check_eligibility_impl(self, segment_df: pd.DataFrame) -> EligibilityResult:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = _build_features(series, exog, self.n_lags).dropna()
        if feats.shape[0] < feats.shape[1]:
            return EligibilityResult(False, "fewer observations than features after lagging")
        X = feats.drop(columns=["y"]).to_numpy()
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            return EligibilityResult(False, f"feature matrix rank {rank} < {X.shape[1]} columns (not full rank)")
        return EligibilityResult(True, "feature matrix has full column rank")

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = _build_features(series, exog, self.n_lags).dropna()
        self._feature_cols = [c for c in feats.columns if c != "y"]
        X = feats[self._feature_cols].to_numpy()
        y = feats["y"].to_numpy()
        self._model = LinearRegression()
        self._model.fit(X, y)
        self._store_lag_residuals(X, y)
        self._last_series = series
        self._last_exog = exog
        self._fitted_model = {"coef": self._model.coef_.tolist(), "intercept": float(self._model.intercept_)}

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
                # carry-forward: repeat the last known exogenous row (PRD §4.3
                # notes a real future-known-exogenous feed is a separate,
                # still-open item for algorithms that need one)
                row += exog.iloc[-1].reindex(self._feature_cols[self.n_lags:]).fillna(0).tolist()
            x = np.array(row[: len(self._feature_cols)]).reshape(1, -1)
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat)
            history.loc[d] = yhat
        return pd.DataFrame({"date": dates, "forecast": preds})
