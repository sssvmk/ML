"""
Shared plumbing for lag-feature regression algorithms (PRD §3.2, #8-19
excepting the deep-learning family). This is infrastructure boilerplate
-- feature construction, save/load, recursive multi-step inference --
not the "statistical test code" PRD §6 says must not be shared by
default. Each algorithm subclass still owns its own estimator choice,
hyperparameters, and (where it has one) its own eligibility condition.
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from .base import AlgorithmModule
from .utils import endogenous_series, exogenous_frame, future_dates


def build_lag_features(series: pd.Series, exog: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    if not exog.empty:
        df = df.join(exog, how="left")
    return df


class SklearnLagModule(AlgorithmModule):
    """Base for a lag-feature regressor. Subclasses set `name`,
    `has_eligibility_condition`, `default_n_lags`, and `_make_estimator`;
    override `_check_eligibility_impl` if they have a real condition."""

    name = "sklearn_lag_base"
    has_eligibility_condition = False
    default_n_lags = 5

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self.n_lags = int(self.hyperparameters.get("n_lags", self.default_n_lags))
        self._model = None
        self._feature_cols: list[str] = []

    def _make_estimator(self):
        raise NotImplementedError

    def _fit_estimator(self, X, y) -> None:
        """Default: build the estimator and fit on all rows. Gradient-boosted
        tree modules override this to add chronological-validation early stopping."""
        self._model = self._make_estimator()
        self._model.fit(X, y)

    def _split_validation_tail(self, X, y):
        """
        Chronological validation tail inside the training window (G-09;
        PRD §3.2 hyperparameter paragraph): the last `validation_fraction`
        of the rows are held out as the eval_set. Rows are in time order
        (build_lag_features preserves it), so this never shuffles. Returns
        None -- early stopping skipped and recorded as such -- when the
        window is too small to spare a meaningful tail.
        """
        frac = float(self.hyperparameters.get("validation_fraction", 0.2))
        n_val = int(len(X) * frac)
        if n_val < 10 or len(X) - n_val < 20:
            return None
        return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

    @property
    def _es_rounds(self) -> int:
        return int(self.hyperparameters.get("early_stopping_rounds", 20))

    def train(self, segment_df: pd.DataFrame) -> None:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = build_lag_features(series, exog, self.n_lags).dropna()
        self._feature_cols = [c for c in feats.columns if c != "y"]
        X = feats[self._feature_cols].to_numpy()
        y = feats["y"].to_numpy()
        self._fit_estimator(X, y)
        self._store_lag_residuals(X, y)
        self._fitted_model = {"n_lags": self.n_lags, "hyperparameters": self.hyperparameters,
                              "early_stopping": getattr(self, "_early_stopping", None)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self._model, "feature_cols": self._feature_cols, "hyperparameters": self.hyperparameters},
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._feature_cols = state["feature_cols"]
        self.hyperparameters = state["hyperparameters"]
        self.n_lags = int(self.hyperparameters.get("n_lags", self.default_n_lags))

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
                row += exog.iloc[-1].reindex(self._feature_cols[self.n_lags :]).fillna(0).tolist()
            x = np.array(row[: len(self._feature_cols)]).reshape(1, -1)
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat)
            history.loc[d] = yhat
        return pd.DataFrame({"date": dates, "forecast": preds})
