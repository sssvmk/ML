"""
Shared simplified backbone for the 19 "deep learning family" algorithms
(PRD §3.2 #20-38). This environment has no GPU and no deep-learning
framework installed, and faithfully reproducing 19 published
architectures (RNN/LSTM/GRU/TCN/WaveNet/DeepAR/DeepState/DeepVAR/TFT/
TiDE/TSMixer/TimesNet/iTransformer/Informer/Autoformer/FEDformer/
ETSformer/N-BEATS/N-HiTS) is out of scope for this harness -- see the
Design doc's implementation plan.

Each subclass here uses a small feed-forward network (scikit-learn's
MLPRegressor) over lagged inputs as a stand-in that satisfies the same
nine-interface contract (§6.1) and can be eliminated against the
baseline like any other candidate. It demonstrates the architecture --
loose coupling, independent eligibility, the registry/orchestrator path
-- not the modeling behavior the published architecture would have.
That distinction is repeated on every subclass's docstring and in the
traceability matrix, not just here.
"""

from __future__ import annotations
from sklearn.neural_network import MLPRegressor

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from ._sklearn_template import SklearnLagModule, build_lag_features
from .utils import endogenous_series, future_dates


class NeuralLagModule(SklearnLagModule):
    """Base for the deep-learning-family stand-ins. `default_n_lags` and
    `hidden_layer_sizes` are the only things most subclasses change."""

    name = "neural_lag_base"
    default_n_lags = 14
    default_hidden_layer_sizes = (32, 16)
    is_simplified_stand_in = True
    #: PRD v10 §3.2: RNN/LSTM/GRU/N-BEATS/N-HiTS (#20-22,37,38) use
    #: N >= lookback+horizon+500 with no extra floor; the
    #: attention/mixer/context family (#30-36) uses
    #: N >= max(1000, context+horizon+500). Subclasses in that second
    #: group set this to 1000; everyone else leaves it at 0.
    context_family_floor = 0
    _pooled = None  # set by train_pooled(): {"scales", "attributes", "series"} -- per-series scales (G-03)

    def required_observations(self, n_exog: int = 0, horizon: int = 4) -> int:
        formula = self.n_lags + horizon + 500
        return max(self.context_family_floor, formula) if self.context_family_floor else formula

    def hyperparameter_search_space(self) -> dict:
        # PRD v10 §3.3.1: tunable hyperparameters + valid ranges for this
        # algorithm. Shared here across all 19 deep-learning-family
        # stand-ins (RNN/LSTM/GRU/.../N-HiTS); TCN/WaveNet override this
        # with their own kernel_size/levels space instead (below).
        return {
            "n_lags": {"type": "int", "low": 3, "high": 30},
            "max_iter": {"type": "int", "low": 100, "high": 800},
            "hidden_layer_sizes": {
                "type": "choice",
                "choices": [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)],
            },
        }

    def _make_estimator(self):
        hidden = tuple(self.hyperparameters.get("hidden_layer_sizes", self.default_hidden_layer_sizes))
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=int(self.hyperparameters.get("max_iter", 500)),
            random_state=0,
        )

    # ---- pooled training (G-03) -- stand-in until the real architectures land (G-25..G-29) ---------------------
    def train_pooled(self, batch) -> None:
        """One MLP on lag windows stacked across ALL series, each series divided by its own scale (no FX
        conversion, decision on G-03). Lags only: the per-process exogenous datasets differ between series."""
        Xs, ys = [], []
        for sid in batch.segment_ids:
            y = endogenous_series(batch.segments[sid]) / batch.scales[sid]
            feats = build_lag_features(y, pd.DataFrame(), self.n_lags).dropna()
            Xs.append(feats.drop(columns="y").to_numpy())
            ys.append(feats["y"].to_numpy())
        X, y = np.vstack(Xs), np.concatenate(ys)
        self._feature_cols = [f"lag_{i}" for i in range(1, self.n_lags + 1)]
        self._fit_estimator(X, y)
        self._store_lag_residuals(X, y)
        self._pooled = {"scales": dict(batch.scales), "attributes": dict(batch.attributes), "series": list(batch.segment_ids)}
        self._fitted_model = {"n_lags": self.n_lags, "hyperparameters": self.hyperparameters, "pooled": self._pooled}

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._pooled is None:
            return super().infer(segment_df, horizon)
        if self._model is None:
            raise RuntimeError("load() or train_pooled() must run before infer()")
        from pooling import series_scale
        series = endogenous_series(segment_df)
        sid = str(segment_df["segment_id"].iloc[0])
        scale = self._pooled["scales"].get(sid) or series_scale(series)  # unseen series: scale from its own history
        history = series / scale
        dates = future_dates(series.index.max(), horizon, freq=self.frequency)
        preds = []
        for d in dates:
            x = np.array([history.iloc[-lag] for lag in range(1, self.n_lags + 1)]).reshape(1, -1)
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat * scale)
            history.loc[d] = yhat
        return pd.DataFrame({"date": dates, "forecast": preds})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "feature_cols": self._feature_cols,
                         "hyperparameters": self.hyperparameters, "pooled": self._pooled}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._feature_cols = state["feature_cols"]
        self.hyperparameters = state["hyperparameters"]
        self._pooled = state.get("pooled")
        self.n_lags = int(self.hyperparameters.get("n_lags", self.default_n_lags))
