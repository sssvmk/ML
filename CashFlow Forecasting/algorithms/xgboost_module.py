from __future__ import annotations
from xgboost import XGBRegressor
from ._sklearn_template import SklearnLagModule


class XGBoostModule(SklearnLagModule):
    """XGBoost (PRD §3.2 #15). No eligibility condition."""

    name = "xgboost"
    has_eligibility_condition = False
    default_n_lags = 7

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.n_estimators = int(self.hyperparameters.get("n_estimators", 200))
        self.max_depth = int(self.hyperparameters.get("max_depth", 4))
        self.learning_rate = float(self.hyperparameters.get("learning_rate", 0.1))

    def _make_estimator(self):
        return XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=0, verbosity=0,
        )

    def _fit_estimator(self, X, y) -> None:
        split = self._split_validation_tail(X, y)
        if split is None:
            self._model = self._make_estimator()
            self._model.fit(X, y)
            self._early_stopping = {"used": False, "reason": "window too small for a validation tail"}
            return
        Xt, yt, Xv, yv = split
        est = XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            random_state=0, verbosity=0, early_stopping_rounds=self._es_rounds,
        )
        est.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        self._model = est
        self._early_stopping = {"used": True, "best_iteration": int(est.best_iteration),
                                "n_estimators": self.n_estimators, "validation_rows": int(len(Xv))}

    def required_observations(self, n_exog: int = 0) -> int:
        return max(200, 20 * (self.n_lags + n_exog))  # PRD v10 §3.2 row #15

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_lags": {"type": "int", "low": 3, "high": 21},
        }
