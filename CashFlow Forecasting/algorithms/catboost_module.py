from __future__ import annotations
from catboost import CatBoostRegressor
from ._sklearn_template import SklearnLagModule


class CatBoostModule(SklearnLagModule):
    """CatBoost (PRD §3.2 #17). No eligibility condition."""

    name = "catboost"
    has_eligibility_condition = False
    default_n_lags = 7

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.iterations = int(self.hyperparameters.get("iterations", 200))
        self.depth = int(self.hyperparameters.get("depth", 4))
        self.learning_rate = float(self.hyperparameters.get("learning_rate", 0.1))

    def _make_estimator(self):
        return CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_state=0, verbose=False,
        )

    def _fit_estimator(self, X, y) -> None:
        split = self._split_validation_tail(X, y)
        if split is None:
            self._model = self._make_estimator()
            self._model.fit(X, y)
            self._early_stopping = {"used": False, "reason": "window too small for a validation tail"}
            return
        Xt, yt, Xv, yv = split
        est = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            random_state=0, verbose=False, early_stopping_rounds=self._es_rounds, use_best_model=True,
        )
        est.fit(Xt, yt, eval_set=(Xv, yv))
        self._model = est
        self._early_stopping = {"used": True, "best_iteration": int(est.get_best_iteration()),
                                "n_estimators": self.iterations, "validation_rows": int(len(Xv))}

    def required_observations(self, n_exog: int = 0) -> int:
        return max(200, 20 * (self.n_lags + n_exog))  # PRD v10 §3.2 row #17

    def hyperparameter_search_space(self) -> dict:
        return {
            "iterations": {"type": "int", "low": 50, "high": 500},
            "depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_lags": {"type": "int", "low": 3, "high": 21},
        }
