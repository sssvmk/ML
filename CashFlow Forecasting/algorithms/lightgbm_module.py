from __future__ import annotations
import lightgbm as lgb
from lightgbm import LGBMRegressor
from ._sklearn_template import SklearnLagModule


class LightGBMModule(SklearnLagModule):
    """LightGBM (PRD §3.2 #16). No eligibility condition."""

    name = "lightgbm"
    has_eligibility_condition = False
    default_n_lags = 7

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.n_estimators = int(self.hyperparameters.get("n_estimators", 200))
        self.max_depth = int(self.hyperparameters.get("max_depth", -1))
        self.learning_rate = float(self.hyperparameters.get("learning_rate", 0.1))

    def _make_estimator(self):
        return LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=0, verbosity=-1, min_child_samples=5,
        )

    def _fit_estimator(self, X, y) -> None:
        split = self._split_validation_tail(X, y)
        self._model = self._make_estimator()
        if split is None:
            self._model.fit(X, y)
            self._early_stopping = {"used": False, "reason": "window too small for a validation tail"}
            return
        Xt, yt, Xv, yv = split
        self._model.fit(Xt, yt, eval_X=Xv, eval_y=yv,
                        callbacks=[lgb.early_stopping(self._es_rounds, verbose=False)])
        self._early_stopping = {"used": True, "best_iteration": int(self._model.best_iteration_ or self.n_estimators),
                                "n_estimators": self.n_estimators, "validation_rows": int(len(Xv))}

    def required_observations(self, n_exog: int = 0) -> int:
        return max(500, 20 * (self.n_lags + n_exog))  # PRD v10 §3.2 row #16

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_lags": {"type": "int", "low": 3, "high": 21},
        }
