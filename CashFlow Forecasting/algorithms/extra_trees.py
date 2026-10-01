from __future__ import annotations
from sklearn.ensemble import ExtraTreesRegressor
from ._sklearn_template import SklearnLagModule


class ExtraTreesModule(SklearnLagModule):
    """Extra Trees (PRD §3.2 #14). No eligibility condition --
    nonparametric, no data-shape requirement."""

    name = "extra_trees"
    has_eligibility_condition = False
    default_n_lags = 5

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.n_estimators = int(self.hyperparameters.get("n_estimators", 200))
        self.max_depth = self.hyperparameters.get("max_depth", 5)

    def _make_estimator(self):
        return ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=0)

    def required_observations(self, n_exog: int = 0) -> int:
        depth = self.max_depth or 10
        return 5 * (2 ** int(depth))  # PRD v10 §3.2 row #14

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "n_lags": {"type": "int", "low": 2, "high": 21},
        }
