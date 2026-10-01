from __future__ import annotations
from sklearn.linear_model import ElasticNet
from ._sklearn_template import SklearnLagModule


class ElasticNetModule(SklearnLagModule):
    """ElasticNet (PRD §3.2 #12). No eligibility condition -- same as
    Ridge/Lasso, regularization guarantees a solvable fit."""

    name = "elastic_net"
    has_eligibility_condition = False
    default_n_lags = 3

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.alpha = float(self.hyperparameters.get("alpha", 0.1))
        self.l1_ratio = float(self.hyperparameters.get("l1_ratio", 0.5))

    def _make_estimator(self):
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)

    def required_observations(self, n_exog: int = 0) -> int:
        return max(50, self.n_lags + n_exog)  # PRD v10 §3.2 row #12

    def hyperparameter_search_space(self) -> dict:
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0},
            "n_lags": {"type": "int", "low": 1, "high": 14},
        }
