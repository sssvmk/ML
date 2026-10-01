from __future__ import annotations
from sklearn.linear_model import Lasso
from ._sklearn_template import SklearnLagModule


class LassoModule(SklearnLagModule):
    """Lasso (PRD §3.2 #11). No eligibility condition -- solvable even
    when the feature count exceeds the observation count."""

    name = "lasso"
    has_eligibility_condition = False
    default_n_lags = 3

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.alpha = float(self.hyperparameters.get("alpha", 0.1))

    def _make_estimator(self):
        return Lasso(alpha=self.alpha, max_iter=5000)

    def required_observations(self, n_exog: int = 0) -> int:
        return max(50, int(0.5 * (self.n_lags + n_exog)))  # PRD v10 §3.2 row #11

    def hyperparameter_search_space(self) -> dict:
        return {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True}, "n_lags": {"type": "int", "low": 1, "high": 14}}
