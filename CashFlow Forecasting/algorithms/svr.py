from __future__ import annotations
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import EligibilityResult
from ._sklearn_template import SklearnLagModule, build_lag_features
from .utils import endogenous_series, exogenous_frame


class SVRModule(SklearnLagModule):
    """
    SVR (PRD §3.2 #18). Eligibility: feature scales must be comparable,
    since kernel distance is otherwise scale-dominated. Checked BEFORE
    the internal StandardScaler is applied (the scaler fixes the
    symptom; the eligibility check still verifies the raw features
    aren't wildly mismatched, since scaling one absurd outlier column
    still distorts distances less predictably than genuinely comparable
    inputs).
    """

    name = "svr"
    has_eligibility_condition = False  # G-15: PRD row #18 says 'None'; the 100-observation floor still gates it
    default_n_lags = 5
    max_scale_ratio = 1e4

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.C = float(self.hyperparameters.get("C", 1.0))
        self.epsilon = float(self.hyperparameters.get("epsilon", 0.1))

    def _make_estimator(self):
        return make_pipeline(StandardScaler(), SVR(C=self.C, epsilon=self.epsilon))

    def required_observations(self, n_exog: int = 0) -> int:
        return 100  # PRD v10 §3.2 row #18: 100 <= N <= compute_ceiling

    # ---- compute_ceiling (G-11) ------------------------------------------------
    # PRD row #18's upper bound. Both settings are read from hyperparameters
    # (config.json) and have NO built-in default: with compute_ceiling unset the
    # bound is not enforced. The behaviour above the ceiling is a policy choice
    # ("limit_window" = train on the most recent `compute_ceiling` observations;
    # "ineligible" = refuse to run), so setting a ceiling without choosing the
    # action is a configuration error rather than a silent guess.
    def _ceiling(self):
        ceiling = self.hyperparameters.get("compute_ceiling")
        if ceiling is None:
            return None, None
        action = self.hyperparameters.get("ceiling_action")
        if action not in ("limit_window", "ineligible"):
            raise ValueError("SVR: compute_ceiling is set, so ceiling_action must be 'limit_window' or 'ineligible'")
        return int(ceiling), action

    def check_eligibility(self, segment_df) -> EligibilityResult:
        ceiling, action = self._ceiling()
        if ceiling is not None and action == "ineligible":
            n = len(endogenous_series(segment_df))
            if n > ceiling:
                return EligibilityResult(False, f"{n} observations exceed compute_ceiling {ceiling}")
        return super().check_eligibility(segment_df)

    def train(self, segment_df) -> None:
        ceiling, action = self._ceiling()
        if ceiling is not None and action == "limit_window":
            dates = sorted(segment_df["date"].unique())
            if len(dates) > ceiling:
                keep = set(dates[-ceiling:])
                segment_df = segment_df[segment_df["date"].isin(keep) | (segment_df["series_role"] == "exogenous")]
        super().train(segment_df)

    def hyperparameter_search_space(self) -> dict:
        return {
            "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
            "epsilon": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
            "n_lags": {"type": "int", "low": 3, "high": 21},
        }

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = build_lag_features(series, exog, self.n_lags).dropna()
        if feats.empty:
            return EligibilityResult(False, "no complete rows after lagging")
        X = feats.drop(columns=["y"]).to_numpy()
        stds = X.std(axis=0)
        stds = stds[stds > 0]
        if len(stds) == 0:
            return EligibilityResult(False, "all features constant")
        ratio = stds.max() / stds.min()
        if ratio > self.max_scale_ratio:
            return EligibilityResult(False, f"feature scale ratio {ratio:.1e} exceeds {self.max_scale_ratio:.0e}")
        return EligibilityResult(True, f"feature scale ratio {ratio:.1e} within {self.max_scale_ratio:.0e}")
