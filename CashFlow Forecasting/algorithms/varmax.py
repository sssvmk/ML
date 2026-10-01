from __future__ import annotations
from pathlib import Path
import pickle
import warnings
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

from .base import AlgorithmModule, EligibilityResult
from .utils import endogenous_series, exogenous_frame, future_dates
from ._statsmodels_template import stationary_by_adf_and_kpss


class VARMAXModule(AlgorithmModule):
    """
    VARMAX (PRD §3.2 #3). Eligibility: the joint system must be
    stationary (or cointegrated if modeled in levels), requiring at
    least 2 endogenous dimensions -- undefined with only one. This
    harness runs single-segment, so a second dimension is only
    available if an exogenous series exists to press into service as a
    demonstration; a genuine multivariate case (two endogenous series,
    e.g. joint AR/AP modeling) needs the pooled/multivariate design that
    is still an open item (PRD §4.2 #15, §4.3).
    """

    name = "varmax"
    has_eligibility_condition = True

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        p = self.hyperparameters.get("p")
        q = self.hyperparameters.get("q")
        if p is not None and q is not None:
            self.order = (int(p), int(q))
        else:
            self.order = tuple(self.hyperparameters.get("order", (1, 0)))  # (p, q)
        self._results = None
        self._second_col: str | None = None

    def hyperparameter_search_space(self) -> dict:
        return {"p": {"type": "int", "low": 0, "high": 2}, "q": {"type": "int", "low": 0, "high": 2}}

    def _endog_matrix(self, segment_df) -> pd.DataFrame:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        if exog.empty:
            return pd.DataFrame({"series_0": series})
        self._second_col = exog.columns[0]
        return pd.DataFrame({"series_0": series}).join(exog[[self._second_col]], how="inner")

    def required_observations(self, n_exog: int = 0, n_dims: int = 2) -> int:
        p, q = self.order
        return max(100, 10 * (n_dims ** 2) * max(p, 1))  # PRD v10 §3.2 row #3

    def _check_eligibility_impl(self, segment_df: pd.DataFrame) -> EligibilityResult:
        mat = self._endog_matrix(segment_df)
        if mat.shape[1] < 2:
            return EligibilityResult(False, "fewer than 2 endogenous dimensions available (needs pooling/multivariate design, PRD §4.2 #15)")
        min_n = self.required_observations(n_dims=mat.shape[1])
        if len(mat) < min_n:
            return EligibilityResult(False, f"only {len(mat)} observations, need >= {min_n} for {mat.shape[1]} dims, order {self.order}")
        oks, reasons = [], []
        for col in mat.columns:
            ok, reason = stationary_by_adf_and_kpss(mat[col].diff().dropna())
            oks.append(ok)
            reasons.append(f"{col}: {reason}")
        return EligibilityResult(all(oks), "; ".join(reasons))

    def train(self, segment_df: pd.DataFrame) -> None:
        mat = self._endog_matrix(segment_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VARMAX(mat, order=self.order)
            self._results = model.fit(disp=False)
        self._store_statespace_residuals(self._results, burn=max(self.order[0], 1))
        self._fitted_model = {"order": self.order, "aic": float(self._results.aic), "dims": list(mat.columns)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"results": self._results, "order": self.order, "second_col": self._second_col, "hyperparameters": self.hyperparameters}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._results = state["results"]
        self.order = state["order"]
        self._second_col = state["second_col"]
        self.hyperparameters = state["hyperparameters"]

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("load() or train() must run before infer()")
        mat = self._endog_matrix(segment_df)
        dates = future_dates(mat.index.max(), horizon, freq=self.frequency)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = self._results.apply(mat)
            fc = refit.get_forecast(steps=horizon)
        pred = fc.predicted_mean["series_0"].to_numpy()
        return pd.DataFrame({"date": dates, "forecast": pred})
