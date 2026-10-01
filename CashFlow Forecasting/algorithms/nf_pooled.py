"""
Pooling-capable NeuralForecast modules (PRD Overview §1; gaps G-03, G-25, G-28, G-29).

ONE model instance is trained on every series of a pooling.PooledBatch (`segment_id` is NeuralForecast's `unique_id`),
conditioned on static attributes derived from the canonical rows (entity, currency, direction; one-hot). Series are not
FX-converted: NeuralForecast scales every input window by its own statistics (decision on G-03).

Deliberately no historical exogenous datasets here: pooled series come from different processes (AR vs AP) whose exogenous
datasets have different names, and the algorithm must not learn source vocabulary (loose coupling). Future-known covariates
are limited to calendar features unless the contract supplies a declared future-known series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .nf_adapter import NeuralForecastModule
from .utils import future_known_datasets, next_period_after
from .base import EligibilityResult


class PooledNeuralForecastModule(NeuralForecastModule):
    pooling_capable = True
    pooled_n_basis = "pooled_total"
    supports_hist_exog = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._cats: dict = {}
        self._attrs: dict = {}
        self._pool_series: list = []

    # single-segment path: one series is not a pool
    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        return EligibilityResult(False, "pooling-capable algorithm: needs a pool of >= 2 series "
                                        "(train it through Orchestrator.full_train_pooled); one segment is not a pool")

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return 1000  # PRD v10 §3.2: Pooled N >= 1,000 (DeepAR/DeepState/DeepVAR); subclasses restate their own row

    # ---- static covariates ----------------------------------------------------------------------------------------
    @staticmethod
    def _col(key: str, value: str) -> str:
        return f"{key}={value}"

    def _static_row(self, sid: str, attrs: dict) -> dict:
        row = {"unique_id": sid}
        for key, values in self._cats.items():
            a = attrs[key]
            if a not in values:
                raise ValueError(f"{key} {a!r} of series {sid!r} was not seen in pooled training (known: {sorted(values)}); "
                                 "the pooled model cannot condition on an unseen category")
            for v in values:
                row[self._col(key, v)] = 1.0 if v == a else 0.0
        return row

    def _static_frame(self, ids: list, attrs: dict) -> pd.DataFrame:
        return pd.DataFrame([self._static_row(i, attrs[i]) for i in ids])

    def _attrs_of(self, sid: str, segment_df: pd.DataFrame) -> dict:
        """Attributes come from the rows supplied now; a known series whose attributes changed is a data inconsistency."""
        now = {"company_code": str(segment_df["company_code"].iloc[0]),
               "currency":     str(segment_df["currency"].iloc[0]),
               "dataset":      str(segment_df.loc[segment_df["series_role"] == "endogenous", "dataset"].iloc[0]
                                    if (segment_df["series_role"] == "endogenous").any() else "")}
        if sid in self._attrs and self._attrs[sid] != now:
            raise ValueError(f"attributes of series {sid!r} changed since pooled training: trained {self._attrs[sid]}, supplied {now}")
        return now

    # ---- pooled training ------------------------------------------------------------------------------------------
    def train_pooled(self, batch) -> None:
        self._cats = batch.categories()
        self._attrs = {sid: dict(a) for sid, a in batch.attributes.items()}
        self._pool_series = list(batch.segment_ids)
        self._static_cols = [self._col(k, v) for k, vals in self._cats.items() for v in vals]
        # future-known covariates, inferred from structure: series of DIFFERENT processes carry differently named datasets, so
        # they are mapped POSITIONALLY (sorted order) to shared channels fk_0..fk_{k-1}; every series must supply the same number
        names_by = {sid: self._fk_names_for(batch.segments[sid]) for sid in batch.segment_ids}
        counts = {len(v) for v in names_by.values()}
        self._fk_positional, self._fk_names, self._fk_note = True, [], None
        if len(counts) == 1 and counts != {0}:
            self._fk_cols = [f"fk_{i}" for i in range(next(iter(counts)))]
        else:
            self._fk_cols, names_by = [], {sid: [] for sid in names_by}
            if counts != {0}:
                self._fk_note = f"future-known series ignored: series carry different numbers of them ({sorted(counts)})"
        frames = [self._frame(batch.segments[sid], series_id=sid, fk_names=names_by[sid]) for sid in batch.segment_ids]
        frame = self._with_calendar(pd.concat(frames, ignore_index=True))
        static = self._static_frame(batch.segment_ids, self._attrs)
        self._exog_cols = []
        self._fit(frame, [], min(len(f) for f in frames), static_df=static)
        self._fitted_model["pooled"] = {"series": self._pool_series, "attributes": self._attrs, "n_series": len(self._pool_series),
                                        "future_known_channels": len(self._fk_cols), "future_known_note": self._fk_note}

    def _store_insample_residuals(self, frame_m: pd.DataFrame) -> None:
        """First-step in-sample residuals of every series, series by series (chronological within a series)."""
        try:
            ins = self._nf.predict_insample(step_size=1)
            col = self._point_column(ins)
            first = ins[ins["ds"] == ins["cutoff"].map(lambda c: next_period_after(c, self.frequency))].sort_values(["unique_id", "ds"])
            self._resid = (first["y"] - first[col]).to_numpy(dtype=float)
        except Exception:
            self._resid = None

    _fk_positional = False
    _fk_note = None

    def _infer_fk_names(self, segment_df: pd.DataFrame) -> list[str]:
        if not self._fk_cols:
            return []
        if not self._fk_positional:                      # trained on one series: its own dataset names are what must be present
            return super()._infer_fk_names(segment_df)
        found = future_known_datasets(segment_df, min_future=self.horizon)
        if len(found) != len(self._fk_cols):
            raise ValueError(f"the pooled model was trained with {len(self._fk_cols)} future-known series per series; these rows "
                             f"supply {len(found)} with >= {self.horizon} future values ({found})")
        return found

    def _predict_extra(self, sid: str, segment_df: pd.DataFrame) -> dict:
        if not self._static_cols:
            return {}   # single-segment use of a pooling-capable model: no static covariates
        return {"static_df": self._static_frame([sid], {sid: self._attrs_of(sid, segment_df)})}

    def _state_extra(self) -> dict:
        return {**super()._state_extra(), "fk_positional": self._fk_positional, "cats": self._cats, "attrs": self._attrs, "static_cols": self._static_cols, "pool_series": self._pool_series}

    def _load_extra(self, st: dict) -> None:
        super()._load_extra(st)
        self._fk_positional = bool(st.get("fk_positional", False))
        self._cats, self._attrs = st.get("cats", {}), st.get("attrs", {})
        self._static_cols, self._pool_series = st.get("static_cols", []), st.get("pool_series", [])

    def train(self, segment_df) -> None:
        raise RuntimeError(f"{self.name} is a pooled model: train it with train_pooled(batch) / Orchestrator.full_train_pooled")
