from __future__ import annotations
import pandas as pd
import numpy as np

from .nf_adapter import NeuralForecastModule
from .utils import endogenous_series, exogenous_frame, future_dates, next_period_after

TARGET_ID = "__target__"


class ITransformerModule(NeuralForecastModule):
    """
    iTransformer (PRD §3.2 #32; gap G-32). Route: NeuralForecast `iTransformer` behind the adapter (D-6). The series
    are inverted: each VARIATE's whole lookback becomes one token (a shared Linear(L -> d_model) embedding), self-attention
    runs across the variate tokens (no positional encoding), a per-token feed-forward follows, and a Linear(d_model -> H)
    projects each token to its forecast; series normalisation on. The endogenous series is one token and EVERY exogenous
    series is an additional token, so adding an exogenous series adds a token. All variates are forecast jointly, which
    is also what lets the module roll beyond the trained horizon without carrying exogenous values forward.
    With no exogenous series it degenerates to a single token (works, but is not the intended use).
    No eligibility condition. Required observations: N >= max(1000, context + horizon + 500).
    """

    name = "itransformer"
    nf_model_name = "iTransformer"
    has_eligibility_condition = False
    context_family_floor = 1000
    supports_hist_exog = False   # exogenous series are variate tokens, not per-timestep features
    supports_futr_exog = False
    forwarded_hp = ("hidden_size", "n_heads", "e_layers", "d_layers", "d_ff", "factor", "dropout", "use_norm")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [32, 64, 128]},
        "n_heads": {"type": "choice", "choices": [2, 4, 8]},
        "e_layers": {"type": "int", "low": 1, "high": 3},
        "d_ff": {"type": "choice", "choices": [64, 128, 256]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }

    # ---- variates as separate series ---------------------------------------------------------------------------
    def _variates(self, segment_df: pd.DataFrame, exog_cols: list[str] | None = None) -> pd.DataFrame:
        y = endogenous_series(segment_df)
        idx = pd.date_range(y.index.min(), y.index.max(), freq=self.frequency)
        parts = [pd.DataFrame({"unique_id": TARGET_ID, "ds": idx, "y": self._fill(y, idx, "endogenous series").to_numpy(dtype=float)})]
        ex = exogenous_frame(segment_df) if bool(self.hyperparameters.get("use_exog", True)) else pd.DataFrame()
        cols = exog_cols if exog_cols is not None else list(ex.columns)
        for c in cols:
            if c not in ex.columns:
                raise ValueError(f"exogenous dataset {c!r} was a variate in training but is absent from the supplied rows")
            parts.append(pd.DataFrame({"unique_id": c, "ds": idx, "y": self._fill(ex[c].astype(float), idx, f"exogenous {c}").to_numpy()}))
        out = pd.concat(parts, ignore_index=True)
        if out["y"].isna().any():
            raise ValueError("variate series contain gaps after alignment (see gap_policy)")
        return out

    def _training_frame(self, segment_df: pd.DataFrame):
        long = self._variates(segment_df)
        cols = [u for u in long["unique_id"].unique() if u != TARGET_ID]
        return long, cols, int((long["unique_id"] == TARGET_ID).sum())

    def _model_kwargs(self) -> dict:
        kw = super()._model_kwargs()
        kw["n_series"] = 1 + len(self._exog_cols)
        return kw

    @property
    def n_variates(self) -> int:
        return 1 + len(self._exog_cols)

    def _store_insample_residuals(self, frame_m) -> None:
        try:
            ins = self._nf.predict_insample(step_size=1)
            ins = ins[ins["unique_id"] == TARGET_ID]
            col = self._point_column(ins)
            first = ins[ins["ds"] == ins["cutoff"].map(lambda c: next_period_after(c, self.frequency))].sort_values("ds")
            self._resid = (first["y"] - first[col]).to_numpy(dtype=float)
        except Exception:
            self._resid = None

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("load() or train() must run before infer()")
        hist = self._variates(segment_df, exog_cols=self._exog_cols)
        last0 = hist["ds"].max()
        target_dates = future_dates(last0, horizon, freq=self.frequency)
        preds: list[float] = []
        while len(preds) < horizon:
            last = hist["ds"].max()
            out = self._nf.predict(df=hist)
            col = self._point_column(out)
            new = out[["unique_id", "ds", col]].rename(columns={col: "y"})
            tgt = new[new["unique_id"] == TARGET_ID].sort_values("ds")["y"].to_numpy(dtype=float)[: self.horizon]
            preds.extend(float(v) for v in tgt)
            if len(preds) >= horizon:
                break
            hist = pd.concat([hist, new], ignore_index=True)     # every variate is extended with its OWN forecast
        return pd.DataFrame({"date": target_dates, "forecast": preds[:horizon]})
