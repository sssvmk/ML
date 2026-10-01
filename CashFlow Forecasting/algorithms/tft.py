from __future__ import annotations
import re
import pandas as pd

from ._futr_required import FutureKnownRequiredModule
from ._vendor import ensure_vendored_neuralforecast

QUANTILES = [0.1, 0.5, 0.9]


class TFTModule(FutureKnownRequiredModule):
    """
    Temporal Fusion Transformer (PRD §3.2 #28; gap G-28). Route: NeuralForecast `TFT` behind the adapter (D-6):
    static covariate encoders (entity / currency / direction, one-hot, when trained on a pool); Variable Selection Networks
    (Gated Residual Networks + softmax) for the static, past-observed and FUTURE-KNOWN inputs; an LSTM encoder over the past and
    a decoder over the future-known covariates; gated skip connections (GLU + LayerNorm) with static enrichment; interpretable
    multi-head attention with a causal mask; a position-wise GRN; quantile outputs (P10/P50/P90) trained on the pinball loss,
    direct multi-horizon. P50 is the point forecast used for the six metrics.

    Future-known covariates come from data (structure-inferred, see _futr_required.py), not from carry-forward. Eligibility is
    derived: without future-known covariate series covering the horizon the model is ineligible. Variable-selection weights and
    attention weights are retrievable after a forecast (`variable_importance`, `attention`). Quantile outputs are made monotone
    (sorted per row) when a head crosses, and the number of crossed rows is reported.
    """

    name = "tft"
    nf_model_name = "TFT"
    valid_loss_label = "MQLoss(0.1/0.5/0.9), original target units"
    forwarded_hp = ("hidden_size", "n_head", "attn_dropout", "dropout", "n_rnn_layers", "rnn_type", "grn_activation")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [16, 32, 64, 128]},
        "n_head": {"type": "choice", "choices": [1, 2, 4]},
        "n_rnn_layers": {"type": "int", "low": 1, "high": 2},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "attn_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }

    def _model_extra_kwargs(self) -> dict:
        return {"loss": ensure_vendored_neuralforecast().losses.pytorch.MQLoss(quantiles=QUANTILES)}

    # ---- outputs ---------------------------------------------------------------------------------------------------
    def _predict_raw(self, segment_df: pd.DataFrame) -> pd.DataFrame:
        names = self._infer_fk_names(segment_df)
        frame = self._frame(segment_df, exog_cols=self._exog_cols, fk_names=names)
        fkw = self._fk_wide(segment_df, names) if names else None
        sid = frame["unique_id"].iloc[0]
        return self._nf.predict(df=self._with_calendar(frame), futr_df=self._futr(sid, frame["ds"].max(), self.horizon, fkw),
                                **self._predict_extra(sid, segment_df))

    def infer_quantiles(self, segment_df: pd.DataFrame) -> pd.DataFrame:
        """Date + q10 / q50 / q90 for the first `horizon` steps, sorted per row so the quantiles are monotone."""
        import numpy as np
        out = self._predict_raw(segment_df)
        cols = {}
        for c in out.columns:
            if c.endswith("-median"):
                cols[c] = 0.5
            else:
                m = re.search(r"-(lo|hi)-([0-9.]+)$", c)    # NeuralForecast names symmetric quantile pairs by level: 80 -> 0.1 / 0.9
                if m:
                    tail = (100.0 - float(m.group(2))) / 200.0
                    cols[c] = round(tail if m.group(1) == "lo" else 1.0 - tail, 4)
        want = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
        pick = {c: want[v] for c, v in cols.items() if v in want}
        if len(pick) != 3:
            raise RuntimeError(f"expected the 0.1/0.5/0.9 quantile columns, found {list(out.columns)}")
        q = out[list(pick)].rename(columns=pick)[["q10", "q50", "q90"]]
        vals = q.to_numpy()
        crossed = int((np.diff(vals, axis=1) < 0).any(axis=1).sum())
        res = pd.DataFrame(np.sort(vals, axis=1), columns=["q10", "q50", "q90"])
        res.insert(0, "date", out["ds"].to_numpy())
        res.attrs["crossed_rows_before_sorting"] = crossed
        return res

    def variable_importance(self, segment_df: pd.DataFrame) -> dict:
        """Variable-selection weights (static / history / future) after a forecast on `segment_df`."""
        self._predict_raw(segment_df)
        return self._nf.models[0].feature_importances()

    def attention(self, segment_df: pd.DataFrame):
        """Mean interpretable-attention weights after a forecast on `segment_df`."""
        self._predict_raw(segment_df)
        return self._nf.models[0].attention_weights()
