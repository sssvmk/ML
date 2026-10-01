from __future__ import annotations
import pandas as pd

from .nf_pooled import PooledNeuralForecastModule
from ._vendor import ensure_vendored_neuralforecast
from .utils import future_dates


class DeepARModule(PooledNeuralForecastModule):
    """
    DeepAR (PRD §3.2 #25; gap G-25). Route: NeuralForecast `DeepAR` behind the adapter (D-6): a global autoregressive
    LSTM trained on ALL pooled series at once (static entity / currency / direction attributes as conditioning), whose
    output layer projects the hidden state to distribution parameters (Student-t by default, or Normal) and is trained on
    the negative log-likelihood. Inference draws sample paths; the median is the point forecast used for the six
    metrics and the quantiles are kept (`infer_quantiles`).

    Eligibility (derived from the pool actually supplied, never a manual flag): >= 2 series and pooled N >= 1,000.
    Quantiles are available for the first `h` steps; beyond `h` the point forecast is rolled forward in chunks.
    """

    name = "deepar"
    nf_model_name = "DeepAR"
    has_eligibility_condition = True
    default_n_lags = 28
    valid_loss_label = "MQLoss(80/90), original target units"
    forwarded_hp = ("lstm_n_layers", "lstm_hidden_size", "lstm_dropout", "decoder_hidden_layers", "decoder_hidden_size",
                    "trajectory_samples")
    model_search_space = {
        "lstm_n_layers": {"type": "int", "low": 1, "high": 3},
        "lstm_hidden_size": {"type": "choice", "choices": [16, 32, 64, 128]},
        "lstm_dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "distribution": {"type": "choice", "choices": ["StudentT", "Normal"]},
    }

    def _model_kwargs(self) -> dict:
        kw = super()._model_kwargs()
        losses = ensure_vendored_neuralforecast().losses.pytorch
        dist = self.hyperparameters.get("distribution", "StudentT")
        kw["loss"] = losses.DistributionLoss(distribution=dist, level=[80, 90])   # negative log-likelihood objective
        kw["valid_loss"] = losses.MQLoss(level=[80, 90])
        if kw.get("lstm_n_layers") == 1 and kw.get("lstm_dropout"):
            kw["lstm_dropout"] = 0.0
        return kw

    def infer_quantiles(self, segment_df: pd.DataFrame) -> pd.DataFrame:
        """Forecast distribution summary for the first `h` steps: date + median, mean and the 80/90 % interval bounds."""
        frame = self._frame(segment_df)
        sid = frame["unique_id"].iloc[0]
        out = self._nf.predict(df=self._with_calendar(frame), futr_df=self._futr(sid, frame["ds"].max(), self.horizon),
                               **self._predict_extra(sid, segment_df))
        out = out.drop(columns=["unique_id"]).rename(columns={"ds": "date"})
        return out.rename(columns={c: c.replace("DeepAR", "").lstrip("-") or "mean" for c in out.columns if c != "date"})
