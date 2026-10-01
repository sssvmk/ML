from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class AutoformerModule(NeuralForecastModule):
    """
    Autoformer (PRD §3.2 #34; gap G-34). Route: NeuralForecast `Autoformer` behind the adapter (D-6): a series
    decomposition block (moving-average trend, seasonal = x - trend) inside every encoder/decoder layer (progressive
    decomposition) and Auto-Correlation attention (FFT-based period-lag correlation, top-k lags, time-delay
    aggregation) in place of self-attention; forecast = seasonal projection + accumulated trend. Future-known calendar
    covariates only. Required observations: N >= max(1000, context + horizon + 500).
    """

    name = "autoformer"
    nf_model_name = "Autoformer"
    has_eligibility_condition = False
    context_family_floor = 1000
    supports_hist_exog = False
    forwarded_hp = ("hidden_size", "n_head", "factor", "encoder_layers", "decoder_layers", "conv_hidden_size", "dropout",
                    "decoder_input_size_multiplier", "MovingAvg_window", "activation")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [32, 64, 128]},
        "n_head": {"type": "choice", "choices": [2, 4, 8]},
        "factor": {"type": "int", "low": 1, "high": 5},
        "encoder_layers": {"type": "int", "low": 1, "high": 3},
        "decoder_layers": {"type": "int", "low": 1, "high": 2},
        "MovingAvg_window": {"type": "choice", "choices": [7, 13, 25]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }
