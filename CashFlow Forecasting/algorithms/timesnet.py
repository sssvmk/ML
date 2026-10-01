from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class TimesNetModule(NeuralForecastModule):
    """
    TimesNet (PRD §3.2 #31; gap G-31). Route: NeuralForecast `TimesNet` behind the adapter (D-6): each TimesBlock takes
    the rFFT of the sequence, picks the top-k amplitude frequencies -> periods, folds the 1-D series into 2-D
    (period x cycles), applies an Inception-style 2-D convolution block, unfolds and aggregates the k branches with
    amplitude-softmax weights, with residual connections. Uses future-known calendar covariates only (library limitation:
    no historical exogenous). Required observations: N >= max(1000, context + horizon + 500).
    """

    name = "timesnet"
    nf_model_name = "TimesNet"
    has_eligibility_condition = False
    context_family_floor = 1000
    supports_hist_exog = False
    forwarded_hp = ("hidden_size", "conv_hidden_size", "top_k", "num_kernels", "encoder_layers", "dropout")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [16, 32, 64]},
        "conv_hidden_size": {"type": "choice", "choices": [16, 32, 64]},
        "top_k": {"type": "int", "low": 2, "high": 5},
        "num_kernels": {"type": "int", "low": 2, "high": 6},
        "encoder_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }
