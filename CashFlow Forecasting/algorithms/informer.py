from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class InformerModule(NeuralForecastModule):
    """
    Informer (PRD §3.2 #33; gap G-33). Route: NeuralForecast `Informer` behind the adapter (D-6): ProbSparse
    self-attention (sampled keys, top-u queries by the max-minus-mean sparsity measure), self-attention distilling
    between encoder layers (Conv1d + pooling), and a generative-style decoder producing all h outputs in one pass from
    `label_len` start tokens (decoder_input_size_multiplier x lookback). Future-known calendar covariates only (no
    historical exogenous). Required observations: N >= max(1000, context + horizon + 500).
    """

    name = "informer"
    nf_model_name = "Informer"
    has_eligibility_condition = False
    context_family_floor = 1000
    supports_hist_exog = False
    forwarded_hp = ("hidden_size", "n_head", "factor", "encoder_layers", "decoder_layers", "conv_hidden_size", "dropout",
                    "distil", "decoder_input_size_multiplier", "activation")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [32, 64, 128]},
        "n_head": {"type": "choice", "choices": [2, 4, 8]},
        "factor": {"type": "int", "low": 1, "high": 5},
        "encoder_layers": {"type": "int", "low": 1, "high": 3},
        "decoder_layers": {"type": "int", "low": 1, "high": 2},
        "decoder_input_size_multiplier": {"type": "float", "low": 0.25, "high": 1.0},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }
