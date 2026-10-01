from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class LSTMModule(NeuralForecastModule):
    """
    LSTM (PRD §3.2 #21; gap G-21). Route: NeuralForecast `LSTM` behind the adapter (D-6). Recurrent core is
    torch.nn.LSTM (input/forget/output gates, cell state), unidirectional/causal, MLP decoder for the h outputs.
    No eligibility condition. Required observations: N >= lookback + horizon + 500.
    """

    name = "lstm"
    nf_model_name = "LSTM"
    has_eligibility_condition = False
    default_gradient_clip = 1.0
    forwarded_hp = ("encoder_n_layers", "encoder_hidden_size", "encoder_dropout", "decoder_hidden_size",
                    "decoder_layers", "context_size")
    model_search_space = {
        "encoder_hidden_size": {"type": "choice", "choices": [16, 32, 64, 128]},
        "encoder_n_layers": {"type": "int", "low": 1, "high": 3},
        "encoder_dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "decoder_hidden_size": {"type": "choice", "choices": [32, 64, 128]},
    }
