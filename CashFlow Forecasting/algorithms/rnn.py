from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class RNNModule(NeuralForecastModule):
    """
    RNN (PRD §3.2 #20; gap G-20). Route: NeuralForecast `RNN` behind the adapter (D-6). The recurrent core is
    torch.nn.RNN (Elman) over the (batch, lookback, features) window, followed by an MLP decoder producing all
    h outputs directly. No eligibility condition -- trains on any signal, including noise, without mechanical failure.
    Required observations: N >= lookback + horizon + 500 (base adapter).
    """

    name = "rnn"
    nf_model_name = "RNN"
    has_eligibility_condition = False
    default_gradient_clip = 1.0  # gradient clipping for BPTT (C-5)
    forwarded_hp = ("encoder_n_layers", "encoder_hidden_size", "encoder_activation", "encoder_dropout",
                    "decoder_hidden_size", "decoder_layers", "context_size")
    model_search_space = {
        "encoder_hidden_size": {"type": "choice", "choices": [16, 32, 64, 128]},
        "encoder_n_layers": {"type": "int", "low": 1, "high": 3},
        "encoder_dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "decoder_hidden_size": {"type": "choice", "choices": [32, 64, 128]},
    }
