from __future__ import annotations
from .nf_adapter import NeuralForecastModule


class FEDformerModule(NeuralForecastModule):
    """
    FEDformer (PRD §3.2 #35; gap G-35). Route: NeuralForecast `FEDformer` behind the adapter (D-6): decomposition
    encoder-decoder whose attention is replaced by Frequency Enhanced Blocks (Fourier or wavelet variant): the sequence
    is transformed to the frequency domain, M selected modes are multiplied by learnable complex weights and inverse
    transformed; Frequency Enhanced (cross) Attention works on the same selected modes. `modes` is clipped to what the
    sequence length can carry. Future-known calendar covariates only. Required observations: N >= max(1000, context +
    horizon + 500).
    """

    name = "fedformer"
    nf_model_name = "FEDformer"
    has_eligibility_condition = False
    context_family_floor = 1000
    supports_hist_exog = False
    forwarded_hp = ("version", "modes", "mode_select", "hidden_size", "n_head", "encoder_layers", "decoder_layers",
                    "conv_hidden_size", "dropout", "decoder_input_size_multiplier", "MovingAvg_window", "activation")
    model_search_space = {
        "modes": {"type": "int", "low": 2, "high": 32},
        "mode_select": {"type": "choice", "choices": ["random", "low"]},
        "hidden_size": {"type": "choice", "choices": [32, 64, 128]},
        "n_head": {"type": "choice", "choices": [2, 4, 8]},
        "encoder_layers": {"type": "int", "low": 1, "high": 3},
        "decoder_layers": {"type": "int", "low": 1, "high": 2},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }

    def _model_kwargs(self) -> dict:
        kw = super()._model_kwargs()
        if "modes" in kw:
            kw["modes"] = int(max(1, min(int(kw["modes"]), self.n_lags // 2)))   # cannot keep more modes than the sequence has
        return kw
