from __future__ import annotations

from ._futr_required import FutureKnownRequiredModule
from ._vendor import ensure_vendored_neuralforecast


class TiDEModule(FutureKnownRequiredModule):
    """
    TiDE (PRD §3.2 #29; gap G-29). Route: NeuralForecast `TiDE` behind the adapter (D-6): a feature projection (residual MLP
    block) reducing each timestep's future-known covariates to a low dimension over lookback + horizon; a dense encoder of
    stacked residual blocks on the flattened [lookback target, projected covariates, static attributes]; a dense decoder producing
    a (horizon x p) matrix; a per-horizon-step TEMPORAL DECODER on [decoded vector_t, projected future covariates_t]; a global
    linear residual from the lookback to the horizon; MSE loss. Same derived, structure-based future-known requirement as TFT.
    """

    name = "tide"
    nf_model_name = "TiDE"
    valid_loss_label = "MSE, original target units"
    forwarded_hp = ("hidden_size", "decoder_output_dim", "temporal_decoder_dim", "dropout", "layernorm",
                    "num_encoder_layers", "num_decoder_layers", "temporal_width")
    model_search_space = {
        "hidden_size": {"type": "choice", "choices": [32, 64, 128, 256]},
        "decoder_output_dim": {"type": "choice", "choices": [8, 16, 32]},
        "temporal_decoder_dim": {"type": "choice", "choices": [16, 32, 64, 128]},
        "num_encoder_layers": {"type": "int", "low": 1, "high": 3},
        "num_decoder_layers": {"type": "int", "low": 1, "high": 3},
        "temporal_width": {"type": "choice", "choices": [2, 4, 8]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
    }

    def _model_extra_kwargs(self) -> dict:
        return {"loss": ensure_vendored_neuralforecast().losses.pytorch.MSE()}

    @staticmethod
    def _fix_final_layernorm(model) -> None:
        """
        Upstream defect worked around here: NeuralForecast's TiDE builds every residual block with LayerNorm(output_dim), including
        the TEMPORAL DECODER whose output has ONE value per horizon step for a point forecast. LayerNorm over a single value is
        identically 0 (x - mean(x) = 0), so that block always returns its bias, the dense encoder/decoder branch and the future
        covariates are disconnected, and TiDE degenerates into its linear skip path. Only that final normalisation is switched
        off; LayerNorm stays in the encoder/decoder blocks as the spec requires. The (now unused) norm parameters are kept so
        saved state dicts stay loadable. The vendored library is not modified.
        """
        td = model.temporal_decoder
        if getattr(td, "layernorm", False) and td.lin2.out_features == 1:
            td.layernorm = False

    def _build_model(self, exog_cols, early_stopping, callbacks):
        model = super()._build_model(exog_cols, early_stopping, callbacks)
        self._fix_final_layernorm(model)
        return model

    def load(self, path) -> None:
        super().load(path)            # NeuralForecast rebuilds the model from its saved config: re-apply the fix
        for m in self._nf.models:
            self._fix_final_layernorm(m)
