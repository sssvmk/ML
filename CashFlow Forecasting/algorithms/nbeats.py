from __future__ import annotations
from .nf_adapter import NeuralForecastModule, mlp_units

STACK_CHOICES = [["identity", "identity", "identity"], ["trend", "seasonality", "identity"], ["trend", "seasonality"]]


class NBEATSModule(NeuralForecastModule):
    """
    N-BEATS (PRD §3.2 #37; gap G-37). Route: NeuralForecast `NBEATS` behind the adapter (D-6): stacks of blocks,
    each block a fully connected ReLU stack feeding backcast/forecast heads through a basis (generic `identity`, or the
    interpretable `trend` polynomial / `seasonality` Fourier bases); double residual stacking (each block's backcast is
    subtracted from the next block's input; the forecast is the sum of block forecasts); direct multi-horizon output.
    Lookback = k*H with k in 2..7 (search space). This model takes no exogenous inputs (library limitation), so exogenous
    datasets are not used. Required observations: N >= lookback + horizon + 500.

    Hyperparameters: stack_types (list), blocks_per_stack, mlp_width, mlp_layers (paper: 4), n_lags (lookback).
    """

    name = "nbeats"
    nf_model_name = "NBEATS"
    has_eligibility_condition = False
    supports_hist_exog = False
    supports_futr_exog = False
    forwarded_hp = ("n_harmonics", "n_polynomials", "basis", "dropout_prob_theta", "activation", "shared_weights")

    def _model_kwargs(self) -> dict:
        hp = self.hyperparameters
        kw = super()._model_kwargs()
        stacks = list(hp.get("stack_types", ["identity", "trend", "seasonality"]))
        kw["stack_types"] = stacks
        kw["n_blocks"] = [int(hp.get("blocks_per_stack", 1))] * len(stacks)
        kw["mlp_units"] = mlp_units(hp.get("mlp_width", 256), hp.get("mlp_layers", 4), len(stacks))
        return kw

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_lags": {"type": "int", "low": 2 * self.horizon, "high": 7 * self.horizon},   # k*H, k = 2..7
            "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
            "stack_types": {"type": "choice", "choices": STACK_CHOICES},
            "blocks_per_stack": {"type": "int", "low": 1, "high": 3},
            "mlp_width": {"type": "choice", "choices": [64, 128, 256, 512]},
            "mlp_layers": {"type": "int", "low": 2, "high": 4},
        }
