from __future__ import annotations
from .nf_adapter import NeuralForecastModule, mlp_units

POOL_CHOICES = [[1, 1, 1], [2, 2, 1], [4, 2, 1], [8, 4, 1]]        # per-stack MaxPool1d kernels, coarse -> fine
DOWNSAMPLE_CHOICES = [[1, 1, 1], [2, 1, 1], [4, 2, 1]]              # per-stack theta downsampling (expressiveness ratio)


class NHiTSModule(NeuralForecastModule):
    """
    N-HiTS (PRD §3.2 #38; gap G-38). Route: NeuralForecast `NHITS` behind the adapter (D-6): N-BEATS-style residual
    blocks plus (1) multi-rate input sampling by per-stack MaxPool1d kernels, (2) hierarchical interpolation -- each block
    emits a coarse theta that is interpolated (linear/nearest) up to the horizon, (3) stack-wise multi-resolution.
    Supports historical/future/static exogenous inputs. Required observations: N >= lookback + horizon + 500.
    """

    name = "nhits"
    nf_model_name = "NHITS"
    has_eligibility_condition = False
    forwarded_hp = ("n_pool_kernel_size", "n_freq_downsample", "pooling_mode", "interpolation_mode",
                    "dropout_prob_theta", "activation")

    def _model_kwargs(self) -> dict:
        hp = self.hyperparameters
        kw = super()._model_kwargs()
        n = len(kw.get("n_pool_kernel_size", POOL_CHOICES[0]))
        kw["stack_types"] = ["identity"] * n
        kw["n_blocks"] = [int(hp.get("blocks_per_stack", 1))] * n
        kw["mlp_units"] = mlp_units(hp.get("mlp_width", 256), hp.get("mlp_layers", 2), n)
        return kw

    def hyperparameter_search_space(self) -> dict:
        space = super().hyperparameter_search_space()
        space.update({
            "n_pool_kernel_size": {"type": "choice", "choices": POOL_CHOICES},
            "n_freq_downsample": {"type": "choice", "choices": DOWNSAMPLE_CHOICES},
            "interpolation_mode": {"type": "choice", "choices": ["linear", "nearest"]},
            "blocks_per_stack": {"type": "int", "low": 1, "high": 3},
            "mlp_width": {"type": "choice", "choices": [64, 128, 256, 512]},
            "mlp_layers": {"type": "int", "low": 1, "high": 3},
        })
        return space
