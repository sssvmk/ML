from __future__ import annotations
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .torch_base import TorchWindowModule
from ._vendor_etsformer import load_etsformer
from .utils import future_dates


class ETSformerNet(nn.Module):
    """Adapts the vendored official ETSformer to (B, L, F) -> (B, h): the target channel's forecast.
    The reference model forecasts every input channel (its level module needs c_out == enc_in), so the exogenous / calendar
    channels are forecast too and only channel 0 is returned."""

    def __init__(self, configs):
        super().__init__()
        self.model = load_etsformer().ETSformer(configs)

    def forward(self, x):
        return self.model(x, None, None, None)[:, :, 0]

    def components(self, x):
        """(level, growth, season) for the target channel, each (B, h) in the scaled space; level is the last level repeated."""
        level, growth, season = self.model(x, None, None, None, decomposed=True)
        return level[:, :, 0].expand(-1, growth.shape[1]), growth[:, :, 0], season[:, :, 0]


class ETSformerModule(TorchWindowModule):
    """
    ETSformer (PRD §3.2 #36; gap G-36). Route (D-12): the OFFICIAL Salesforce implementation, vendored unmodified at a pinned
    commit under vendor/etsformer (BSD-3-Clause notice retained, OSS/licence scan in OSS_SCAN.md), exposed only through this
    adapter. Architecture: multi-head exponential-smoothing attention (learnable smoothing alpha per head, exponentially
    decaying weights computed by FFT convolution) extracts growth; frequency attention keeps the top-K amplitude frequencies
    and extrapolates seasonality into the horizon; a level module with learned smoothing; a decoder with growth damping (learned
    damping factor); forecast = level + growth + seasonality, all three retrievable through `decompose()`.

    Training-time augmentation (jitter / scale / shift of the input window, sigma = `std`) is part of the reference model and is
    kept. No eligibility condition. Required observations: N >= max(1000, context + horizon + 500).
    """

    name = "etsformer"
    has_eligibility_condition = False
    context_family_floor = 1000
    default_max_epochs = 20

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.n_lags = int(hp.get("n_lags", 48))

    def lookback(self) -> int:
        return self.n_lags

    def _configs(self, n_features: int):
        hp = self.hyperparameters
        layers = int(hp.get("e_layers", 2))
        d_model = int(hp.get("d_model", 32))
        n_heads = int(hp.get("n_heads", 4))
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        k = int(max(1, min(int(hp.get("K", 3)), self.n_lags // 2 - 2)))   # cannot keep more frequencies than the window has
        return types.SimpleNamespace(
            seq_len=self.n_lags, label_len=0, pred_len=self.horizon, enc_in=n_features, c_out=n_features,
            d_model=d_model, n_heads=n_heads, e_layers=layers, d_layers=layers,     # reference model requires equal depths
            d_ff=int(hp.get("d_ff", 2 * d_model)), dropout=float(hp.get("dropout", 0.2)),
            activation=str(hp.get("activation", "sigmoid")), output_attention=False, K=k, std=float(hp.get("std", 0.2)))

    def build_network(self, n_features: int):
        return ETSformerNet(self._configs(n_features))

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return max(self.context_family_floor, self.n_lags + int(horizon if horizon is not None else self.horizon) + 500)

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_lags": {"type": "int", "low": 24, "high": 96},
            "d_model": {"type": "choice", "choices": [16, 32, 64]},
            "n_heads": {"type": "choice", "choices": [2, 4, 8]},
            "e_layers": {"type": "int", "low": 1, "high": 3},
            "K": {"type": "int", "low": 1, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "std": {"type": "float", "low": 0.0, "high": 0.3},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }

    # ---- interpretability (G-36 acceptance: components separately returned) ---------------------------------------------
    def decompose(self, segment_df: pd.DataFrame) -> pd.DataFrame:
        """Level, growth and seasonality of the next `h` periods in original units; level + growth + season == forecast."""
        if self._net is None:
            raise RuntimeError("load() or train() must run before decompose()")
        frame, _, M = self._matrix(segment_df, exog_cols=self._exog_cols)
        Xs, _, mu, sd = self._scale_windows(M[-self.lookback():][None], None)
        self._net.eval()
        with torch.no_grad():
            lvl, gro, sea = (t.numpy()[0] for t in self._net.components(torch.tensor(Xs)))
        dates = future_dates(pd.Timestamp(frame["ds"].max()), self.horizon, freq=self.frequency)
        s, m = float(sd[0, 0]), float(mu[0, 0])
        return pd.DataFrame({"date": dates, "level": lvl * s + m, "growth": gro * s, "season": sea * s,
                             "forecast": (lvl + gro + sea) * s + m})
