from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .base import EligibilityResult
from .torch_base import TorchWindowModule
from .utils import endogenous_series


def receptive_field(kernel_size: int, levels: int) -> int:
    """RF of the network below: two dilated causal convs per block, dilation 2**i  ->  1 + 2*(k-1)*(2**levels - 1)."""
    return 1 + 2 * (int(kernel_size) - 1) * (2 ** int(levels) - 1)


class CausalConv1d(nn.Module):
    """Dilated causal convolution: left-pad by (k-1)*d so output[t] never sees input[t+1] (weight-normalised)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    """[dilated causal conv -> ReLU -> dropout] x 2 + residual (1x1 conv when the channel count changes)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.conv1(x)))
        h = self.drop(F.relu(self.conv2(h)))
        return F.relu(h + self.down(x))


class TCNNet(nn.Module):
    """Stack of `levels` temporal blocks with dilation 2**i; last timestep -> Linear -> horizon (direct multi-step)."""

    def __init__(self, n_features: int, horizon: int, kernel_size: int, levels: int, channels: int, dropout: float):
        super().__init__()
        blocks, c_in = [], n_features
        for i in range(levels):
            blocks.append(TemporalBlock(c_in, channels, kernel_size, 2 ** i, dropout))
            c_in = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):                       # x: (B, L, F)
        h = x.transpose(1, 2)                   # -> (B, F, L)
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, :, -1])           # last timestep -> horizon outputs


class TCNModule(TorchWindowModule):
    """
    TCN (PRD §3.2 #23; gap G-23). Custom PyTorch per the matrix spec (decision: write it to spec rather than use
    NeuralForecast's variant): residual blocks of two weight-normalised dilated causal Conv1d (ReLU, dropout), dilation
    2**i for i = 0..levels-1, 1x1 residual conv when channels change, last timestep -> Linear -> horizon.

    Eligibility (necessary): history must cover the receptive field, computed from the real network structure,
    RF = 1 + 2*(k-1)*(2**levels - 1). Required observations: N >= RF + horizon + 500. The lookback window IS the
    receptive field, so kernel_size, levels, channels and dropout all change the fitted model.
    """

    name = "tcn"
    has_eligibility_condition = True

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.kernel_size = int(hp.get("kernel_size", 3))
        self.levels = int(hp.get("levels", 4))
        self.channels = int(hp.get("channels", 32))
        self.dropout = float(hp.get("dropout", 0.1))
        self.receptive_field = receptive_field(self.kernel_size, self.levels)

    def lookback(self) -> int:
        return self.receptive_field

    def build_network(self, n_features: int):
        return TCNNet(n_features, self.horizon, self.kernel_size, self.levels, self.channels, self.dropout)

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return self.receptive_field + int(horizon if horizon is not None else self.horizon) + 500

    def hyperparameter_search_space(self) -> dict:
        return {
            "kernel_size": {"type": "int", "low": 2, "high": 5},
            "levels": {"type": "int", "low": 2, "high": 6},
            "channels": {"type": "choice", "choices": [16, 32, 64]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        n = len(endogenous_series(segment_df))
        if n < self.receptive_field + self.horizon:
            return EligibilityResult(False, f"history {n} < receptive field {self.receptive_field} + horizon {self.horizon} "
                                            f"(kernel={self.kernel_size}, levels={self.levels}): no training window is constructible")
        return EligibilityResult(True, f"history {n} covers receptive field {self.receptive_field}")
