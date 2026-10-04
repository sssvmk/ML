"""8.4 (12.2)  Fixed-scale heuristic: W_ij ~ U(-1/sqrt(m), 1/sqrt(m)) for a layer with m inputs.
`gain` multiplies the bound: the book says the optimal scale lies near but not exactly at the theoretical value."""
import math

import torch
import torch.nn as nn

from ._init_utils import layers, zero_biases
from .base import Method


class InitFixedScale(Method):
    name = "init_fixed_scale"
    title = "Fixed-scale heuristic U(-1/sqrt(m), 1/sqrt(m))"
    section = "8.4 (12.2)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "gain": 1.0}

    def space(self, trial):
        return {"gain": trial.suggest_float("gain", 0.5, 4.0, log=True)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        for lin in layers(model):
            b = hp["gain"] / math.sqrt(lin.in_features)
            nn.init.uniform_(lin.weight, -b, b)
        zero_biases(model)
