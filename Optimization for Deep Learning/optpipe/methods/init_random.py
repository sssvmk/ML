"""8.4 (12.1)  Random Gaussian or uniform initialisation: breaks the symmetry between units.
The scale does NOT depend on the layer width; the book stresses that the scale matters more than the distribution."""
import math

import torch
import torch.nn as nn

from ._init_utils import layers, zero_biases
from .base import Method


class InitRandom(Method):
    name = "init_random"
    title = "Random Gaussian / uniform initialization"
    section = "8.4 (12.1)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "dist": "normal", "std": 0.05}

    def space(self, trial):
        return {"dist": trial.suggest_categorical("dist", ["normal", "uniform"]),
                "std": trial.suggest_float("std", 1e-3, 0.5, log=True)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        for lin in layers(model):
            if hp["dist"] == "normal":
                nn.init.normal_(lin.weight, 0.0, hp["std"])
            else:
                b = math.sqrt(3.0) * hp["std"]                     # uniform with the same standard deviation
                nn.init.uniform_(lin.weight, -b, b)
        zero_biases(model)
