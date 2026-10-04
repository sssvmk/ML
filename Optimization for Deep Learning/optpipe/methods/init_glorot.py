"""8.4 (12.3)  Normalized (Glorot / Xavier) initialisation:  W_ij ~ U(-sqrt(6/(m+n)), sqrt(6/(m+n))).
Compromise between keeping forward activations (m inputs) and back-propagated gradients (n outputs) at constant scale."""
import math

import torch
import torch.nn as nn

from ._init_utils import layers, zero_biases
from .base import Method


class InitGlorot(Method):
    name = "init_glorot"
    title = "Normalized (Glorot/Xavier) initialization"
    section = "8.4 (12.3)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "gain": 1.0}

    def space(self, trial):
        return {"gain": trial.suggest_float("gain", 0.5, 3.0, log=True)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        for lin in layers(model):
            b = hp["gain"] * math.sqrt(6.0 / (lin.in_features + lin.out_features))
            nn.init.uniform_(lin.weight, -b, b)
        zero_biases(model)
