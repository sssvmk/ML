"""8.4 (12.4)  Orthogonal initialisation with a gain factor g (Saxe et al., 2013).
Each weight matrix is (semi-)orthogonal times g, so the singular values are all g: under a linear-chain model the number
of iterations to converge does not depend on depth.  g = sqrt(2) is the usual value for ReLU."""
import math

import torch
import torch.nn as nn

from ._init_utils import layers, zero_biases
from .base import Method


class InitOrthogonal(Method):
    name = "init_orthogonal"
    title = "Orthogonal initialization with gain (Saxe et al.)"
    section = "8.4 (12.4)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "gain": math.sqrt(2.0)}

    def space(self, trial):
        return {"gain": trial.suggest_float("gain", 0.5, 3.0)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        g = torch.Generator().manual_seed(ctx.seed)
        for lin in layers(model):
            w = torch.empty_like(lin.weight)
            nn.init.orthogonal_(w, gain=hp["gain"], generator=g)
            lin.weight.copy_(w)
        zero_biases(model)
