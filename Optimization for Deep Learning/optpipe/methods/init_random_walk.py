"""8.4 (12.5)  Gain-tuned "random walk" initialisation (Sussillo, 2014).
In a feedforward net the norm of the back-propagated signal does a random walk from layer to layer (every layer has its
own matrix). Choose each layer's gain so that this walk has no drift: starting from random Gaussian weights and a random
unit error signal at the output, rescale layers top-down so that ||W^T (delta * relu')|| = ratio * ||delta||.
ratio = 1 is the critical value; the book notes the best value is near, not exactly at, the theoretical one (tuned).
Because biases are zero and ReLU is positively homogeneous, rescaling a layer never changes any ReLU mask."""
import math

import torch
import torch.nn.functional as F

from ..data import to_float
from ._init_utils import layers, zero_biases
from .base import Method


@torch.no_grad()
def random_walk_init(model, x, ratio: float = 1.0, seed: int = 0) -> list:
    """Initialise `model` in place; returns the per-layer backward norm ratios measured after calibration."""
    g = torch.Generator().manual_seed(seed)
    ls = layers(model)
    for lin in ls:
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / math.sqrt(lin.in_features))
    zero_biases(model)
    h, masks = model.norm(x), []
    for lin in model.hidden:
        z = lin(h)
        masks.append((z > 0).float())
        h = F.relu(z)
    delta = torch.randn(x.shape[0], model.out.out_features, generator=g)
    delta = delta / delta.norm(dim=1, keepdim=True)
    measured = []
    for i in reversed(range(len(ls))):
        back = delta @ ls[i].weight
        r = (back.norm(dim=1) / delta.norm(dim=1)).mean()
        ls[i].weight.mul_(ratio / r)
        back = back * (ratio / r)
        measured.append(float((back.norm(dim=1) / delta.norm(dim=1)).mean()))
        if i > 0:
            delta = back * masks[i - 1]
    return measured[::-1]


class InitRandomWalk(Method):
    name = "init_random_walk"
    title = "Gain-tuned random-walk initialization (Sussillo)"
    section = "8.4 (12.5)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "ratio": 1.0}

    def space(self, trial):
        return {"ratio": trial.suggest_float("ratio", 0.5, 1.5)}

    def initialize(self, model, hp, ctx, data):
        x = to_float(data.x_train[:512]).to(next(model.parameters()).device)
        measured = random_walk_init(model, x, hp["ratio"], ctx.seed)
        return {"backward_norm_ratios": measured}
