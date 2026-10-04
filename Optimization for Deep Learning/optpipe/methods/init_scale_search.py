"""8.4 (12.7)  Initial scale as a hyperparameter, found two ways.

  mode="search": a per-layer multiplier on the He standard deviation is itself a tuned hyperparameter
                 (the pipeline's Optuna search plays the role of the book's random search over scales);
  mode="lsuv"  : look at activations on ONE minibatch (Mishkin & Matas, 2015): start from orthonormal weights and
                 rescale each layer, in order, until its pre-activation variance on the batch is 1."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import to_float
from ._init_utils import layers, zero_biases
from .base import Method


@torch.no_grad()
def lsuv_init(model, x, tol: float = 0.02, max_iter: int = 20, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    ls = layers(model)
    for lin in ls:
        w = torch.empty_like(lin.weight)
        nn.init.orthogonal_(w, generator=g)
        lin.weight.copy_(w)
    zero_biases(model)
    h, variances = model.norm(x), []
    for i, lin in enumerate(ls):
        for _ in range(max_iter):
            var = float(lin(h).var())
            if abs(var - 1.0) < tol:
                break
            lin.weight.div_(math.sqrt(var))
        variances.append(float(lin(h).var()))
        if i < len(model.hidden):
            h = F.relu(lin(h))
    return variances


class InitScaleSearch(Method):
    name = "init_scale_search"
    title = "Initial scale as a hyperparameter (search / LSUV)"
    section = "8.4 (12.7)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "mode": "lsuv", "scale_0": 1.0, "scale_1": 1.0, "scale_out": 1.0}

    def space(self, trial):
        mode = trial.suggest_categorical("mode", ["search", "lsuv"])
        hp = {"mode": mode}
        if mode == "search":
            for k in ("scale_0", "scale_1", "scale_out"):
                hp[k] = trial.suggest_float(k, 0.2, 3.0, log=True)
        return hp

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        if hp["mode"] == "lsuv":
            x = to_float(data.x_train[:512]).to(next(model.parameters()).device)
            return {"lsuv_preact_variances": lsuv_init(model, x, seed=ctx.seed)}
        ls = layers(model)
        names = [f"scale_{i}" for i in range(len(ls) - 1)] + ["scale_out"]
        for lin, name in zip(ls, names):
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            lin.weight.mul_(hp.get(name, 1.0))
        zero_biases(model)
        return None
