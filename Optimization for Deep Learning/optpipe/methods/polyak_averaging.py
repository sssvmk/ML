"""8.7.3  Polyak averaging: evaluate (and deploy) an exponential moving average of the parameter trajectory,
theta_avg <- tau theta_avg + (1 - tau) theta  with tau = 1 - one_minus_tau (tuned on a log scale),   the form the book recommends for non-convex problems (eq. 8.40).
The weights that are evaluated, plotted and saved are the AVERAGED ones; BatchNorm-style buffers (none here) would be copied."""
import copy

import torch

from .base import Method


class PolyakAveraging(Method):
    name = "polyak_averaging"
    title = "Polyak averaging (exponential moving average of weights)"
    section = "8.7.3"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "one_minus_tau": 0.01}

    def space(self, trial):
        return {"one_minus_tau": trial.suggest_float("one_minus_tau", 1e-4, 0.2, log=True)}

    def init_state(self, model, hp, ctx):
        return {"avg": [p.detach().clone() for p in model.parameters()]}

    @torch.no_grad()
    def after_step(self, model, hp, state):
        for a, p in zip(state["avg"], model.parameters()):
            a.mul_(1.0 - hp["one_minus_tau"]).add_(p.detach(), alpha=hp["one_minus_tau"])

    @torch.no_grad()
    def eval_model(self, model, state):
        avg = copy.deepcopy(model)
        for a, p in zip(state["avg"], avg.parameters()):
            p.copy_(a)
        return avg
