"""8.3.3  Nesterov momentum (Algorithm 8.3): the gradient is evaluated AFTER the velocity is applied,
at theta + a v.  The optimizer's prepare() moves the parameters to that interim point before the forward pass."""
from ..optim import SGD as SGDOpt
from .base import Method


class Nesterov(Method):
    name = "nesterov"
    title = "Nesterov momentum"
    section = "8.3.3"
    group = "update rules"
    lr_range = (1e-3, 0.3)

    def defaults(self):
        return {**super().defaults(), "alpha": 0.9}

    def space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 0.5, 0.99)}

    def make_optimizer(self, model, hp, ctx):
        return SGDOpt(model.parameters(), lr=hp["lr"], momentum=hp["alpha"], nesterov=True)

    def lr_factor(self, t, hp, ctx):
        return 1.0
