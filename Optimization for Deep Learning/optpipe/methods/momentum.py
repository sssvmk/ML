"""8.3.2  Momentum (Algorithm 8.2):  v <- a v - e g;  theta <- theta + v.  Constant learning rate."""
from ..optim import SGD as SGDOpt
from .base import Method


class Momentum(Method):
    name = "momentum"
    title = "Momentum"
    section = "8.3.2"
    group = "update rules"
    lr_range = (1e-3, 0.3)

    def defaults(self):
        return {**super().defaults(), "alpha": 0.9}

    def space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 0.5, 0.99)}

    def make_optimizer(self, model, hp, ctx):
        return SGDOpt(model.parameters(), lr=hp["lr"], momentum=hp["alpha"])

    def lr_factor(self, t, hp, ctx):
        return 1.0
