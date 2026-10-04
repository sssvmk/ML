"""8.5.2  RMSProp (Algorithm 8.5): AdaGrad with an exponentially weighted moving average of squared gradients."""
from ..optim import RMSProp as RMSPropOpt
from .base import Method


class RMSProp(Method):
    name = "rmsprop"
    title = "RMSProp"
    section = "8.5.2"
    group = "adaptive learning rate"
    lr_range = (1e-4, 1e-2)

    def defaults(self):
        return {**super().defaults(), "lr": 1e-3, "rho": 0.9}

    def space(self, trial):
        return {"rho": trial.suggest_float("rho", 0.8, 0.995)}

    def make_optimizer(self, model, hp, ctx):
        return RMSPropOpt(model.parameters(), lr=hp["lr"], rho=hp["rho"])

    def lr_factor(self, t, hp, ctx):
        return 1.0
