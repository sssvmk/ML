"""8.5.2  RMSProp with Nesterov momentum (Algorithm 8.6): the gradient is taken at the interim point theta + a v."""
from ..optim import RMSProp as RMSPropOpt
from .base import Method


class RMSPropNesterov(Method):
    name = "rmsprop_nesterov"
    title = "RMSProp with Nesterov momentum"
    section = "8.5.2"
    group = "adaptive learning rate"
    lr_range = (1e-5, 5e-3)

    def defaults(self):
        return {**super().defaults(), "lr": 1e-4, "rho": 0.9, "alpha": 0.9}

    def space(self, trial):
        return {"rho": trial.suggest_float("rho", 0.8, 0.995), "alpha": trial.suggest_float("alpha", 0.5, 0.95)}

    def make_optimizer(self, model, hp, ctx):
        return RMSPropOpt(model.parameters(), lr=hp["lr"], rho=hp["rho"], momentum=hp["alpha"])

    def lr_factor(self, t, hp, ctx):
        return 1.0
