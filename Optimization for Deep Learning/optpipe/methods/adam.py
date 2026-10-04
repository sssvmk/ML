"""8.5.3  Adam (Algorithm 8.7): bias-corrected first and second moment estimates of the gradient."""
from ..optim import Adam as AdamOpt
from .base import Method


class Adam(Method):
    name = "adam"
    title = "Adam"
    section = "8.5.3"
    group = "adaptive learning rate"
    lr_range = (1e-4, 1e-2)

    def defaults(self):
        return {**super().defaults(), "lr": 1e-3, "rho1": 0.9, "rho2": 0.999}

    def space(self, trial):
        return {"rho1": trial.suggest_float("rho1", 0.8, 0.95), "rho2": trial.suggest_float("rho2", 0.99, 0.9999)}

    def make_optimizer(self, model, hp, ctx):
        return AdamOpt(model.parameters(), lr=hp["lr"], rho1=hp["rho1"], rho2=hp["rho2"])

    def lr_factor(self, t, hp, ctx):
        return 1.0
