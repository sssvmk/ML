"""8.5.1  AdaGrad (Algorithm 8.4): per-parameter rates scaled by the inverse root of the SUM of all past squared gradients."""
from ..optim import AdaGrad as AdaGradOpt
from .base import Method


class AdaGrad(Method):
    name = "adagrad"
    title = "AdaGrad"
    section = "8.5.1"
    group = "adaptive learning rate"
    lr_range = (1e-3, 0.5)

    def make_optimizer(self, model, hp, ctx):
        return AdaGradOpt(model.parameters(), lr=hp["lr"])

    def lr_factor(self, t, hp, ctx):
        return 1.0
