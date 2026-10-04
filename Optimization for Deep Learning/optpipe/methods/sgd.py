"""8.3.1  Minibatch stochastic gradient descent (Algorithm 8.1): theta <- theta - e * g_minibatch.
Plain SGD: no momentum, constant learning rate, so it isolates the update rule."""
from ..optim import SGD as SGDOpt
from .base import Method


class MinibatchSGD(Method):
    name = "sgd"
    title = "Minibatch stochastic gradient descent"
    section = "8.3.1"
    group = "update rules"

    def make_optimizer(self, model, hp, ctx):
        return SGDOpt(model.parameters(), lr=hp["lr"], momentum=0.0)

    def lr_factor(self, t, hp, ctx):
        return 1.0
