"""8.1.3  Batch (deterministic) gradient descent:  theta <- theta - e * grad J(theta) on the WHOLE training set.

One update costs a full pass, so for the same data-pass budget it makes ~N/batch times fewer updates than
minibatch SGD; the book's point about minibatches is visible in the 'passes' column."""
from ..fullbatch import Stepper
from .base import Method


class FullBatchGD(Stepper):
    def __init__(self, lr):
        self.lr = lr

    def step(self, prob):
        f, g = prob.value_and_grad()
        prob.set(prob.get() - self.lr * g)
        return {"loss": f, "alpha": self.lr}


class BatchGD(Method):
    name = "batch_gd"
    title = "Batch (full-dataset) gradient descent"
    section = "8.1.3"
    group = "update rules"
    mode = "fullbatch"
    lr_range = (0.05, 5.0)

    def defaults(self):
        return {**super().defaults(), "lr": 0.5}

    def make_stepper(self, model, hp, ctx):
        return FullBatchGD(hp["lr"])
