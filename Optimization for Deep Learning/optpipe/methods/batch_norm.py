"""8.7.1  Batch normalisation: every hidden layer normalises its pre-activations over the minibatch (then learns a scale and a
shift), which keeps activation statistics stable and decouples the layers' learning rates.  Training uses batch statistics;
inference uses running averages (stored in the bundle)."""
from ..model import MLP
from .base import Method


class BatchNorm(Method):
    name = "batch_norm"
    title = "Batch normalization"
    section = "8.7.1"
    group = "meta-algorithms"
    regime = "BatchNorm before each ReLU"
    lr_range = (1e-2, 1.0)

    def defaults(self):
        return {**super().defaults(), "lr": 0.1}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, batch_norm=True, init=ctx.init)
