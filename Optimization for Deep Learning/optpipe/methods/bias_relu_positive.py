"""8.4 (12.10)  Small positive bias on ReLU hidden units (e.g. 0.1) so that units are not dead (saturated at 0) at the start.
Not recommended with initialisations that do not expect strong input from the biases (e.g. random-walk)."""
import torch

from .base import Method


class BiasReluPositive(Method):
    name = "bias_relu_positive"
    title = "Small positive ReLU bias"
    section = "8.4 (12.10)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "bias": 0.1}

    def space(self, trial):
        return {"bias": trial.suggest_float("bias", 0.0, 0.5)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        for lin in model.hidden:
            lin.bias.fill_(hp["bias"])
        model.out.bias.zero_()
