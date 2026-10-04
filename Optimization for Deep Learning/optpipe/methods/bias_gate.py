"""8.4 (12.11)  Gate biases set near 1.  A gate h in [0,1] multiplies a unit's output u (output u*h). If h starts near 0
the unit u 'does not have a chance to learn'; the LSTM forget-gate bias of 1 (Jozefowicz et al.) is the standard example.
Here every hidden layer is gated: output = sigmoid(W_g x + b_g) * relu(W x + b); the gate bias b_g is initialised to c
(tuned; c = 0 starts with half-open gates, c ~ 1 with mostly open gates)."""
import torch

from ..model import MLP
from .base import Method


class BiasGate(Method):
    name = "bias_gate"
    title = "Gate biases near 1 (gated hidden units)"
    section = "8.4 (12.11)"
    group = "initialization"
    regime = "gated hidden layers"

    def defaults(self):
        return {**super().defaults(), "gate_bias": 1.0}

    def space(self, trial):
        return {"gate_bias": trial.suggest_float("gate_bias", -2.0, 4.0)}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, gated=True, init=ctx.init)

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        for gate in model.gates:
            gate.bias.fill_(hp["gate_bias"])
