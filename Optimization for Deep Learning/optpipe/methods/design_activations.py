"""8.7.5  Designing models to aid optimisation (a): near-linear activations.
ReLU, leaky ReLU, ELU and maxout keep the gradient large over much of their domain; sigmoid and tanh saturate. The tuner
picks the activation; the diagnostic `init_grad_norm_first_over_last` shows how much gradient survives to the first layer."""
from ..model import MLP
from .base import Method


class DesignActivations(Method):
    name = "design_activations"
    title = "Model design: near-linear activations (ReLU, leaky ReLU, ELU, maxout vs tanh, sigmoid)"
    section = "8.7.5 (a)"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "activation": "relu"}

    def space(self, trial):
        return {"activation": trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "elu", "maxout", "tanh", "sigmoid"])}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, activation=hp["activation"], init=ctx.init)
