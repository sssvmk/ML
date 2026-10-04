"""7.1.1  L2 parameter regularization (weight decay):  J~ = J + (alpha/2) * ||w||^2.

Shrinks weights along directions of low Hessian curvature (each component is scaled by
lambda_i / (lambda_i + alpha)).  Biases are not penalised."""
from .base import Method, weights_only


class L2WeightDecay(Method):
    name = "l2_weight_decay"
    title = "L2 regularization (weight decay)"
    section = "7.1.1"

    def defaults(self):
        return {"lr": 0.05, "alpha": 1e-4}

    def space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-6, 3e-2, log=True)}

    def penalty(self, model, batch, hp):
        return 0.5 * hp["alpha"] * sum((w ** 2).sum() for w in weights_only(model))
