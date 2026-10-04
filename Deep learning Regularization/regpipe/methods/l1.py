"""7.1.2  L1 regularization:  J~ = J + alpha * ||w||_1.

Sub-gradient alpha*sign(w) pushes weights toward zero at a constant rate, giving sparse
solutions.  (Plain SGD hovers around zero rather than hitting it exactly, so sparsity is
reported as the fraction of weights with |w| < 1e-3.)"""
from .base import Method, weights_only


class L1(Method):
    name = "l1"
    title = "L1 regularization"
    section = "7.1.2"

    def defaults(self):
        return {"lr": 0.05, "alpha": 1e-5}

    def space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-7, 1e-3, log=True)}

    def penalty(self, model, batch, hp):
        return hp["alpha"] * sum(w.abs().sum() for w in weights_only(model))
