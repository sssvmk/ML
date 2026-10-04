"""7.10  Sparse representations.

Penalise the ACTIVATIONS (not the weights):  J~ = J + lambda * sum_layers mean(|h|).
ReLU units already output exact zeros, so the penalty drives a growing fraction of hidden units
to be inactive for any given input.  Reported: fraction of hidden activations that are zero."""
import torch.nn.functional as F

from .base import Method


class SparseRepresentations(Method):
    name = "sparse_representations"
    title = "Sparse representations (activation L1)"
    section = "7.10"

    def defaults(self):
        return {"lr": 0.05, "lam": 1e-2}

    def space(self, trial):
        return {"lam": trial.suggest_float("lam", 1e-4, 3e-1, log=True)}

    def loss(self, model, batch, hp):
        logits, acts = model(batch["x"], return_hidden=True)
        return F.cross_entropy(logits, batch["y"]) + hp["lam"] * sum(a.abs().mean() for a in acts)
