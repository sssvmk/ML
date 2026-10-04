"""7.7  Multitask learning.

A shared trunk (all hidden layers) feeds the main head (10-way digit) plus two auxiliary heads
whose labels are derived for free from the digit: parity (even/odd) and magnitude (>= 5).
Loss = CE(digit) + lambda * (CE(parity) + CE(magnitude)).  The shared representation must
explain all tasks, which acts as a prior that regularizes the trunk.  Reported metrics are for
the MAIN task; auxiliary accuracies are diagnostics."""
import torch
import torch.nn.functional as F

from ..data import to_float
from ..model import MLP
from .base import Method


def aux_labels(y):
    return {"parity": y % 2, "high": (y >= 5).long()}


class Multitask(Method):
    name = "multitask"
    title = "Multitask learning (digit + parity + magnitude)"
    section = "7.7"

    def defaults(self):
        return {"lr": 0.05, "lam": 0.3}

    def space(self, trial):
        return {"lam": trial.suggest_float("lam", 0.02, 2.0, log=True)}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, init=ctx.init, aux_heads={"parity": 2, "high": 2})

    def data_loss(self, model, batch, hp):
        out = model.forward_all(batch["x"])
        y = batch["y"]
        aux = aux_labels(y)
        loss = F.cross_entropy(out["logits"], y)
        return loss + hp["lam"] * sum(F.cross_entropy(out[f"aux_{k}"], v) for k, v in aux.items())

    @torch.no_grad()
    def extras(self, result, data, ctx, hp):
        m = result.model.eval()
        out = m.forward_all(to_float(data.x_val))
        aux = aux_labels(data.y_val)
        return {f"aux_{k}_val_acc": (out[f"aux_{k}"].argmax(1) == v).float().mean().item() for k, v in aux.items()}
