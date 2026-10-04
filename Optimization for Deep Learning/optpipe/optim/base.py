"""Minimal optimizer base. The update rules are implemented from the algorithm boxes of the book,
not delegated to torch.optim (tests check them against torch.optim / reference formulas)."""
import torch
from torch.optim import Optimizer


class BookOptimizer(Optimizer):
    """prepare() is called BEFORE the forward pass (Nesterov-type methods evaluate the gradient at an
    interim point); step() is called after backward()."""

    def prepare(self) -> None:
        return None

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]

    def set_lr(self, lr: float) -> None:
        for g in self.param_groups:
            g["lr"] = lr


def params_with_grad(group):
    return [p for p in group["params"] if p.grad is not None]
