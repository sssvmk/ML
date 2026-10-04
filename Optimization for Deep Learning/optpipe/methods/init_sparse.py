"""8.4 (12.6)  Sparse initialisation (Martens, 2010): every unit gets exactly k non-zero incoming weights.
The total input to a unit stays independent of the number of inputs m WITHOUT making each weight shrink with m;
different units see different inputs, which adds diversity. The non-zero weights are N(0, s^2) (strong prior: large values)."""
import torch

from ._init_utils import layers, zero_biases
from .base import Method


@torch.no_grad()
def sparse_init_(weight: torch.Tensor, k: int, std: float, generator=None) -> None:
    out_f, in_f = weight.shape
    k = min(int(k), in_f)
    weight.zero_()
    for u in range(out_f):
        idx = torch.randperm(in_f, generator=generator)[:k]
        weight[u, idx] = torch.randn(k, generator=generator) * std


class InitSparse(Method):
    name = "init_sparse"
    title = "Sparse initialization (k non-zero weights per unit)"
    section = "8.4 (12.6)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "k": 15, "std": 0.5}

    def space(self, trial):
        return {"k": trial.suggest_int("k", 3, 64, log=True), "std": trial.suggest_float("std", 0.05, 2.0, log=True)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        g = torch.Generator().manual_seed(ctx.seed)
        for lin in layers(model):
            sparse_init_(lin.weight, hp["k"], hp["std"], g)
        zero_biases(model)
