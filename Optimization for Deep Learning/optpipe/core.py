"""Shared small types: run context, training set, fit result, penalties, seeding."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Ctx:
    """Everything a method needs that is not a hyperparameter."""
    cfg: dict
    device: torch.device
    seed: int
    phase: str = "final"        # "tune" | "final"
    epochs: int = 20            # epochs (minibatch methods) or iterations (full-batch methods)

    @property
    def hidden(self):
        return self.cfg["model"]["hidden"]

    @property
    def init(self):
        return self.cfg["model"]["init"]

    @property
    def batch_size(self):
        return self.cfg["data"]["batch_size"]

    @property
    def momentum(self):
        return self.cfg["train"]["momentum"]

    @property
    def lr_decay(self):
        return self.cfg["train"]["lr_decay"]


@dataclass
class TrainSet:
    x: torch.Tensor                  # uint8 [N,784]
    y: torch.Tensor                  # int64 [N]
    extras: dict = field(default_factory=dict)


@dataclass
class FitResult:
    model: torch.nn.Module
    history: list
    extras: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def spec(self):
        return self.model.spec


class DivergenceError(RuntimeError):
    pass


L1_EPS = 1e-3


def set_l1_smoothing(eps: float) -> None:
    global L1_EPS
    L1_EPS = float(eps)


def weight_tensors(model) -> list:
    """Weight matrices (2-D parameters). Biases, BatchNorm scales and the precision parameter
    are never regularised."""
    return [p for p in model.parameters() if p.ndim == 2]


def penalty(model, l1: float, l2: float):
    """L1 + L2 (elastic-net) penalty:  l2/2 * sum w^2  +  l1 * sum sqrt(w^2 + eps^2).
    |w| is smoothed (pseudo-Huber) so that full-batch line searches and curvature stay valid."""
    ws = weight_tensors(model)
    total = 0.0
    if l2:
        total = total + 0.5 * l2 * sum((w ** 2).sum() for w in ws)
    if l1:
        total = total + l1 * sum(torch.sqrt(w ** 2 + L1_EPS ** 2).sum() for w in ws)
    return total


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
