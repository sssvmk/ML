"""Helpers shared by the initialisation modules (section 8.4)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core import Ctx


def layers(model) -> list:
    """Hidden layers followed by the output layer (the nn.Linear layers that carry the signal)."""
    return model.weight_layers()


@torch.no_grad()
def zero_biases(model) -> None:
    for lin in layers(model):
        lin.bias.zero_()


def pretrain_epochs(frac: float, ctx: Ctx) -> int:
    """Epochs spent on pretraining; subtracted from the fine-tuning budget so total compute is equal."""
    return max(1, min(ctx.epochs - 1, round(frac * ctx.epochs))) if ctx.epochs > 1 else 0
