"""Method interface.  One regularization method = one subclass in its own module.

The trainer is generic; a method customises it ONLY through these hooks, so each module
shows exactly what that regularizer changes and nothing else:

    space(trial)           method-specific hyperparameters to tune (lr is added by the pipeline)
    build(hp, ctx)         the model (dropout, multi-head, tied, ...)
    prepare(data, hp, ctx) which data the method trains on (subset, labeled/unlabeled, ...)
    batches(ts, hp, ctx)   how mini-batches are drawn
    transform(batch, ...)  change inputs/targets before the forward pass (augmentation, noise)
    forward_context(...)   wrap forward+backward (weight noise)
    data_loss(...)         the data term of the objective (label smoothing, adversarial, ...)
    penalty(...)           additive penalty on the objective (L1, L2, sparsity, tying, tangent prop)
    after_step(...)        post-update projection (max-norm)
    should_stop(...)       early stopping
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..data import N_CLASSES, batch_indices, to_float
from ..model import MLP, build_model, n_params


@dataclass
class Ctx:
    """Everything a method needs that is not a hyperparameter."""
    cfg: dict
    device: torch.device
    seed: int
    phase: str = "final"          # "tune" | "final"
    epochs: int = 20

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
    x: torch.Tensor                   # uint8 [N,784]
    y: torch.Tensor                   # int64 [N]
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


def soft_cross_entropy(logits: torch.Tensor, y: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Cross-entropy against (optionally smoothed) targets: 1-eps on the true class and
    eps/(K-1) on every other class (Deep Learning book, section 7.5.1)."""
    if eps <= 0:
        return F.cross_entropy(logits, y)
    k = logits.shape[-1]
    target = torch.full_like(logits, eps / (k - 1))
    target.scatter_(1, y.unsqueeze(1), 1.0 - eps)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


class Method:
    name = "baseline"
    title = "Baseline (no regularization)"
    section = "reference"
    regime = "full labels"
    selection = "last"               # which weights are evaluated: "last" epoch or "best" validation epoch
    champion_eligible = True         # False for methods trained on a reduced data regime

    # ------------------------------------------------------------------ configuration
    def settings(self, ctx: Ctx) -> dict:
        return ctx.cfg.get("method_settings", {}).get(self.name, {})

    def defaults(self) -> dict:
        return {"lr": 0.05}

    def space(self, trial) -> dict:
        return {}

    # ------------------------------------------------------------------ model and data
    def build(self, hp: dict, ctx: Ctx) -> torch.nn.Module:
        return MLP(ctx.hidden, init=ctx.init)

    def prepare(self, data, hp: dict, ctx: Ctx) -> TrainSet:
        return TrainSet(data.x_train, data.y_train)

    def batches(self, ts: TrainSet, hp: dict, ctx: Ctx, gen: torch.Generator):
        for idx in batch_indices(len(ts.y), ctx.batch_size, gen):
            yield {"x": to_float(ts.x[idx]).to(ctx.device), "y": ts.y[idx].to(ctx.device)}

    # ------------------------------------------------------------------ per-step hooks
    def transform(self, batch: dict, model, hp: dict) -> dict:
        return batch

    def forward_context(self, model, hp: dict):
        return contextlib.nullcontext()

    def data_loss(self, model, batch: dict, hp: dict) -> torch.Tensor:
        return F.cross_entropy(model(batch["x"]), batch["y"])

    def penalty(self, model, batch: dict, hp: dict) -> torch.Tensor:
        return 0.0

    def after_step(self, model, hp: dict) -> None:
        return None

    def should_stop(self, history: list, hp: dict) -> bool:
        return False

    # ------------------------------------------------------------------ orchestration hooks
    def loss(self, model, batch, hp):
        return self.data_loss(model, batch, hp) + self.penalty(model, batch, hp)

    def fit(self, data, hp: dict, ctx: Ctx, on_epoch=None) -> FitResult:
        from ..trainer import fit_single
        return fit_single(self, data, hp, ctx, on_epoch=on_epoch)

    def extras(self, result: FitResult, data, ctx: Ctx, hp: dict) -> dict:
        """Method-specific diagnostics reported next to the standard metrics (validation set only)."""
        return {}

    def describe(self, hp: dict) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in hp.items())


def weights_only(model):
    """Weight matrices (biases are never regularized, as in the book)."""
    return model.weight_matrices() if hasattr(model, "weight_matrices") else \
        [p for n, p in model.named_parameters() if p.ndim > 1]


def rebuild(model):
    return build_model(model.spec)
