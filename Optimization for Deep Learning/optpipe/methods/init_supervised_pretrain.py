"""8.4 (12.14)  Supervised training on a related (or unrelated) task as the starting point.
The hidden layers are trained on an auxiliary task, then the digit classifier is fine-tuned with a fresh output layer.
  task="related"  : 4-class label (parity x magnitude) derived from the digit - a coarser version of the real task;
  task="unrelated": 10-class labels produced by a FIXED RANDOM linear teacher on the pixels - supervised, but about nothing
                    a digit classifier needs (the book: even this can beat random initialisation).
Pretraining epochs are taken out of the epoch budget."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import to_float
from ..model import init_linear
from ..trainer import quick_sgd
from ._init_utils import pretrain_epochs
from .base import Method


def task_labels(task: str, x_u8, y, seed: int):
    if task == "related":
        return ((y % 2) * 2 + (y >= 5).long()), 4
    g = torch.Generator().manual_seed(seed + 77)
    teacher = torch.randn(x_u8.shape[1], 10, generator=g)
    return (to_float(x_u8) @ teacher).argmax(1), 10


class InitSupervisedPretrain(Method):
    name = "init_supervised_pretrain"
    title = "Supervised pretraining on a related / unrelated task"
    section = "8.4 (12.14)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "task": "related", "pre_frac": 0.3, "pre_lr": 0.05}

    def space(self, trial):
        return {"task": trial.suggest_categorical("task", ["related", "unrelated"]),
                "pre_frac": trial.suggest_float("pre_frac", 0.1, 0.6),
                "pre_lr": trial.suggest_float("pre_lr", 5e-3, 0.2, log=True)}

    def epochs(self, hp, ctx):
        return max(1, ctx.epochs - pretrain_epochs(hp["pre_frac"], ctx))

    def initialize(self, model, hp, ctx, data):
        p = pretrain_epochs(hp["pre_frac"], ctx)
        labels, k = task_labels(hp["task"], data.x_train, data.y_train, ctx.seed)
        head = nn.Linear(model.hidden[-1].out_features, k).to(ctx.device)
        init_linear(head, ctx.init)

        def loss_fn(xb, yb):
            h = model.norm(xb)
            for lin in model.hidden:
                h = F.relu(lin(h))
            return F.cross_entropy(head(h), yb)

        losses = quick_sgd(loss_fn, [*model.hidden.parameters(), *head.parameters()], data.x_train, labels, p,
                           hp["pre_lr"], ctx.batch_size, ctx.device, ctx.seed)
        model.out.apply(lambda m: init_linear(m, ctx.init))
        return {"pretrain_epochs": p, "pretrain_task": hp["task"], "pretrain_final_loss": losses[-1] if losses else None}
