"""Method interface.  One technique from chapter 8 = one subclass in its own module.

The trainer is generic; a method customises it ONLY through these hooks:

  defaults() / space(trial)      hyperparameters (l1, l2 and, if tune_lr, lr are added by the pipeline)
  build(hp, ctx)                 the model                           (architecture studies)
  initialize(model, hp, ctx, d)  parameter initialisation            (section 8.4, incl. pretraining)
  epochs(hp, ctx)                epoch budget left after pretraining
  make_optimizer(model, hp, ctx) the update rule                     (8.3, 8.5; custom steppers for 8.6)
  lr_factor(t, hp, ctx)          learning-rate schedule              (8.3.1)
  prepare / batches / transform  data presentation                   (curriculum, continuation)
  loss(model, batch, hp)         objective = cross-entropy + L1/L2 penalty (+ method terms)
  clip(model, hp, state)         gradient clipping                   (8.2.4)
  epoch_start / after_step       coordinate descent, Polyak averaging
  eval_model / finalize          which weights are evaluated and saved
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import Ctx, FitResult, TrainSet, penalty
from ..data import batch_indices, to_float
from ..model import MLP
from ..optim import SGD


class Method:
    name = "baseline"
    title = "Baseline: SGD + momentum + exponential decay, He init"
    section = "reference"
    group = "reference"
    regime = "full labels"
    mode = "minibatch"               # "minibatch" | "fullbatch"
    champion_eligible = True         # False for methods evaluated in a reduced regime
    tune_lr = True
    lr_range = (5e-3, 5e-1)

    # ------------------------------------------------------------------ configuration
    def settings(self, ctx: Ctx) -> dict:
        return ctx.cfg.get("method_settings", {}).get(self.name, {})

    def defaults(self) -> dict:
        return {"lr": 0.05, "l2": 1e-5, "l1": 1e-7}

    def space(self, trial) -> dict:
        return {}

    def budget(self, cfg: dict, phase: str) -> int:
        if self.mode == "fullbatch":
            return int(cfg["full_batch"]["iterations"] if phase == "final" else cfg["tune"]["fullbatch_iters"])
        return int(cfg["train"]["epochs"] if phase == "final" else cfg["tune"]["epochs"])

    # ------------------------------------------------------------------ model, initialisation, optimizer
    def build(self, hp: dict, ctx: Ctx):
        return MLP(ctx.hidden, init=ctx.init)

    def initialize(self, model, hp: dict, ctx: Ctx, data) -> dict | None:
        return None

    def epochs(self, hp: dict, ctx: Ctx) -> int:
        return ctx.epochs

    def make_optimizer(self, model, hp: dict, ctx: Ctx):
        return SGD(model.parameters(), lr=hp["lr"], momentum=ctx.momentum)

    def make_stepper(self, model, hp: dict, ctx: Ctx):
        raise NotImplementedError

    def lr_factor(self, t: float, hp: dict, ctx: Ctx) -> float:
        """Multiplier on hp['lr'] after t epochs (float).  Default: exponential decay per epoch."""
        return ctx.lr_decay ** int(t)

    # ------------------------------------------------------------------ data presentation
    def prepare(self, data, hp: dict, ctx: Ctx) -> TrainSet:
        return TrainSet(data.x_train, data.y_train)

    def batches(self, ts: TrainSet, hp: dict, ctx: Ctx, gen: torch.Generator, epoch: int):
        for idx in batch_indices(len(ts.y), ctx.batch_size, gen):
            yield {"x": to_float(ts.x[idx]).to(ctx.device), "y": ts.y[idx].to(ctx.device)}

    def transform(self, batch: dict, hp: dict, epoch: int, ctx: Ctx) -> dict:
        return batch

    # ------------------------------------------------------------------ objective and per-step hooks
    def data_loss(self, model, batch: dict, hp: dict):
        return F.cross_entropy(model(batch["x"]), batch["y"])

    def loss(self, model, batch: dict, hp: dict):
        return self.data_loss(model, batch, hp) + penalty(model, hp.get("l1", 0.0), hp.get("l2", 0.0))

    def init_state(self, model, hp: dict, ctx: Ctx) -> dict:
        return {}

    def epoch_start(self, model, opt, hp: dict, epoch: int, state: dict) -> None:
        return None

    def clip(self, model, hp: dict, state: dict) -> None:
        return None

    def after_step(self, model, hp: dict, state: dict) -> None:
        return None

    def eval_model(self, model, state: dict):
        return model

    def final_stats(self, model, state: dict) -> dict:
        return {}

    def finalize(self, model):
        return model

    # ------------------------------------------------------------------ orchestration hooks
    def fit(self, data, hp: dict, ctx: Ctx, on_epoch=None) -> FitResult:
        from ..fullbatch import fit_fullbatch
        from ..trainer import fit_minibatch
        return (fit_fullbatch if self.mode == "fullbatch" else fit_minibatch)(self, data, hp, ctx, on_epoch)

    def extras(self, result: FitResult, data, ctx: Ctx, hp: dict) -> dict:
        return {}
