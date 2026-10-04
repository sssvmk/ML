"""Generic training machinery: SGD + momentum with exponential learning-rate decay.

Per epoch it records the optimised objective, hard-label train/validation cross-entropy
(comparable across methods), validation accuracy and AUC, the learning rate and wall time.
"""
from __future__ import annotations

import copy
import time

import torch

from .data import to_float
from .evaluate import evaluate
from .methods.base import Ctx, FitResult, Method, TrainSet


class DivergenceError(RuntimeError):
    pass


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """One model, one optimizer.  Bagging drives several of these in lock-step."""

    def __init__(self, method: Method, model, ts: TrainSet, hp: dict, ctx: Ctx, seed: int):
        self.method, self.model, self.ts, self.hp, self.ctx = method, model, ts, hp, ctx
        self.gen = torch.Generator().manual_seed(seed)
        self.opt = torch.optim.SGD(model.parameters(), lr=hp["lr"], momentum=ctx.momentum)
        self.model.to(ctx.device)

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def train_epoch(self) -> float:
        m, model, hp = self.method, self.model, self.hp
        model.train()
        total, n = 0.0, 0
        for batch in m.batches(self.ts, hp, self.ctx, self.gen):
            batch = m.transform(batch, model, hp)
            self.opt.zero_grad(set_to_none=True)
            with m.forward_context(model, hp):
                loss = m.loss(model, batch, hp)
                if not torch.isfinite(loss):
                    raise DivergenceError(f"non-finite loss with hp={hp}")
                loss.backward()
            self.opt.step()
            m.after_step(model, hp)
            total += loss.item() * len(batch["y"])
            n += len(batch["y"])
        return total / max(n, 1)

    def step_lr(self) -> None:
        for g in self.opt.param_groups:
            g["lr"] *= self.ctx.lr_decay


def eval_row(model, data, ts: TrainSet, device, epoch: int, obj: float, lr: float, secs: float) -> dict:
    n = min(10_000, len(ts.y))
    tr = evaluate(model, ts.x[:n], ts.y[:n], device)
    va = evaluate(model, data.x_val, data.y_val, device)
    return {"epoch": epoch, "train_obj": obj, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "val_loss": va["loss"], "val_acc": va["accuracy"], "val_auc": va["auc"],
            "lr": lr, "epoch_seconds": secs}


def fit_single(method: Method, data, hp: dict, ctx: Ctx, on_epoch=None) -> FitResult:
    set_seed(ctx.seed)
    t0 = time.time()
    model = method.build(hp, ctx)
    ts = method.prepare(data, hp, ctx)
    tr = Trainer(method, model, ts, hp, ctx, ctx.seed)
    history, best = [], None
    for epoch in range(1, ctx.epochs + 1):
        t1 = time.time()
        obj = tr.train_epoch()
        row = eval_row(model, data, ts, ctx.device, epoch, obj, tr.lr, time.time() - t1)
        history.append(row)
        if method.selection == "best" and (best is None or row["val_loss"] < best[0]):
            best = (row["val_loss"], epoch, copy.deepcopy(model.state_dict()))
        if on_epoch is not None:
            on_epoch(row)
        tr.step_lr()
        if method.should_stop(history, hp):
            break
    extras = {"epochs_run": len(history)}
    if best is not None:
        model.load_state_dict(best[2])
        extras["best_epoch"] = best[1]
    return FitResult(model.cpu(), history, extras, time.time() - t0)
