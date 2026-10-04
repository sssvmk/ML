"""Generic minibatch training loop and shared helpers.

Per epoch it records: the optimised objective, hard-label train/validation cross-entropy (comparable across
methods), validation accuracy and AUC, the learning rate, cumulative data passes and wall time.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .core import Ctx, DivergenceError, FitResult, TrainSet, set_seed
from .data import batch_indices, to_float
from .evaluate import evaluate, init_diagnostics


def eval_row(model, data, ts: TrainSet, device, epoch: int, obj: float, lr: float, secs: float) -> dict:
    n = min(10_000, len(ts.y))
    tr = evaluate(model, ts.x[:n], ts.y[:n], device)
    va = evaluate(model, data.x_val, data.y_val, device)
    return {"epoch": epoch, "train_obj": obj, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "val_loss": va["loss"], "val_acc": va["accuracy"], "val_auc": va["auc"], "lr": lr, "epoch_seconds": secs}


def fit_minibatch(method, data, hp: dict, ctx: Ctx, on_epoch=None) -> FitResult:
    set_seed(ctx.seed)
    t0 = time.time()
    model = method.build(hp, ctx).to(ctx.device)
    info = method.initialize(model, hp, ctx, data) or {}
    ts = method.prepare(data, hp, ctx)
    diag = init_diagnostics(model, ts.x, ts.y, ctx.device)
    opt = method.make_optimizer(model, hp, ctx)
    state = method.init_state(model, hp, ctx)
    gen = torch.Generator().manual_seed(ctx.seed)
    n_ep = method.epochs(hp, ctx)
    spe = math.ceil(len(ts.y) / ctx.batch_size)
    pre = info.get("pretrain_epochs", 0)
    history, step = [], 0
    for epoch in range(1, n_ep + 1):
        t1 = time.time()
        model.train()
        method.epoch_start(model, opt, hp, epoch, state)
        total, n = 0.0, 0
        for batch in method.batches(ts, hp, ctx, gen, epoch):
            for g in opt.param_groups:
                g["lr"] = hp["lr"] * method.lr_factor((epoch - 1) + (step % spe) / spe, hp, ctx)
            batch = method.transform(batch, hp, epoch, ctx)
            opt.prepare()
            opt.zero_grad(set_to_none=True)
            loss = method.loss(model, batch, hp)
            if not torch.isfinite(loss):
                raise DivergenceError(f"non-finite loss with hp={hp}")
            loss.backward()
            method.clip(model, hp, state)
            opt.step()
            method.after_step(model, hp, state)
            total += loss.item() * len(batch["y"])
            n += len(batch["y"])
            step += 1
        ev = method.eval_model(model, state)
        row = eval_row(ev, data, ts, ctx.device, epoch, total / max(n, 1), opt.param_groups[0]["lr"], time.time() - t1)
        row["passes"] = epoch + pre
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
    final = method.finalize(method.eval_model(model, state))
    extras = {"epochs_run": len(history), "passes": history[-1]["passes"], **diag, **info,
              **method.final_stats(model, state)}
    return FitResult(final.cpu(), history, extras, time.time() - t0)


def quick_sgd(loss_fn, params, x_u8, y, epochs: int, lr: float, batch_size: int, device, seed: int,
              momentum: float = 0.9) -> list:
    """Small self-contained SGD loop used by pretraining stages. loss_fn(xb, yb) -> scalar tensor."""
    from .optim import SGD
    opt = SGD(params, lr=lr, momentum=momentum)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(max(0, epochs)):
        tot, n = 0.0, 0
        for idx in batch_indices(len(x_u8), batch_size, gen):
            xb = to_float(x_u8[idx]).to(device)
            yb = y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(xb, yb)
            if not torch.isfinite(loss):
                raise DivergenceError("non-finite pretraining loss")
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        losses.append(tot / max(n, 1))
    return losses
