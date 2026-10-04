"""8.7.6  Curriculum learning: present EASY examples first and widen the pool to the whole training set.
Difficulty = loss of a quick reference model (softmax regression trained for `scorer_epochs` epochs). The sampling pool at
epoch e is the easiest fraction  f_e = start_frac + (1 - start_frac) * (e - 1) / T  (T = pace_frac * epochs, capped at 1).
Every epoch still draws N examples (with replacement from the pool), so compute equals the other methods."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import TrainSet
from ..data import to_float
from ..trainer import quick_sgd
from .base import Method


class Curriculum(Method):
    name = "curriculum"
    title = "Curriculum learning (easy-to-hard sampling)"
    section = "8.7.6 (b)"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "start_frac": 0.3, "pace_frac": 0.5}

    def space(self, trial):
        return {"start_frac": trial.suggest_float("start_frac", 0.05, 0.8), "pace_frac": trial.suggest_float("pace_frac", 0.2, 0.9)}

    def prepare(self, data, hp, ctx):
        torch.manual_seed(ctx.seed)
        scorer = nn.Linear(data.x_train.shape[1], 10).to(ctx.device)
        epochs = int(self.settings(ctx).get("scorer_epochs", 2))
        quick_sgd(lambda xb, yb: F.cross_entropy(scorer(xb), yb), scorer.parameters(), data.x_train, data.y_train, epochs,
                  0.1, ctx.batch_size, ctx.device, ctx.seed)
        with torch.no_grad():
            losses = torch.cat([F.cross_entropy(scorer(to_float(data.x_train[i:i + 4096]).to(ctx.device)),
                                                data.y_train[i:i + 4096].to(ctx.device), reduction="none").cpu()
                                for i in range(0, len(data.y_train), 4096)])
        return TrainSet(data.x_train, data.y_train, {"order": torch.argsort(losses)})

    @staticmethod
    def pool_fraction(hp, epoch: int, total: int) -> float:
        T = max(1.0, hp["pace_frac"] * total)
        return min(1.0, hp["start_frac"] + (1.0 - hp["start_frac"]) * (epoch - 1) / T)

    def batches(self, ts, hp, ctx, gen, epoch):
        n = len(ts.y)
        pool = ts.extras["order"][: max(ctx.batch_size, math.ceil(self.pool_fraction(hp, epoch, ctx.epochs) * n))]
        idx_all = pool[torch.randint(len(pool), (n,), generator=gen)]
        for i in range(0, n, ctx.batch_size):
            idx = idx_all[i:i + ctx.batch_size]
            yield {"x": to_float(ts.x[idx]).to(ctx.device), "y": ts.y[idx].to(ctx.device)}
