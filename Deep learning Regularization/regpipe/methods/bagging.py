"""7.11  Bagging and other ensemble methods.

k MLPs, each trained on its own bootstrap resample of the training set (sampling with
replacement) from a different random initialisation; predictions are the average of the
members' softmax outputs.  Members train in lock-step so the loss curves are those of the
ENSEMBLE.  Diagnostic: mean single-member validation accuracy vs the ensemble's."""
import time

import numpy as np
import torch

from ..evaluate import evaluate
from ..model import MLP, Ensemble
from .base import FitResult, Method, TrainSet


class Bagging(Method):
    name = "bagging"
    title = "Bagging (ensemble of bootstrap-trained MLPs)"
    section = "7.11"

    @property
    def regime(self):
        return f"ensemble of {getattr(self, '_k', 5)} MLPs"

    def fit(self, data, hp, ctx, on_epoch=None):
        from ..trainer import Trainer, eval_row, set_seed
        st = self.settings(ctx)
        k = int(st.get("tune_members" if ctx.phase == "tune" else "n_members", 5))
        self._k = k
        set_seed(ctx.seed)
        t0 = time.time()
        g = np.random.default_rng(ctx.seed)
        n = len(data.y_train)
        trainers = []
        for j in range(k):
            idx = torch.from_numpy(g.integers(0, n, n))               # bootstrap resample
            torch.manual_seed(ctx.seed + 1000 + j)                    # different initialisation
            model = MLP(ctx.hidden, init=ctx.init)
            trainers.append(Trainer(self, model, TrainSet(data.x_train[idx], data.y_train[idx]), hp, ctx, ctx.seed + j))
        ensemble = Ensemble([t.model for t in trainers])
        ref = TrainSet(data.x_train, data.y_train)                    # curves use the original train set
        history = []
        for epoch in range(1, ctx.epochs + 1):
            t1 = time.time()
            obj = float(np.mean([t.train_epoch() for t in trainers]))
            row = eval_row(ensemble, data, ref, ctx.device, epoch, obj, trainers[0].lr, time.time() - t1)
            history.append(row)
            if on_epoch is not None:
                on_epoch(row)
            for t in trainers:
                t.step_lr()
        return FitResult(ensemble.cpu(), history, {"n_members": k}, time.time() - t0)

    def defaults(self):
        return {"lr": 0.05}

    def extras(self, result, data, ctx, hp):
        accs = [evaluate(m, data.x_val, data.y_val, torch.device("cpu"))["accuracy"] for m in result.model.members]
        ens = evaluate(result.model, data.x_val, data.y_val, torch.device("cpu"))["accuracy"]
        return {"n_members": len(accs), "member_val_acc_mean": float(np.mean(accs)),
                "member_val_acc_min": float(np.min(accs)), "ensemble_val_acc": ens}
