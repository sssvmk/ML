"""8.4 (12.13)  Unsupervised pretraining as the starting point.
The hidden layers are first trained as the encoder of an autoencoder (mirror-image decoder, sigmoid output, MSE) on the
training images WITHOUT labels; the classifier is then fine-tuned from there with a freshly initialised output layer.
The pretraining epochs are taken out of the epoch budget, so total compute equals the other methods."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import init_linear
from ..trainer import quick_sgd
from ._init_utils import pretrain_epochs
from .base import Method


class InitUnsupervisedPretrain(Method):
    name = "init_unsupervised_pretrain"
    title = "Unsupervised (autoencoder) pretraining as initialization"
    section = "8.4 (12.13)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "pre_frac": 0.3, "pre_lr": 0.05}

    def space(self, trial):
        return {"pre_frac": trial.suggest_float("pre_frac", 0.1, 0.6),
                "pre_lr": trial.suggest_float("pre_lr", 5e-3, 0.2, log=True)}

    def epochs(self, hp, ctx):
        return max(1, ctx.epochs - pretrain_epochs(hp["pre_frac"], ctx))

    def initialize(self, model, hp, ctx, data):
        p = pretrain_epochs(hp["pre_frac"], ctx)
        dims = [lin.in_features for lin in model.hidden] + [model.hidden[-1].out_features]
        dec = nn.ModuleList(nn.Linear(b, a) for a, b in zip(dims[:-1], dims[1:]))[::-1]
        dec = nn.ModuleList(list(dec))
        dec.apply(lambda m: init_linear(m, ctx.init))
        dec.to(ctx.device)

        def loss_fn(xb, _):
            h = model.norm(xb)
            for lin in model.hidden:
                h = F.relu(lin(h))
            for layer in dec[:-1]:
                h = F.relu(layer(h))
            return F.mse_loss(torch.sigmoid(dec[-1](h)), xb)

        params = [*model.hidden.parameters(), *dec.parameters()]
        losses = quick_sgd(loss_fn, params, data.x_train, data.y_train, p, hp["pre_lr"], ctx.batch_size, ctx.device,
                           ctx.seed)
        model.out.apply(lambda m: init_linear(m, ctx.init))                 # fresh output layer
        return {"pretrain_epochs": p, "pretrain_final_recon_mse": losses[-1] if losses else None}
