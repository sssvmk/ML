"""7.6  Semi-supervised learning.

Only `n_labeled` training images keep their labels (default 1,000); the rest of the training
set is used WITHOUT labels.  A generative criterion shares parameters with the discriminative
one (the book's recipe): the encoder feeds both a softmax classifier and a reconstruction
decoder, and the total loss is  CE(labeled) + lambda * MSE(reconstruction of labeled+unlabeled).
lambda = 0 recovers the supervised-only model on the same labels (reported as an ablation).
NOTE: trained with n_labeled labels, so metrics are NOT comparable with full-label methods."""
import math

import torch
import torch.nn.functional as F

from ..data import stratified_subset, to_float
from ..model import MLP
from .base import Method, TrainSet


class SemiSupervised(Method):
    name = "semi_supervised"
    title = "Semi-supervised learning (shared-encoder autoencoder)"
    section = "7.6"
    champion_eligible = False

    @property
    def regime(self):
        return f"{getattr(self, '_nl', 1000)} labels + unlabeled rest"

    def defaults(self):
        return {"lr": 0.05, "lam": 10.0}

    def space(self, trial):
        return {"lam": trial.suggest_float("lam", 0.3, 100.0, log=True)}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, init=ctx.init, with_decoder=True)

    def prepare(self, data, hp, ctx):
        n_lab = int(self.settings(ctx).get("n_labeled", 1000))
        self._nl = n_lab
        lab = stratified_subset(data.y_train, n_lab, ctx.seed)
        mask = torch.ones(len(data.y_train), dtype=torch.bool)
        mask[lab] = False
        return TrainSet(data.x_train[lab], data.y_train[lab],
                        extras={"x_unl": data.x_train[mask], "n_full": len(data.y_train)})

    def batches(self, ts, hp, ctx, gen):
        steps = math.ceil(ts.extras["n_full"] / ctx.batch_size)   # same compute as the full-data methods
        xu = ts.extras["x_unl"]
        for _ in range(steps):
            li = torch.randint(len(ts.y), (ctx.batch_size,), generator=gen)
            ui = torch.randint(len(xu), (ctx.batch_size,), generator=gen)
            yield {"x": to_float(ts.x[li]).to(ctx.device), "y": ts.y[li].to(ctx.device),
                   "x_unl": to_float(xu[ui]).to(ctx.device)}

    def data_loss(self, model, batch, hp):
        b = len(batch["y"])
        xs = torch.cat([batch["x"], batch["x_unl"]])
        out = model.forward_all(xs)
        ce = F.cross_entropy(out["logits"][:b], batch["y"])
        rec = F.mse_loss(out["recon"], xs)
        return ce + hp["lam"] * rec

    def extras(self, result, data, ctx, hp):
        from ..trainer import fit_single
        sup = fit_single(self, data, {**hp, "lam": 0.0}, ctx)
        return {"n_labeled": self._nl, "ablation_labeled_only_val_acc": sup.history[-1]["val_acc"],
                "ablation_labeled_only_val_loss": sup.history[-1]["val_loss"]}
