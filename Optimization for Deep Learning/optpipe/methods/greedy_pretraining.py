"""8.7.4  Supervised (greedy, layer-wise) pretraining (Bengio et al., 2007).
Stage s trains a SHALLOW supervised network with s hidden layers (re-using the layers learned in stage s-1, optionally frozen),
for s = 1 .. L-1. The learned layers then initialise the full L-layer network, which is fine-tuned jointly for the remaining
epochs (pretraining epochs are taken out of the epoch budget, so compute is equal)."""
import torch.nn.functional as F

from ..model import MLP
from ..trainer import quick_sgd
from .base import Method


class GreedyPretraining(Method):
    name = "greedy_pretraining"
    title = "Supervised greedy layer-wise pretraining"
    section = "8.7.4"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "pre_frac": 0.4, "pre_lr": 0.05, "freeze": False}

    def space(self, trial):
        return {"pre_frac": trial.suggest_float("pre_frac", 0.15, 0.7),
                "pre_lr": trial.suggest_float("pre_lr", 5e-3, 0.2, log=True),
                "freeze": trial.suggest_categorical("freeze", [False, True])}

    @staticmethod
    def _per_stage(hp, ctx) -> int:
        stages = len(ctx.hidden) - 1
        if stages < 1 or ctx.epochs < 2:
            return 0
        return max(1, min((ctx.epochs - 1) // stages, round(hp["pre_frac"] * ctx.epochs / stages)))

    def epochs(self, hp, ctx):
        return max(1, ctx.epochs - self._per_stage(hp, ctx) * max(len(ctx.hidden) - 1, 0))

    def initialize(self, model, hp, ctx, data):
        L, per = len(model.hidden), self._per_stage(hp, ctx)
        if per == 0:
            return {"pretrain_epochs": 0, "stages": 0}
        trained, last = [], None
        for s in range(1, L):
            stage = MLP(ctx.hidden[:s], init=ctx.init).to(ctx.device)
            for j, lay in enumerate(trained):
                stage.hidden[j].load_state_dict(lay.state_dict())
            params = [*stage.hidden[s - 1].parameters(), *stage.out.parameters()] if (hp["freeze"] and s > 1) \
                else list(stage.parameters())
            losses = quick_sgd(lambda xb, yb, st=stage: F.cross_entropy(st(xb), yb), params, data.x_train,
                               data.y_train, per, hp["pre_lr"], ctx.batch_size, ctx.device, ctx.seed + s)
            trained = [stage.hidden[j] for j in range(s)]
            last = losses[-1]
        for j, lay in enumerate(trained):
            model.hidden[j].load_state_dict(lay.state_dict())
        return {"pretrain_epochs": per * (L - 1), "stages": L - 1, "last_stage_train_loss": last}
