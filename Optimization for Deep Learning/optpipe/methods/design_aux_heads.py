"""8.7.5  Designing models to aid optimisation (c): auxiliary heads (deeply-supervised nets, GoogLeNet).
Extra classifiers are attached to the hidden layers and trained on the same task, so lower layers receive a large gradient
through a short path. They are DISCARDED after training (finalize), so inference uses the plain network. Diagnostic: same
deep network with the auxiliary weight set to 0 (validation only)."""
import torch.nn.functional as F

from ..model import MLP, strip_aux
from ..trainer import fit_minibatch
from .base import Method


class DesignAuxHeads(Method):
    name = "design_aux_heads"
    title = "Model design: auxiliary heads (deep supervision)"
    section = "8.7.5 (c)"
    group = "meta-algorithms"

    @property
    def regime(self):
        return f"deep MLP, {getattr(self, '_d', 6)} layers, heads discarded after training"

    def defaults(self):
        return {**super().defaults(), "depth": 6, "aux_weight": 0.3}

    def space(self, trial):
        return {"depth": trial.suggest_int("depth", 3, 10), "aux_weight": trial.suggest_float("aux_weight", 0.01, 1.0, log=True)}

    def build(self, hp, ctx):
        self._d = hp["depth"]
        return MLP([ctx.hidden[0]] * hp["depth"], aux_heads=list(range(hp["depth"] - 1)), init=ctx.init)

    def data_loss(self, model, batch, hp):
        out = model.forward_all(batch["x"])
        loss = F.cross_entropy(out["logits"], batch["y"])
        aux = [F.cross_entropy(v, batch["y"]) for k, v in out.items() if k.startswith("aux_")]
        return loss + hp["aux_weight"] * sum(aux) / max(len(aux), 1)

    def finalize(self, model):
        return strip_aux(model)

    def extras(self, result, data, ctx, hp):
        plain = fit_minibatch(self, data, {**hp, "aux_weight": 0.0}, ctx)
        return {"depth": hp["depth"], "ablation_no_aux_val_acc": plain.history[-1]["val_acc"],
                "ablation_no_aux_val_loss": plain.history[-1]["val_loss"]}
