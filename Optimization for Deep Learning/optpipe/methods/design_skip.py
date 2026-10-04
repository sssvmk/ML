"""8.7.5  Designing models to aid optimisation (b): skip connections.
`depth` equal-width hidden layers with identity skip connections, h_{i+1} = h_i + relu(W h_i + b), shorten the shortest path from
the lower layers to the output and so mitigate vanishing gradients. Diagnostic: the SAME deep network without skips, same
hyperparameters (validation only)."""
from ..model import MLP
from ..trainer import fit_minibatch
from .base import Method


class DesignSkip(Method):
    name = "design_skip"
    title = "Model design: skip connections (deep residual MLP)"
    section = "8.7.5 (b)"
    group = "meta-algorithms"

    @property
    def regime(self):
        return f"deep MLP, {getattr(self, '_d', 6)} equal-width layers"

    def defaults(self):
        return {**super().defaults(), "depth": 6}

    def space(self, trial):
        return {"depth": trial.suggest_int("depth", 3, 10)}

    def build(self, hp, ctx):
        self._d = hp["depth"]
        return MLP([ctx.hidden[0]] * hp["depth"], residual=hp.get("skip", True), init=ctx.init)

    def extras(self, result, data, ctx, hp):
        plain = fit_minibatch(self, data, {**hp, "skip": False}, ctx)
        return {"depth": hp["depth"], "ablation_no_skips_val_acc": plain.history[-1]["val_acc"],
                "ablation_no_skips_val_loss": plain.history[-1]["val_loss"],
                "ablation_no_skips_init_grad_first_over_last": plain.extras.get("init_grad_norm_first_over_last")}
