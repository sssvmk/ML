"""8.2.4 / 10.11.1 (13.3)  Clipping the back-propagated gradient with respect to the hidden units (Graves, 2013): a hook on every
hidden pre-activation clamps dL/dz to [-v, v] DURING back-propagation, before it reaches the layers below, so large error
signals cannot compound as they travel down.  (Parameter gradients are then computed from the clipped signals.)"""
from ..model import MLP
from .base import Method


class ClipBackprop(Method):
    name = "clip_backprop"
    title = "Gradient clipping: back-propagated gradient w.r.t. hidden units"
    section = "8.2.4 (13.3)"
    group = "gradient clipping"

    def defaults(self):
        return {**super().defaults(), "v": 5e-3}

    def space(self, trial):
        return {"v": trial.suggest_float("v", 1e-5, 1e-1, log=True)}

    def build(self, hp, ctx):
        m = MLP(ctx.hidden, init=ctx.init)
        m.act_grad_clip = hp["v"]
        return m

    def final_stats(self, model, state):
        hits, tot = model.clip_hits
        return {"clipped_backprop_element_fraction": hits / max(tot, 1)}

    def finalize(self, model):
        model.act_grad_clip = None
        return model
