"""8.2.4 / 10.11.1 (13.2)  Norm clipping (Pascanu et al., 2013): if ||g|| > v then g <- g v / ||g||, with ONE scale factor for
all parameters, so the step still points along the gradient and its length is bounded by e*v."""
import torch

from .base import Method


class ClipNorm(Method):
    name = "clip_norm"
    title = "Gradient clipping: norm"
    section = "8.2.4 (13.2)"
    group = "gradient clipping"

    def defaults(self):
        return {**super().defaults(), "v": 2.0}

    def space(self, trial):
        return {"v": trial.suggest_float("v", 0.05, 10.0, log=True)}

    def clip(self, model, hp, state):
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), hp["v"]))     # returns the pre-clip norm
        state["steps"] = state.get("steps", 0) + 1
        state["clipped"] = state.get("clipped", 0) + int(norm > hp["v"])
        state["norm_sum"] = state.get("norm_sum", 0.0) + norm

    def final_stats(self, model, state):
        n = max(state.get("steps", 1), 1)
        return {"clipped_step_fraction": state.get("clipped", 0) / n, "mean_grad_norm_before_clip": state.get("norm_sum", 0.0) / n}
