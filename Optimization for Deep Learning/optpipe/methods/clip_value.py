"""8.2.4 / 10.11.1 (13.1)  Element-wise gradient clipping (Mikolov, 2012): every component of the minibatch gradient is
clipped to [-v, v] just before the update.  The update direction is no longer the gradient direction (still a descent direction)."""
import torch

from .base import Method


class ClipValue(Method):
    name = "clip_value"
    title = "Gradient clipping: element-wise (value)"
    section = "8.2.4 (13.1)"
    group = "gradient clipping"

    def defaults(self):
        return {**super().defaults(), "v": 0.05}

    def space(self, trial):
        return {"v": trial.suggest_float("v", 1e-3, 1.0, log=True)}

    def clip(self, model, hp, state):
        hits = tot = 0
        for p in model.parameters():
            if p.grad is not None:
                hits += int((p.grad.abs() > hp["v"]).sum())
                tot += p.grad.numel()
        state["hits"] = state.get("hits", 0) + hits
        state["tot"] = state.get("tot", 0) + tot
        torch.nn.utils.clip_grad_value_(model.parameters(), hp["v"])

    def final_stats(self, model, state):
        return {"clipped_element_fraction": state.get("hits", 0) / max(state.get("tot", 1), 1)}
