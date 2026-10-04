"""7.2  Norm penalties as constrained optimisation.

Instead of a penalty, impose the hard constraint ||w_unit||_2 <= c on the incoming weights of
every unit and enforce it by explicit reprojection after each update (Hinton et al. 2012;
the book notes this is equivalent to a penalty with a separate KKT multiplier per unit, and
works well with a high learning rate)."""
import torch

from .base import Method, weights_only


def project_max_norm(model, c: float) -> None:
    with torch.no_grad():
        for w in weights_only(model):                       # w: [out, in]; each row = one unit
            norms = w.norm(dim=1, keepdim=True)
            w.mul_(torch.clamp(c / (norms + 1e-12), max=1.0))


class ConstrainedNorm(Method):
    name = "constrained_norm"
    title = "Norm penalties as constrained optimization (max-norm)"
    section = "7.2"

    def defaults(self):
        return {"lr": 0.1, "max_norm": 3.0}

    def space(self, trial):
        return {"max_norm": trial.suggest_float("max_norm", 1.0, 8.0, log=True)}

    def after_step(self, model, hp):
        project_max_norm(model, hp["max_norm"])

    def extras(self, result, data, ctx, hp):
        with torch.no_grad():
            at = [(w.norm(dim=1) >= 0.999 * hp["max_norm"]).float().mean().item() for w in weights_only(result.model)]
        return {"fraction_units_at_constraint": float(sum(at) / len(at))}
