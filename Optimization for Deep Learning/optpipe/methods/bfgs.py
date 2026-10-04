"""8.6.3  BFGS: build an approximation of the INVERSE Hessian from successive gradient differences,
  H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1/(y^T s),   d = -H g,
with a strong-Wolfe line search (which guarantees y^T s > 0).  The dense n x n matrix is the book's reason for
L-BFGS, so this method runs on a TINY network (784-8-10, 6,370 parameters); the regime is labelled in the report
and the method is excluded from champion selection."""
import torch

from ..core import Ctx
from ..fullbatch import Stepper
from ..linesearch import strong_wolfe
from ..model import MLP
from .base import Method


def bfgs_update(H: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> bool:
    """In-place inverse-Hessian update. Returns False (and skips) if the curvature condition fails."""
    sy = float(s @ y)
    if sy <= 1e-10 * float(s.norm() * y.norm()):
        return False
    rho = 1.0 / sy
    Hy = H @ y
    H.addr_(s, Hy, alpha=-rho)
    H.addr_(Hy, s, alpha=-rho)
    H.addr_(s, s, alpha=rho * rho * float(y @ Hy) + rho)
    return True


class BFGSStepper(Stepper):
    def __init__(self, c2=0.9):
        self.c2 = c2
        self.H = None
        self.f = self.g = None
        self.skipped = 0

    def step(self, prob):
        if self.g is None:
            self.f, self.g = prob.value_and_grad()
        theta, g, f = prob.get(), self.g, self.f
        d = -(self.H @ g) if self.H is not None else -g
        dphi0 = float(g @ d)
        if dphi0 >= 0:
            self.H, d = None, -g
            dphi0 = -float(g @ g)

        def phi(a):
            f_, g_ = prob.value_and_grad(theta + a * d)
            return f_, float(g_ @ d), g_

        a0 = 1.0 if self.H is not None else min(1.0, 1.0 / max(float(g.abs().sum()), 1e-12))
        res = strong_wolfe(phi, f, dphi0, a0, c2=self.c2, alpha_max=20.0, max_iter=12)
        if res is None:
            prob.set(theta)
            if self.H is None:
                self.done = True
            self.H = None                                  # reset to steepest descent and try again next iteration
            return {"loss": f, "alpha": 0.0}
        a, f_new, g_new, _ = res
        s = a * d
        prob.set(theta + s)
        self.note_progress(f, f_new)
        self.note_gradient(g_new)
        y = g_new - g
        if self.H is None:
            gamma = float(s @ y) / max(float(y @ y), 1e-20)
            self.H = torch.eye(prob.n, device=g.device) * (gamma if gamma > 0 else 1.0)
        if not bfgs_update(self.H, s, y):
            self.skipped += 1
        self.f, self.g = f_new, g_new
        return {"loss": f_new, "alpha": a}

    def summary(self):
        return {"skipped_updates": self.skipped}


class BFGS(Method):
    name = "bfgs"
    title = "BFGS (dense inverse-Hessian, reduced network)"
    section = "8.6.3"
    group = "second-order"
    mode = "fullbatch"
    tune_lr = False
    champion_eligible = False

    @property
    def regime(self):
        return f"reduced network 784-{getattr(self, '_h', 8)}-10 (dense n x n matrix)"

    def defaults(self):
        d = super().defaults()
        d.pop("lr")
        return {**d, "c2": 0.9}

    def space(self, trial):
        return {"c2": trial.suggest_float("c2", 0.5, 0.95)}

    def build(self, hp, ctx: Ctx):
        hidden = self.settings(ctx).get("hidden", [8])
        self._h = hidden[0]
        return MLP(hidden, init=ctx.init)

    def make_stepper(self, model, hp, ctx):
        return BFGSStepper(hp["c2"])
