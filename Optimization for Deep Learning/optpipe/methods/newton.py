"""8.6.1  Newton's method:  theta <- theta - H^-1 grad J(theta)   (regularised: H + lambda I).

Exact Newton needs the n x n Hessian (n ~ 235,000 here: ~220 GB), so this is the damped, truncated Newton
(Newton-CG / 'Hessian-free') form:
  * the Newton system (H + lambda I) d = -g is solved by conjugate gradients using exact Hessian-vector
    products (double back-propagation) on a curvature sub-sample - the Hessian is never formed;
  * CG stops on negative curvature (Steihaug), so saddle points do not send the step uphill;
  * lambda is adapted Levenberg-Marquardt style from the ratio of actual to predicted reduction;
  * an Armijo backtracking line search guards each step.
For a quadratic objective and enough CG iterations this is exactly one Newton step (tests check this)."""
import torch

from ..fullbatch import Stepper
from ..linesearch import backtracking
from .base import Method


def truncated_cg(hvp, g, max_iter: int, tol: float):
    """Solve B d = -g with CG (B = hvp). Returns (d, B d). Stops on small residual or non-positive curvature."""
    d = torch.zeros_like(g)
    Bd = torch.zeros_like(g)
    r = -g.clone()
    p = r.clone()
    rs = float(r @ r)
    for k in range(max_iter):
        Bp = hvp(p)
        pBp = float(p @ Bp)
        if pBp <= 1e-12 * float(p @ p):                 # negative / zero curvature: stop here
            if k == 0:
                return p, Bp
            break
        a = rs / pBp
        d += a * p
        Bd += a * Bp
        r -= a * Bp
        rs_new = float(r @ r)
        if rs_new ** 0.5 <= tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return d, Bd


class NewtonCGStepper(Stepper):
    def __init__(self, damping, cg_iters, curv_batch, seed, exact=False):
        self.lam, self.cg_iters, self.curv_batch = damping, int(cg_iters), int(curv_batch)
        self.exact = exact                               # solve the Newton system to machine precision (tests)
        self.gen = torch.Generator().manual_seed(seed)
        self.rejected = 0

    def step(self, prob):
        f, g = prob.value_and_grad()
        gnorm = float(g.norm())
        self.note_gradient(g)
        if self.done:
            return {"loss": f, "alpha": 0.0}
        idx = torch.randperm(prob.N, generator=self.gen)[: self.curv_batch]
        lam = self.lam
        d, Bd = truncated_cg(lambda v: prob.hvp(v, idx) + lam * v, g, self.cg_iters,
                             tol=1e-13 if self.exact else min(0.5, gnorm ** 0.5) * gnorm)   # inexact-Newton forcing term
        pred = -(float(g @ d) + 0.5 * float(d @ Bd))
        theta = prob.get()
        slope = float(g @ d)
        if slope >= 0:                                   # not a descent direction: fall back to steepest descent
            d, slope, pred = -g, -float(g @ g), 0.0
        res = backtracking(lambda a: prob.value(theta + a * d), f, slope, alpha=1.0)
        if res is None:
            prob.set(theta)
            self.lam *= 4.0
            self.rejected += 1
            return {"loss": f, "alpha": 0.0}
        a, f_new = res
        prob.set(theta + a * d)
        self.note_progress(f, f_new)
        rho = (f - f_new) / pred if (a == 1.0 and pred > 0) else 0.0
        if rho > 0.75:
            self.lam = max(self.lam * 2.0 / 3.0, 1e-8)
        elif rho < 0.25:
            self.lam *= 1.5
        return {"loss": f_new, "alpha": a}

    def summary(self):
        return {"final_damping": self.lam, "rejected_steps": self.rejected}


class Newton(Method):
    name = "newton"
    title = "Newton's method (damped, truncated / Hessian-free)"
    section = "8.6.1"
    group = "second-order"
    mode = "fullbatch"
    tune_lr = False

    def defaults(self):
        d = super().defaults()
        d.pop("lr")
        return {**d, "damping": 1.0, "cg_iters": 20}

    def space(self, trial):
        return {"damping": trial.suggest_float("damping", 1e-3, 30.0, log=True),
                "cg_iters": trial.suggest_int("cg_iters", 5, 40)}

    def make_stepper(self, model, hp, ctx):
        return NewtonCGStepper(hp["damping"], hp["cg_iters"], self.settings(ctx).get("curvature_batch", 5000), ctx.seed)
