"""8.6.3  L-BFGS: BFGS without the matrix. Only the last m pairs (s_i, y_i) are stored and d = -H g is produced
by the two-loop recursion, starting from H0 = gamma I with gamma = s^T y / y^T y.  Memory O(m n), so it runs on the
full network.  Strong-Wolfe line search; full-batch (L-BFGS is not robust to minibatch noise)."""
import torch

from ..fullbatch import Stepper
from ..linesearch import strong_wolfe
from .base import Method


def two_loop(g, s_list, y_list):
    """Return H g for the L-BFGS inverse-Hessian approximation (two-loop recursion)."""
    if not s_list:
        return g.clone()
    q = g.clone()
    rhos = [1.0 / float(y @ s) for s, y in zip(s_list, y_list)]
    alphas = []
    for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rhos)):
        a = rho * float(s @ q)
        alphas.append(a)
        q -= a * y
    gamma = float(s_list[-1] @ y_list[-1]) / float(y_list[-1] @ y_list[-1])
    r = gamma * q
    for (s, y, rho), a in zip(zip(s_list, y_list, rhos), reversed(alphas)):
        b = rho * float(y @ r)
        r += (a - b) * s
    return r


class LBFGSStepper(Stepper):
    def __init__(self, history=10, c2=0.9):
        self.m, self.c2 = int(history), c2
        self.s, self.y = [], []
        self.f = self.g = None
        self.skipped = 0

    def step(self, prob):
        if self.g is None:
            self.f, self.g = prob.value_and_grad()
        theta, g, f = prob.get(), self.g, self.f
        d = -two_loop(g, self.s, self.y)
        dphi0 = float(g @ d)
        if dphi0 >= 0:
            self.s, self.y, d = [], [], -g
            dphi0 = -float(g @ g)

        def phi(a):
            f_, g_ = prob.value_and_grad(theta + a * d)
            return f_, float(g_ @ d), g_

        a0 = 1.0 if self.s else min(1.0, 1.0 / max(float(g.abs().sum()), 1e-12))
        res = strong_wolfe(phi, f, dphi0, a0, c2=self.c2, alpha_max=20.0, max_iter=12)
        if res is None:
            prob.set(theta)
            if not self.s:
                self.done = True
            self.s, self.y = [], []
            return {"loss": f, "alpha": 0.0}
        a, f_new, g_new, _ = res
        s = a * d
        prob.set(theta + s)
        self.note_progress(f, f_new)
        self.note_gradient(g_new)
        y = g_new - g
        if float(s @ y) > 1e-10 * float(s.norm() * y.norm()):
            self.s.append(s)
            self.y.append(y)
            if len(self.s) > self.m:
                self.s.pop(0)
                self.y.pop(0)
        else:
            self.skipped += 1
        self.f, self.g = f_new, g_new
        return {"loss": f_new, "alpha": a}

    def summary(self):
        return {"skipped_updates": self.skipped}


class LBFGS(Method):
    name = "lbfgs"
    title = "L-BFGS (limited-memory BFGS)"
    section = "8.6.3"
    group = "second-order"
    mode = "fullbatch"
    tune_lr = False

    def defaults(self):
        d = super().defaults()
        d.pop("lr")
        return {**d, "history": 10, "c2": 0.9}

    def space(self, trial):
        return {"history": trial.suggest_categorical("history", [5, 10, 20]),
                "c2": trial.suggest_float("c2", 0.5, 0.95)}

    def make_stepper(self, model, hp, ctx):
        return LBFGSStepper(hp["history"], hp["c2"])
