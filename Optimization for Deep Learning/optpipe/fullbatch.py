"""Full-batch machinery for batch gradient descent, conjugate gradients, BFGS, L-BFGS and Newton-CG.

FlatProblem exposes the training objective (mean cross-entropy + L1/L2 penalty) as a function of ONE
flat parameter vector, with exact gradients and exact Hessian-vector products, and counts the
data passes used (one pass = one forward+backward over the whole training set; a Hessian-vector
product on m examples is counted as 2 m / N passes).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .core import Ctx, DivergenceError, FitResult, penalty, set_seed
from .data import to_float


class FlatProblem:
    def __init__(self, model, x_u8, y, l1: float, l2: float, device, chunk: int = 10000):
        self.model = model.to(device).train()
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.n = sum(p.numel() for p in self.params)
        self.x, self.y, self.l1, self.l2, self.device, self.chunk = x_u8, y, l1, l2, device, chunk
        self.N = len(y)
        self.passes = 0.0

    # -- parameter vector <-> model
    def get(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params]).clone()

    @torch.no_grad()
    def set(self, vec: torch.Tensor) -> None:
        i = 0
        for p in self.params:
            p.copy_(vec[i : i + p.numel()].view_as(p))
            i += p.numel()

    def _flat(self, grads) -> torch.Tensor:
        return torch.cat([g.reshape(-1) for g in grads])

    def _grad(self, out, create_graph=False):
        """autograd.grad that returns zeros for parameters not used by `out` (e.g. biases in the penalty)."""
        gs = torch.autograd.grad(out, self.params, create_graph=create_graph, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, self.params)]

    def _batches(self, idx=None):
        n = self.N if idx is None else len(idx)
        for i in range(0, n, self.chunk):
            sl = slice(i, i + self.chunk) if idx is None else idx[i : i + self.chunk]
            yield to_float(self.x[sl]).to(self.device), self.y[sl].to(self.device)

    # -- objective
    def value_and_grad(self, vec=None):
        if vec is not None:
            self.set(vec)
        total, grad = 0.0, torch.zeros(self.n, device=self.device)
        for xb, yb in self._batches():
            loss = F.cross_entropy(self.model(xb), yb, reduction="sum") / self.N
            grad += self._flat(self._grad(loss))
            total += float(loss)
        pen = penalty(self.model, self.l1, self.l2)
        if torch.is_tensor(pen):
            grad += self._flat(self._grad(pen))
            total += float(pen)
        self.passes += 1.0
        if not torch.isfinite(grad).all() or total != total:
            raise DivergenceError("non-finite objective or gradient")
        return total, grad

    @torch.no_grad()
    def value(self, vec=None) -> float:
        if vec is not None:
            self.set(vec)
        total = 0.0
        for xb, yb in self._batches():
            total += float(F.cross_entropy(self.model(xb), yb, reduction="sum")) / self.N
        pen = penalty(self.model, self.l1, self.l2)
        self.passes += 1.0 / 3.0
        return total + (float(pen) if torch.is_tensor(pen) else 0.0)

    def hvp(self, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Exact Hessian-vector product of the objective, with the data term taken on the examples `idx`
        (the penalty is exact).  The model parameters must already be at the current point."""
        out = torch.zeros(self.n, device=self.device)
        m = len(idx)
        for xb, yb in self._batches(idx):
            loss = F.cross_entropy(self.model(xb), yb, reduction="sum") / m
            g = self._grad(loss, create_graph=True)
            gv = sum((gi * vi.view_as(gi)).sum() for gi, vi in zip(g, self._split(v)))
            out += self._flat(self._grad(gv))
        pen = penalty(self.model, self.l1, self.l2)
        if torch.is_tensor(pen):
            g = self._grad(pen, create_graph=True)
            gv = sum((gi * vi.view_as(gi)).sum() for gi, vi in zip(g, self._split(v)))
            out += self._flat(self._grad(gv))
        self.passes += 2.0 * m / self.N
        return out

    def _split(self, v):
        i, parts = 0, []
        for p in self.params:
            parts.append(v[i : i + p.numel()])
            i += p.numel()
        return parts


def fit_fullbatch(method, data, hp: dict, ctx: Ctx, on_epoch=None) -> FitResult:
    """One iteration of the method's stepper per 'epoch'; every iteration is evaluated on the validation set."""
    from .evaluate import init_diagnostics
    from .trainer import eval_row
    set_seed(ctx.seed)
    t0 = time.time()
    model = method.build(hp, ctx)
    info = method.initialize(model, hp, ctx, data) or {}
    ts = method.prepare(data, hp, ctx)
    diag = init_diagnostics(model, ts.x, ts.y, ctx.device)
    prob = FlatProblem(model, ts.x, ts.y, hp["l1"], hp["l2"], ctx.device)
    stepper = method.make_stepper(model, hp, ctx)
    stepper.gtol = float(ctx.cfg.get("full_batch", {}).get("gtol", stepper.gtol))
    history = []
    for it in range(1, ctx.epochs + 1):
        t1 = time.time()
        stats = stepper.step(prob)
        row = eval_row(model, data, ts, ctx.device, it, stats["loss"], stats.get("alpha", 0.0), time.time() - t1)
        row["passes"] = prob.passes
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if stepper.done:
            break
    extras = {"iterations_run": len(history), "passes": prob.passes, **diag, **info, **stepper.summary()}
    return FitResult(model.cpu(), history, extras, time.time() - t0)


class Stepper:
    """One full-batch iteration per step().  Subclasses implement step(prob) -> {"loss", "alpha"}."""
    done = False
    gtol = 1e-5                                   # stop when max |gradient component| <= gtol (scipy's convention)

    def step(self, prob: FlatProblem) -> dict:
        raise NotImplementedError

    def note_gradient(self, g: torch.Tensor) -> None:
        if float(g.abs().max()) <= self.gtol:
            self.done = True

    def note_progress(self, f_old: float, f_new: float, tol: float = 1e-8, patience: int = 3) -> None:
        """Stop once the objective has stopped decreasing (to float precision) for `patience` iterations, instead of
        burning data passes in line searches that can no longer find a decrease."""
        if f_old - f_new <= tol * max(1.0, abs(f_old)):
            self._stalls = getattr(self, "_stalls", 0) + 1
            if self._stalls >= patience:
                self.done = True
        else:
            self._stalls = 0

    def summary(self) -> dict:
        return {}
