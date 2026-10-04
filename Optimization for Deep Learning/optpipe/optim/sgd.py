"""Algorithms 8.1-8.3: SGD, SGD with momentum, SGD with Nesterov momentum.

momentum:   v <- a v - e g(theta);              theta <- theta + v
nesterov:   theta~ = theta + a v;  g at theta~; v <- a v - e g;  theta <- theta + v
            (implemented as: prepare() moves to theta~, step() applies theta <- theta~ - e g)
"""
import torch

from .base import BookOptimizer, params_with_grad


class SGD(BookOptimizer):
    def __init__(self, params, lr=0.05, momentum=0.0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def prepare(self):
        for group in self.param_groups:
            if group["nesterov"] and group["momentum"] > 0:
                for p in group["params"]:
                    v = self.state[p].get("v")
                    if v is not None:
                        p.add_(v, alpha=group["momentum"])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, a = group["lr"], group["momentum"]
            for p in params_with_grad(group):
                g = p.grad
                if a == 0:
                    p.add_(g, alpha=-lr)
                    continue
                st = self.state[p]
                v = st.get("v")
                if v is None:
                    v = st["v"] = torch.zeros_like(p)
                v.mul_(a).add_(g, alpha=-lr)                 # v <- a v - e g
                if group["nesterov"]:
                    p.add_(g, alpha=-lr)                     # theta~ - e g  ==  theta + v_new
                else:
                    p.add_(v)
