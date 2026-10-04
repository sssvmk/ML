"""Algorithm 8.5 RMSProp and Algorithm 8.6 RMSProp with Nesterov momentum.

RMSProp:           r <- rho r + (1-rho) g*g;   theta <- theta - e / sqrt(delta + r) * g      (delta = 1e-6)
RMSProp+Nesterov:  theta~ = theta + a v;  g at theta~;  r <- rho r + (1-rho) g*g;
                   v <- a v - e / sqrt(r) * g;  theta <- theta + v
"""
import torch

from .base import BookOptimizer, params_with_grad


class RMSProp(BookOptimizer):
    def __init__(self, params, lr=1e-3, rho=0.9, delta=1e-6, momentum=0.0):
        super().__init__(params, dict(lr=lr, rho=rho, delta=delta, momentum=momentum))

    @torch.no_grad()
    def prepare(self):
        for group in self.param_groups:
            if group["momentum"] > 0:
                for p in group["params"]:
                    v = self.state[p].get("v")
                    if v is not None:
                        p.add_(v, alpha=group["momentum"])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, rho, a = group["lr"], group["rho"], group["momentum"]
            for p in params_with_grad(group):
                g = p.grad
                st = self.state[p]
                r = st.get("r")
                if r is None:
                    r = st["r"] = torch.zeros_like(p)
                r.mul_(rho).addcmul_(g, g, value=1 - rho)
                if a == 0:
                    p.addcdiv_(g, (r + group["delta"]).sqrt_(), value=-lr)
                else:
                    v = st.get("v")
                    if v is None:
                        v = st["v"] = torch.zeros_like(p)
                    step = g / (r + group["delta"]).sqrt()
                    v.mul_(a).add_(step, alpha=-lr)
                    p.add_(step, alpha=-lr)                  # theta~ - e g/sqrt(r)  ==  theta + v_new
