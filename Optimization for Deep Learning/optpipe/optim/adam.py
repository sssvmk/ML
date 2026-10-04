"""Algorithm 8.7 Adam:
   s <- r1 s + (1-r1) g;   r <- r2 r + (1-r2) g*g;   s^ = s/(1-r1^t);  r^ = r/(1-r2^t)
   theta <- theta - e * s^ / (sqrt(r^) + delta)          (defaults e=1e-3, r1=0.9, r2=0.999, delta=1e-8)"""
import torch

from .base import BookOptimizer, params_with_grad


class Adam(BookOptimizer):
    def __init__(self, params, lr=1e-3, rho1=0.9, rho2=0.999, delta=1e-8):
        super().__init__(params, dict(lr=lr, rho1=rho1, rho2=rho2, delta=delta))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            r1, r2 = group["rho1"], group["rho2"]
            for p in params_with_grad(group):
                g = p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["s"] = torch.zeros_like(p)
                    st["r"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                st["s"].mul_(r1).add_(g, alpha=1 - r1)
                st["r"].mul_(r2).addcmul_(g, g, value=1 - r2)
                s_hat = st["s"] / (1 - r1 ** t)
                r_hat = st["r"] / (1 - r2 ** t)
                p.addcdiv_(s_hat, r_hat.sqrt().add_(group["delta"]), value=-group["lr"])
