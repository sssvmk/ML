"""Image transforms used by continuation methods."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import IMG


def gaussian_kernel(sigma: float, radius: int = 3) -> torch.Tensor:
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(ax ** 2) / (2 * max(sigma, 1e-6) ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of flat images x [B,784]; sigma<=0 is the identity."""
    if sigma <= 1e-3:
        return x
    k = gaussian_kernel(sigma).to(x.device)
    r = (len(k) - 1) // 2
    img = x.view(-1, 1, IMG, IMG)
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="constant"), k.view(1, 1, 1, -1))
    img = F.conv2d(F.pad(img, (0, 0, r, r), mode="constant"), k.view(1, 1, -1, 1))
    return img.reshape(-1, IMG * IMG)
