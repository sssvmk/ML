"""Batched image transforms shared by dataset augmentation and tangent propagation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import IMG


def affine_images(x: torch.Tensor, tx=0.0, ty=0.0, rot_deg=0.0, scale=0.0, shear=0.0) -> torch.Tensor:
    """Apply an affine transform to flat images x [B, 784] in [0,1].

    Each argument is a scalar or a [B] tensor.  tx/ty are in PIXELS, rot in degrees,
    scale is a relative zoom (0.1 = +10%), shear is a dimensionless x-shear.
    """
    b = x.shape[0]
    dev = x.device

    def t(v):
        v = torch.as_tensor(v, dtype=torch.float32, device=dev)
        return v.expand(b) if v.ndim == 0 else v

    tx, ty, rot, scale, shear = t(tx), t(ty), t(rot_deg) * math.pi / 180.0, t(scale), t(shear)
    s = 1.0 + scale
    cos, sin = torch.cos(rot) / s, torch.sin(rot) / s
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin + shear
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = -2.0 * tx / IMG   # affine_grid works in [-1,1] coordinates (2 units = 28 px)
    theta[:, 1, 2] = -2.0 * ty / IMG
    grid = F.affine_grid(theta, (b, 1, IMG, IMG), align_corners=False)
    out = F.grid_sample(x.view(b, 1, IMG, IMG), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(b, IMG * IMG)


def random_affine(x, max_shift: float, max_rot: float, max_scale: float) -> torch.Tensor:
    """Independent random shift / rotation / zoom per image (label-preserving for small ranges)."""
    b = x.shape[0]
    u = lambda m: (torch.rand(b, device=x.device) * 2 - 1) * m
    return affine_images(x, tx=u(max_shift), ty=u(max_shift), rot_deg=u(max_rot), scale=u(max_scale))


# Generators of the transformation group used for tangent vectors: name -> kwargs for +delta
TANGENT_GENERATORS = {
    "shift_x": dict(tx=1.0),
    "shift_y": dict(ty=1.0),
    "rotation": dict(rot_deg=10.0),
    "scale": dict(scale=0.1),
    "shear": dict(shear=0.1),
}


def affine_tangents(x: torch.Tensor, delta: float = 0.5, unit_norm: bool = True) -> torch.Tensor:
    """Tangent vectors of the transformation manifold at each image: [B, K, 784].

    Central finite difference  (T_{+d}(x) - T_{-d}(x)) / 2d  for each generator.
    """
    vecs = []
    for kw in TANGENT_GENERATORS.values():
        plus = affine_images(x, **{k: v * delta for k, v in kw.items()})
        minus = affine_images(x, **{k: -v * delta for k, v in kw.items()})
        vecs.append((plus - minus) / (2 * delta))
    v = torch.stack(vecs, dim=1)
    if unit_norm:
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v
