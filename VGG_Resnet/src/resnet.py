"""CIFAR ResNets (He et al., 2015, section 4.2): 6n+2 layers, 3 stages of n basic blocks, widths 16/32/64.

resnet20 (n=3), resnet32 (n=5), resnet56 (n=9); resnet44 and resnet110 are also available.
Shortcut "A" (paper default) is parameter-free: stride-2 subsampling plus zero-padded channels.
Shortcut "B" uses a 1x1 projection convolution when the shape changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS_PER_STAGE = {"resnet20": 3, "resnet32": 5, "resnet44": 7, "resnet56": 9, "resnet110": 18}
BASE_WIDTHS = (16, 32, 64)


class PadShortcut(nn.Module):
    """Option A: take every 2nd pixel and zero-pad the new channels (no parameters)."""

    def __init__(self, extra_channels: int):
        super().__init__()
        self.lo = extra_channels // 2
        self.hi = extra_channels - self.lo

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.lo, self.hi))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, shortcut: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        elif shortcut == "A":
            self.shortcut = PadShortcut(out_ch - in_ch)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, arch="resnet20", num_classes=10, width_mult=1.0, shortcut="A", zero_init_residual=False):
        super().__init__()
        if arch not in BLOCKS_PER_STAGE:
            raise ValueError(f"arch must be one of {sorted(BLOCKS_PER_STAGE)}")
        if shortcut not in ("A", "B"):
            raise ValueError("shortcut must be 'A' or 'B'")
        n = BLOCKS_PER_STAGE[arch]
        widths = [max(4, int(w * width_mult)) for w in BASE_WIDTHS]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        blocks, in_ch = [], widths[0]
        for stage, out_ch in enumerate(widths):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                blocks.append(BasicBlock(in_ch, out_ch, stride, shortcut))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc.weight, 0, 0.01)  # small logits -> initial loss close to ln(K)
        nn.init.zeros_(self.fc.bias)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def build_resnet(model_cfg: dict, num_classes: int) -> CifarResNet:
    return CifarResNet(
        arch=model_cfg["arch"], num_classes=num_classes, width_mult=model_cfg.get("width_mult", 1.0),
        shortcut=model_cfg.get("shortcut", "A"), zero_init_residual=model_cfg.get("zero_init_residual", False))
