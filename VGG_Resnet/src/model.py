"""VGG-11/13/16/19 adapted to 32x32 inputs, plus the `build_model` factory (ResNets live in resnet.py)."""
from __future__ import annotations

import torch.nn as nn

CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, arch="vgg16", num_classes=100, batch_norm=True, width_mult=1.0,
                 head_hidden=512, dropout=0.5):
        super().__init__()
        if arch not in CFGS:
            raise ValueError(f"arch must be one of {sorted(CFGS)}")
        layers, in_ch = [], 3
        for v in CFGS[arch]:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            out_ch = max(8, int(v * width_mult))
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 32x32 input is already 1x1 here; keeps other sizes usable
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_ch, head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(head_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # small logits -> initial loss ~ ln(K)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    """Plug-in point: the architecture family is inferred from the arch name (vgg16, resnet20, ...)."""
    arch = str(model_cfg["arch"])
    if arch.startswith("resnet"):
        from src.resnet import build_resnet
        return build_resnet(model_cfg, num_classes)
    return VGG(
        arch=arch, num_classes=num_classes, batch_norm=model_cfg.get("batch_norm", True),
        width_mult=model_cfg.get("width_mult", 1.0), head_hidden=model_cfg.get("head_hidden", 512),
        dropout=model_cfg.get("dropout", 0.5),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
