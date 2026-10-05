"""SSD with a ResNet base network: the paper's multi-map design (conv4_3 -> layer2, fc7 -> layer3, paper extras)."""
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

from ssd_voc.boxes import generate_priors

WEIGHTS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights", "resnet50": "ResNet50_Weights",
           "resnet101": "ResNet101_Weights"}
FREEZE_ORDER = ["stem", "layer1", "layer2"]


class L2Norm(nn.Module):
    """Paper sec. 3.1: scale the first map's feature norm to a learnable value (initially 20) at every location."""

    def __init__(self, channels: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), float(scale)))

    def forward(self, x):
        return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10) * self.weight.view(1, -1, 1, 1)


def _extra_block(in_ch, mid, out, stride, pad):
    return nn.Sequential(nn.Conv2d(in_ch, mid, 1), nn.ReLU(inplace=True),
                         nn.Conv2d(mid, out, 3, stride=stride, padding=pad), nn.ReLU(inplace=True))


class SSD(nn.Module):
    def __init__(self, num_classes: int, mcfg: dict, pretrained: bool = False):
        """num_classes includes the background (index 0)."""
        super().__init__()
        name = mcfg["backbone"]
        if name not in WEIGHTS:
            raise ValueError(f"backbone must be one of {sorted(WEIGHTS)}")
        self.num_classes, self.input_size = num_classes, int(mcfg["input_size"])
        self.conf_init_gain = float(mcfg.get("conf_init_gain", 0.1))
        weights = getattr(torchvision.models, WEIGHTS[name]).IMAGENET1K_V1 if pretrained else None
        norm = FrozenBatchNorm2d if mcfg.get("freeze_bn", True) else nn.BatchNorm2d
        net = getattr(torchvision.models, name)(weights=weights, norm_layer=norm)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self._freeze(mcfg.get("freeze_up_to", "layer1"))
        with torch.no_grad():
            x = torch.zeros(1, 3, self.input_size, self.input_size)
            f1 = self.layer2(self.layer1(self.stem(x)))
            f2 = self.layer3(f1)
        chans, sizes, in_ch = [f1.shape[1], f2.shape[1]], [f1.shape[-1], f2.shape[-1]], f2.shape[1]
        self.extras = nn.ModuleList()
        y = f2
        for mid, out, stride, pad in mcfg["extras"]:
            blk = _extra_block(in_ch, mid, out, stride, pad)
            self.extras.append(blk)
            with torch.no_grad():
                y = blk(y)
            chans.append(out)
            sizes.append(y.shape[-1])
            in_ch = out
        self.feature_sizes, self.channels = sizes, chans
        self.boxes_per_loc = list(mcfg["boxes_per_loc"])
        if len(self.boxes_per_loc) != len(chans):
            raise ValueError(f"boxes_per_loc has {len(self.boxes_per_loc)} entries but there are {len(chans)} maps")
        self.l2norm = L2Norm(chans[0], mcfg.get("l2norm_scale", 20.0)) if mcfg.get("l2norm_first_map", True) else None
        self.drop = nn.Dropout2d(mcfg["dropout"]) if mcfg.get("dropout", 0.0) > 0 else nn.Identity()
        self.loc_heads = nn.ModuleList(nn.Conv2d(c, k * 4, 3, padding=1) for c, k in zip(chans, self.boxes_per_loc))
        self.conf_heads = nn.ModuleList(nn.Conv2d(c, k * num_classes, 3, padding=1)
                                        for c, k in zip(chans, self.boxes_per_loc))
        self.init_new_layers()
        pc = mcfg["priors"]
        self.register_buffer("priors", generate_priors(sizes, self.boxes_per_loc, pc["first_scale"], pc["min_scale"],
                                                       pc["max_scale"]), persistent=False)

    def _freeze(self, upto: str) -> None:
        if not upto:
            return
        if upto not in FREEZE_ORDER:
            raise ValueError(f"freeze_up_to must be one of {FREEZE_ORDER} or empty")
        mods = {"stem": self.stem, "layer1": self.layer1, "layer2": self.layer2}
        for n in FREEZE_ORDER[: FREEZE_ORDER.index(upto) + 1]:
            for p in mods[n].parameters():
                p.requires_grad_(False)

    def init_new_layers(self) -> None:
        """Paper sec. 3.1: Xavier for the extra layers and the box-regression heads (zero biases). Deviation: the class
        heads start with N(0, (gain / sqrt(fan_in))^2), gain 0.1, so initial logits are small whatever the feature scale:
        with Xavier the initial loss was several times its expected value (the sanity check measures this)."""
        for m in [*self.extras.modules(), *self.loc_heads.modules()]:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.reinit_conf_heads()

    def reinit_conf_heads(self) -> None:
        for m in self.conf_heads.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                nn.init.normal_(m.weight, std=self.conf_init_gain / fan_in ** 0.5)
                nn.init.zeros_(m.bias)

    @property
    def num_priors(self) -> int:
        return int(self.priors.shape[0])

    def forward(self, x):
        f1 = self.layer2(self.layer1(self.stem(x)))
        f2 = self.layer3(f1)
        feats = [self.l2norm(f1) if self.l2norm is not None else f1, f2]
        y = f2
        for blk in self.extras:
            y = blk(y)
            feats.append(y)
        feats = [self.drop(f) for f in feats]
        loc = torch.cat([h(f).permute(0, 2, 3, 1).flatten(1, 2).unflatten(2, (k, 4)).flatten(1, 2)
                         for h, f, k in zip(self.loc_heads, feats, self.boxes_per_loc, strict=True)], dim=1)
        conf = torch.cat([h(f).permute(0, 2, 3, 1).flatten(1, 2).unflatten(2, (k, self.num_classes)).flatten(1, 2)
                          for h, f, k in zip(self.conf_heads, feats, self.boxes_per_loc, strict=True)], dim=1)
        return loc, conf

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on conv/linear weights only (the paper decays weights; biases and the L2Norm scale are exempt)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.ndim > 1 else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_model(cfg: dict, pretrained: bool | None = None) -> SSD:
    """pretrained=None follows cfg (model.pretrained == 'imagenet'); inference code passes False (no weight download)."""
    use = (cfg["model"]["pretrained"] == "imagenet") if pretrained is None else pretrained
    return SSD(cfg["data"]["num_foreground"] + 1, cfg["model"], pretrained=use)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def load_compatible(model: nn.Module, state: dict, skip_prefixes=("conf_heads.",)) -> dict:
    """Load matching tensors; skip class-specific heads (and anything whose shape differs). Returns a report."""
    own = model.state_dict()
    ok, skipped = {}, []
    for k, v in state.items():
        if k.startswith(tuple(skip_prefixes)) or k not in own or own[k].shape != v.shape:
            skipped.append(k)
        else:
            ok[k] = v
    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped": skipped, "missing": [k for k in own if k not in ok]}
