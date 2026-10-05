"""DETR (paper section 3): ResNet backbone -> 1x1 projection -> transformer encoder-decoder -> class + box heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

WEIGHTS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights", "resnet50": "ResNet50_Weights",
           "resnet101": "ResNet101_Weights"}


class Backbone(nn.Module):
    """ImageNet-pretrained torchvision ResNet, frozen BatchNorm, stem and layer1 frozen (as in the official code)."""

    def __init__(self, name: str, pretrained: bool, dilation: bool, freeze_bn: bool):
        super().__init__()
        if name not in WEIGHTS:
            raise ValueError(f"backbone must be one of {sorted(WEIGHTS)}")
        if dilation and name in ("resnet18", "resnet34"):
            raise ValueError("model.dilation (DETR-DC5) needs resnet50 or resnet101: BasicBlock cannot be dilated")
        weights = getattr(torchvision.models, WEIGHTS[name]).IMAGENET1K_V1 if pretrained else None
        net = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], weights=weights,
                                                norm_layer=FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d)
        for n, p in net.named_parameters():
            if "layer2" not in n and "layer3" not in n and "layer4" not in n:
                p.requires_grad_(False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.num_channels = 512 if name in ("resnet18", "resnet34") else 2048

    def forward(self, x, mask):
        f = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        m = F.interpolate(mask[None].float(), size=f.shape[-2:])[0].to(torch.bool)
        return f, m


class PositionEmbeddingSine(nn.Module):
    """2-D sine positional encoding (paper appendix A.4); padded pixels do not count."""

    def __init__(self, num_pos_feats=128, temperature=10000, scale=2 * math.pi):
        super().__init__()
        self.num_pos_feats, self.temperature, self.scale = num_pos_feats, temperature, scale

    def forward(self, mask):
        not_mask = ~mask
        y = not_mask.cumsum(1, dtype=torch.float32)
        x = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y = y / (y[:, -1:, :] + eps) * self.scale
        x = x / (x[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x, pos_y = x[:, :, :, None] / dim_t, y[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1, self.linear2 = nn.Linear(d, dim_ff), nn.Linear(dim_ff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout, self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src, key_padding_mask, pos):
        q = k = src + pos                                  # positional encodings are added at EVERY attention layer
        src = self.norm1(src + self.dropout1(self.self_attn(q, k, value=src, key_padding_mask=key_padding_mask)[0]))
        return self.norm2(src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src))))))


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1, self.linear2 = nn.Linear(d, dim_ff), nn.Linear(dim_ff, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask, pos, query_pos):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, k, value=tgt)[0]))
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(query=tgt + query_pos, key=memory + pos, value=memory,
                                                             key_padding_mask=memory_key_padding_mask)[0]))
        return self.norm3(tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(tgt))))))


class Transformer(nn.Module):
    def __init__(self, d, nhead, enc_layers, dec_layers, dim_ff, dropout):
        super().__init__()
        self.encoder = nn.ModuleList(EncoderLayer(d, nhead, dim_ff, dropout) for _ in range(enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d, nhead, dim_ff, dropout) for _ in range(dec_layers))
        self.decoder_norm = nn.LayerNorm(d)               # shared layer norm applied to every decoder output (aux losses)
        for p in self.parameters():                       # paper: Xavier initialisation of all transformer weights
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos):
        b = src.shape[0]
        src, pos = src.flatten(2).permute(2, 0, 1), pos.flatten(2).permute(2, 0, 1)
        query_pos = query_embed.unsqueeze(1).repeat(1, b, 1)
        mask = mask.flatten(1)
        memory = src
        for layer in self.encoder:
            memory = layer(memory, mask, pos)
        out, inter = torch.zeros_like(query_pos), []
        for layer in self.decoder:
            out = layer(out, memory, mask, pos, query_pos)
            inter.append(self.decoder_norm(out))
        return torch.stack(inter).transpose(1, 2)         # [decoder layers, batch, queries, d]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, [*dims[1:], out_dim], strict=True))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, num_classes: int, mcfg: dict, pretrained: bool = False):
        """num_classes excludes 'no object'; the class head has num_classes + 1 outputs (last = no object)."""
        super().__init__()
        d = mcfg["hidden_dim"]
        self.num_classes, self.aux_loss = num_classes, bool(mcfg.get("aux_loss", True))
        self.backbone = Backbone(mcfg["backbone"], pretrained, mcfg.get("dilation", False), mcfg.get("freeze_bn", True))
        self.pos_embed = PositionEmbeddingSine(d // 2)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d, 1)
        self.transformer = Transformer(d, mcfg["nheads"], mcfg["enc_layers"], mcfg["dec_layers"], mcfg["dim_feedforward"],
                                       mcfg["dropout"])
        self.query_embed = nn.Embedding(mcfg["num_queries"], d)
        self.class_embed = nn.Linear(d, num_classes + 1)
        self.bbox_embed = MLP(d, d, 4, 3)

    def forward(self, images, mask):
        feats, m = self.backbone(images, mask)
        pos = self.pos_embed(m).to(feats.dtype)
        hs = self.transformer(self.input_proj(feats), m, self.query_embed.weight, pos)
        logits, boxes = self.class_embed(hs), self.bbox_embed(hs).sigmoid()      # boxes: normalised (cx, cy, w, h)
        out = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(logits[:-1], boxes[:-1], strict=True)]
        return out

    def param_groups(self, lr: float, lr_backbone: float, weight_decay: float) -> list[dict]:
        """Backbone at a 10x lower LR (paper: stabilises the first epochs); weight decay 1e-4 on everything (AdamW)."""
        bb = [p for n, p in self.named_parameters() if n.startswith("backbone") and p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("backbone") and p.requires_grad]
        return [{"params": rest, "lr": lr, "lr_mult": 1.0, "weight_decay": weight_decay},
                {"params": bb, "lr": lr_backbone, "lr_mult": lr_backbone / lr, "weight_decay": weight_decay}]


def build_model(cfg: dict, pretrained: bool | None = None) -> DETR:
    """pretrained=None follows cfg (model.pretrained == 'imagenet'); inference code passes False (no weight download)."""
    use = (cfg["model"]["pretrained"] == "imagenet") if pretrained is None else pretrained
    return DETR(cfg["data"]["num_foreground"], cfg["model"], pretrained=use)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def load_compatible(model: nn.Module, state: dict, skip_prefixes=("class_embed.",)) -> dict:
    """Load matching tensors; skip the class head (and anything whose shape differs). Returns a report."""
    own = model.state_dict()
    ok, skipped = {}, []
    for k, v in state.items():
        if k.startswith(tuple(skip_prefixes)) or k not in own or own[k].shape != v.shape:
            skipped.append(k)
        else:
            ok[k] = v
    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped": skipped}
