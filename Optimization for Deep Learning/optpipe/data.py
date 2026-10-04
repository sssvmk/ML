"""MNIST loading, splitting, auditing and batching.

Does MNIST come pre-split?  Yes, into 60,000 train / 10,000 test images (official split).
It has NO validation set, so a stratified validation set is carved out of the official train
set (default 10,000 -> 50,000 train / 10,000 validation / 10,000 test).  The test set is never
used for tuning or model selection.

Images are kept as uint8 [N, 784] and converted to float32 in [0, 1] per batch.
Standardisation lives INSIDE the model (model.Normalize) so training and serving cannot skew.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

IMG = 28
DIM = IMG * IMG
N_CLASSES = 10


@dataclass
class Splits:
    x_train: torch.Tensor  # uint8 [N, 784]
    y_train: torch.Tensor  # int64 [N]
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- loading
def _load_torchvision(root: str) -> dict:
    from torchvision import datasets  # imported lazily: only needed for real MNIST

    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return dict(
        x_train=tr.data.reshape(-1, DIM).contiguous(), y_train=tr.targets.long(),
        x_test=te.data.reshape(-1, DIM).contiguous(), y_test=te.targets.long(),
    )


def _load_npz(path: str) -> dict:
    z = np.load(path)
    t = lambda a: torch.from_numpy(np.asarray(a))
    return dict(
        x_train=t(z["x_train"]).reshape(-1, DIM).to(torch.uint8), y_train=t(z["y_train"]).long(),
        x_test=t(z["x_test"]).reshape(-1, DIM).to(torch.uint8), y_test=t(z["y_test"]).long(),
    )


def make_synthetic(n_train: int, n_test: int, seed: int = 0) -> dict:
    """MNIST-shaped synthetic digits for smoke tests ONLY (class prototypes + shifts + noise)."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    protos = []
    for _ in range(N_CLASSES):
        img = np.zeros((IMG, IMG))
        for _ in range(4):  # a few gaussian strokes per class
            cy, cx = g.uniform(6, 22, 2)
            sy, sx = g.uniform(1.5, 4.0, 2)
            img += np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
        protos.append(img / img.max())
    protos = np.stack(protos)

    def draw(n):
        y = g.integers(0, N_CLASSES, n)
        x = np.empty((n, IMG, IMG))
        for i, c in enumerate(y):
            dy, dx = g.integers(-3, 4, 2)
            x[i] = np.roll(protos[c], (dy, dx), (0, 1)) * g.uniform(0.7, 1.0) + g.normal(0, 0.45, (IMG, IMG))
        x = np.clip(x, 0, 1)
        return torch.from_numpy((x * 255).astype(np.uint8)).reshape(n, DIM), torch.from_numpy(y).long()

    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    return dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)


def load_raw(data_cfg: dict, seed: int = 0) -> dict:
    src = data_cfg["source"]
    if src == "mnist":
        raw = _load_torchvision(data_cfg["root"])
    elif src.startswith("npz:"):
        raw = _load_npz(src[4:])
    elif src == "synthetic":
        raw = make_synthetic(data_cfg.get("synthetic_train", 6000), data_cfg.get("synthetic_test", 1500), seed)
    else:
        raise ValueError(f"unknown data.source {src!r}")
    raw["pre_split"] = True  # MNIST ships train/test; no validation split
    raw["source"] = src
    return raw


# --------------------------------------------------------------------------- splitting
def stratified_holdout(y: torch.Tensor, n_hold: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keep_idx, hold_idx) with class proportions preserved in the held-out part."""
    g = np.random.default_rng(seed)
    y_np = y.numpy()
    frac = n_hold / len(y_np)
    hold = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y_np == c)
        g.shuffle(idx)
        hold.append(idx[: int(round(len(idx) * frac))])
    hold = np.sort(np.concatenate(hold))
    keep = np.setdiff1d(np.arange(len(y_np)), hold)
    return torch.from_numpy(keep), torch.from_numpy(hold)


def make_splits(raw: dict, val_size: int, seed: int) -> Splits:
    keep, hold = stratified_holdout(raw["y_train"], val_size, seed)
    s = Splits(
        x_train=raw["x_train"][keep], y_train=raw["y_train"][keep],
        x_val=raw["x_train"][hold], y_val=raw["y_train"][hold],
        x_test=raw["x_test"], y_test=raw["y_test"],
    )
    s.info = {
        "source": raw["source"],
        "dataset_pre_split": True,
        "official_split": "train/test (no validation split)",
        "validation_carved_from_train": True,
        "sizes": {"train": len(s.y_train), "val": len(s.y_val), "test": len(s.y_test)},
        "data_hash": data_hash(raw),
    }
    return s


def data_hash(raw: dict) -> str:
    h = hashlib.sha256()
    for k in ("x_train", "y_train", "x_test", "y_test"):
        h.update(raw[k].numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- audit
def _row_hashes(x: torch.Tensor) -> set:
    a = x.numpy()
    return {hashlib.md5(a[i].tobytes()).digest() for i in range(len(a))}


def audit_splits(s: Splits) -> dict:
    """Sizes, class balance, value ranges and exact-duplicate images across splits."""
    out = {"sizes": s.info["sizes"], "class_balance": {}, "problems": []}
    for name in ("train", "val", "test"):
        y = getattr(s, f"y_{name}")
        x = getattr(s, f"x_{name}")
        counts = torch.bincount(y, minlength=N_CLASSES).float()
        out["class_balance"][name] = (counts / counts.sum()).round(decimals=4).tolist()
        if x.dtype != torch.uint8:
            out["problems"].append(f"{name}: expected uint8 pixels, got {x.dtype}")
    ht, hv, hs = _row_hashes(s.x_train), _row_hashes(s.x_val), _row_hashes(s.x_test)
    out["duplicates_train_val"] = len(ht & hv)
    out["duplicates_train_test"] = len(ht & hs)
    out["duplicates_val_test"] = len(hv & hs)
    # MNIST contains a handful of genuine duplicate images; flag, do not hide.
    if out["duplicates_train_test"] > 0 or out["duplicates_val_test"] > 0:
        out["problems"].append("exact duplicate images between train/val and test (see counts)")
    return out


# --------------------------------------------------------------------------- batching
def to_float(xb: torch.Tensor) -> torch.Tensor:
    return xb.to(torch.get_default_dtype()).div_(255.0)


def stratified_subset(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Indices of a class-balanced subset of size ~n."""
    g = np.random.default_rng(seed)
    per = max(1, n // N_CLASSES)
    idx = []
    for c in range(N_CLASSES):
        ci = np.flatnonzero(y.numpy() == c)
        g.shuffle(ci)
        idx.append(ci[:per])
    idx = np.concatenate(idx)
    g.shuffle(idx)
    return torch.from_numpy(idx)


def batch_indices(n: int, batch_size: int, gen: torch.Generator, shuffle: bool = True):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield order[i : i + batch_size]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
