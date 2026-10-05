"""Data loading, splitting and preprocessing. Imported by training AND serving (no train/serve skew)."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4865, 0.4409)   # CIFAR-100 train per-channel statistics (default when a checkpoint has none)
STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 32

# Per-channel train-set statistics. They are stored inside every checkpoint so serving always matches training.
DATASETS = {
    "cifar100": {"cls": "CIFAR100", "num_classes": 100, "mean": MEAN, "std": STD},
    "cifar10": {"cls": "CIFAR10", "num_classes": 10, "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.2470, 0.2435, 0.2616)},
    "synthetic": {"cls": None, "num_classes": None, "mean": MEAN, "std": STD},  # offline smoke tests only
}


def get_norm(cfg: dict):
    name = cfg["data"]["dataset"]
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]["mean"], DATASETS[name]["std"]


def train_transform(mean=MEAN, std=STD):
    return transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def eval_transform(mean=MEAN, std=STD):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def preprocess_uint8_batch(x, mean=MEAN, std=STD) -> torch.Tensor:
    """Serving-side preprocessing: uint8 NHWC -> normalized float NCHW (matches eval_transform)."""
    t = torch.as_tensor(np.asarray(x))
    if t.dtype != torch.uint8:
        raise ValueError(f"expected uint8 images, got {t.dtype}")
    if t.ndim != 4 or t.shape[-1] != 3:
        raise ValueError(f"expected shape (N, H, W, 3), got {tuple(t.shape)}")
    t = t.permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    sd = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (t - m) / sd


def pil_to_uint8(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Center-crop to square and resize to size x size RGB uint8."""
    img = ImageOps.fit(img.convert("RGB"), (size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def denormalize(x: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    sd = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * sd + m).clamp(0, 1)


class SyntheticCIFAR(Dataset):
    """Tiny learnable stand-in for CIFAR-100 (class = colour tint + noise). For offline smoke tests only."""

    def __init__(self, n, num_classes, data_seed, transform=None, tint_seed=0):
        tints = np.random.default_rng(tint_seed).uniform(0, 255, (num_classes, 3))
        rng = np.random.default_rng(data_seed)
        labels = np.arange(n) % num_classes
        rng.shuffle(labels)
        noise = rng.uniform(0, 255, (n, IMG_SIZE, IMG_SIZE, 3))
        imgs = 0.25 * noise + 0.75 * tints[labels][:, None, None, :]
        self.data = imgs.clip(0, 255).astype(np.uint8)
        self.targets = labels.tolist()
        self.classes = [f"class_{i}" for i in range(num_classes)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def stratified_split(targets, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.sort(train_idx), np.sort(val_idx)


def build_datasets(cfg: dict) -> dict:
    d, seed = cfg["data"], cfg["seed"]
    mean, std = get_norm(cfg)
    name = d["dataset"]
    if name in ("cifar10", "cifar100"):
        cls = getattr(datasets, DATASETS[name]["cls"])

        def make(train, tf):
            return cls(d["root"], train=train, download=True, transform=tf)
        train_aug = make(True, train_transform(mean, std))
        train_eval, test = make(True, eval_transform(mean, std)), make(False, eval_transform(mean, std))
        md5s = "".join(m for _, m in cls.train_list + cls.test_list)  # official checksums verified by torchvision
        data_version = f"{name}-{hashlib.sha1(md5s.encode()).hexdigest()[:12]}"
    elif name == "synthetic":
        n, k = d["synthetic_size"], d["num_classes"]
        train_aug = SyntheticCIFAR(n, k, seed, train_transform(mean, std))
        train_eval = SyntheticCIFAR(n, k, seed, eval_transform(mean, std))
        test = SyntheticCIFAR(max(k * 2, n // 4), k, seed + 1, eval_transform(mean, std))
        data_version = f"synthetic-n{n}-seed{seed}"
    else:
        raise ValueError(f"unknown dataset: {name}")

    train_idx, val_idx = stratified_split(train_eval.targets, d["val_fraction"], seed)
    rng = np.random.default_rng(seed)
    if d.get("subset_train"):
        train_idx = np.sort(rng.permutation(train_idx)[: d["subset_train"]])
    if d.get("subset_eval"):
        val_idx = np.sort(rng.permutation(val_idx)[: d["subset_eval"]])
    test_set = Subset(test, list(range(min(d["subset_eval"], len(test))))) if d.get("subset_eval") else test

    assert not set(train_idx.tolist()) & set(val_idx.tolist()), "train/val leakage"
    return {
        "train": Subset(train_aug, train_idx.tolist()),
        "val": Subset(train_eval, val_idx.tolist()),
        "test": test_set,
        "classes": list(train_eval.classes),
        "data_version": data_version,
        "norm": (mean, std),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def _seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def build_loaders(cfg: dict, ds: dict, device: torch.device) -> dict:
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    common = dict(
        batch_size=d["batch_size"], num_workers=d["num_workers"],
        pin_memory=bool(d.get("pin_memory", False)) and device.type == "cuda", persistent_workers=d["num_workers"] > 0,
        worker_init_fn=_seed_worker,
    )
    return {
        "train": DataLoader(ds["train"], shuffle=True, drop_last=True, generator=g, **common),
        "val": DataLoader(ds["val"], shuffle=False, **common),
        "test": DataLoader(ds["test"], shuffle=False, **common),
    }
