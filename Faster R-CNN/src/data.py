"""Pascal VOC 2007 detection data: parsing, splits, augmentation, loaders (+ a synthetic stand-in for offline tests).

Box convention everywhere: float32 XYXY in 0-based pixel coordinates. Labels: 1..20 (0 is background).
VOC annotations are 1-based inclusive, so xmin/ymin are shifted by -1.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
VOC_URLS = {"trainval": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "test": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"}


def voc_dir(root) -> Path:
    return Path(root) / "VOCdevkit" / "VOC2007"


def voc_available(root) -> bool:
    d = voc_dir(root)
    return all((d / p).is_dir() for p in ("Annotations", "JPEGImages")) and \
        (d / "ImageSets" / "Main" / "test.txt").exists() and (d / "ImageSets" / "Main" / "trainval.txt").exists()


def ensure_voc2007(root, download: bool = True) -> None:
    """Make sure VOC2007 trainval + test (with annotations) exist under root, downloading via torchvision if allowed."""
    if voc_available(root):
        return
    msg = (f"VOC2007 not found under {Path(root).resolve()}. Download and extract both archives into that folder "
           f"so that {voc_dir(root)} exists:\n  {VOC_URLS['trainval']}\n  {VOC_URLS['test']}")
    if not download:
        raise FileNotFoundError(msg)
    try:
        from torchvision.datasets import VOCDetection
        for split in ("trainval", "test"):
            VOCDetection(str(root), year="2007", image_set=split, download=True)
    except Exception as e:  # network blocked, proxy TLS inspection, disk full ...
        raise RuntimeError(f"{msg}\n(automatic download failed: {type(e).__name__}: {e})") from e


def read_ids(root, split: str) -> list[str]:
    path = voc_dir(root) / "ImageSets" / "Main" / f"{split}.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def parse_voc_xml(path) -> dict:
    tree = ET.parse(path).getroot()
    size = tree.find("size")
    objs = []
    for o in tree.findall("object"):
        bb = o.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        diff = o.find("difficult")
        objs.append({"name": o.find("name").text.strip(), "box": [x1 - 1.0, y1 - 1.0, x2, y2],
                     "difficult": bool(int(diff.text)) if diff is not None else False})
    return {"width": int(size.find("width").text), "height": int(size.find("height").text), "objects": objs}


def _target(boxes, labels, difficult, canvas_size, idx, with_difficult):
    t = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size),
         "labels": labels, "image_id": torch.tensor(idx)}
    if with_difficult:
        t["difficult"] = difficult
    return t


class VOC2007Detection(Dataset):
    def __init__(self, root, ids, transforms=None, include_difficult=True, return_difficult=False):
        self.root, self.ids, self.transforms = Path(root), list(ids), transforms
        self.include_difficult, self.return_difficult = include_difficult, return_difficult
        self.classes = list(VOC_CLASSES)
        self._cls = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = voc_dir(self.root)
        img = Image.open(d / "JPEGImages" / f"{self.ids[i]}.jpg").convert("RGB")
        ann = parse_voc_xml(d / "Annotations" / f"{self.ids[i]}.xml")
        objs = [o for o in ann["objects"] if self.include_difficult or not o["difficult"]]
        boxes = torch.tensor([o["box"] for o in objs], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([self._cls[o["name"]] for o in objs], dtype=torch.int64)
        diff = torch.tensor([o["difficult"] for o in objs], dtype=torch.bool)
        tgt = _target(boxes, labels, diff, (img.height, img.width), i, self.return_difficult)
        return self.transforms(img, tgt) if self.transforms else (img, tgt)


class SyntheticDetection(Dataset):
    """Coloured rectangles on noise; class = colour. Deterministic per index. Offline smoke tests only."""

    def __init__(self, n, num_classes, size=96, seed=0, transforms=None, return_difficult=False):
        self.n, self.k, self.size, self.seed = n, num_classes, size, seed
        self.transforms, self.return_difficult = transforms, return_difficult
        self.classes = [f"c{i}" for i in range(num_classes)]
        hues = np.linspace(0, 1, num_classes, endpoint=False)
        self.colors = [tuple(int(255 * c) for c in __import__("colorsys").hsv_to_rgb(h, 0.9, 0.95)) for h in hues]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        s = self.size
        img = Image.fromarray(rng.integers(60, 130, (s, s, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        boxes, labels = [], []
        for _ in range(int(rng.integers(1, 3))):
            w, h = rng.integers(s // 4, s // 2, 2)
            x, y = rng.integers(0, s - w), rng.integers(0, s - h)
            c = int(rng.integers(0, self.k))
            draw.rectangle([int(x), int(y), int(x + w) - 1, int(y + h) - 1], fill=self.colors[c])
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(c + 1)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        tgt = _target(boxes_t, torch.tensor(labels, dtype=torch.int64), torch.zeros(len(labels), dtype=torch.bool),
                      (s, s), i, self.return_difficult)
        return self.transforms(img, tgt) if self.transforms else (img, tgt)


def build_transforms(train: bool, aug: str = "standard"):
    """Detection-aware augmentation (boxes follow the image). 'strong' is the SSD-style recipe."""
    if not train or aug == "none":
        steps = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    elif aug == "flip":
        steps = [v2.ToImage(), v2.RandomHorizontalFlip(0.5), v2.ToDtype(torch.float32, scale=True)]
    elif aug == "standard":
        steps = [v2.ToImage(), v2.RandomPhotometricDistort(p=0.5), v2.RandomHorizontalFlip(0.5),
                 v2.SanitizeBoundingBoxes(), v2.ToDtype(torch.float32, scale=True)]
    elif aug == "strong":
        steps = [v2.ToImage(), v2.RandomPhotometricDistort(p=1.0), v2.RandomZoomOut(fill=114, side_range=(1.0, 2.0),
                                                                                    p=0.5),
                 v2.RandomIoUCrop(), v2.RandomHorizontalFlip(0.5), v2.SanitizeBoundingBoxes(),
                 v2.ToDtype(torch.float32, scale=True)]
    else:
        raise ValueError(f"aug must be none|flip|standard|strong, got {aug!r}")
    return v2.Compose(steps)


def split_ids(ids, val_fraction: float, seed: int):
    """Deterministic random split of VOC trainval ids into (train, val)."""
    ids = sorted(ids)
    if val_fraction <= 0:
        return ids, []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = sorted(ids[i] for i in perm[:n_val])
    train = sorted(ids[i] for i in perm[n_val:])
    return train, val


def build_datasets(cfg: dict) -> dict:
    d, seed = cfg["data"], cfg["seed"]
    aug = d.get("aug", "standard")
    rng = np.random.default_rng(seed)
    if d["dataset"] == "voc2007":
        ensure_voc2007(d["root"], d.get("download", True))
        trainval, test_ids = read_ids(d["root"], "trainval"), read_ids(d["root"], "test")
        tr_ids, va_ids = split_ids(trainval, d["val_fraction"], seed)
        if d.get("subset_train"):
            tr_ids = sorted(rng.permutation(tr_ids)[: d["subset_train"]].tolist())
        if d.get("subset_eval"):
            va_ids = sorted(rng.permutation(va_ids)[: d["subset_eval"]].tolist()) if va_ids else va_ids
            test_ids = sorted(rng.permutation(test_ids)[: d["subset_eval"]].tolist())
        inc = d.get("include_difficult_train", True)
        mk = lambda ids, train, rd: VOC2007Detection(d["root"], ids, build_transforms(train, aug), inc if train else True, rd)  # noqa: E731
        train = mk(tr_ids, True, False)
        train_eval = mk(tr_ids, False, True)           # un-augmented view of the train images (for train mAP)
        val, test = mk(va_ids, False, True), mk(test_ids, False, True)
        h = hashlib.sha1(("|".join(tr_ids) + "#" + "|".join(va_ids) + "#" + "|".join(test_ids)).encode())
        data_version = f"voc2007-tr{len(tr_ids)}-va{len(va_ids)}-te{len(test_ids)}-{h.hexdigest()[:10]}"
        classes = list(VOC_CLASSES)
    else:
        k, n, size = d["num_classes"], d["synthetic_size"], d.get("synthetic_image_size", 96)
        n_val = max(1, int(n * d["val_fraction"])) if d["val_fraction"] > 0 else 0
        mk = lambda n_, train, rd, sd: SyntheticDetection(n_, k, size, sd, build_transforms(train, aug), rd)  # noqa: E731
        train, train_eval = mk(n - n_val, True, False, seed), mk(n - n_val, False, True, seed)
        val = mk(n_val, False, True, seed + 1)
        test = mk(max(8, n // 4), False, True, seed + 2)
        data_version = f"synthetic-n{n}-k{k}-seed{seed}"
        classes = train.classes
    return {"train": train, "train_eval": train_eval, "val": val, "test": test, "classes": classes,
            "data_version": data_version}


def collate(batch):
    imgs, tgts = zip(*batch, strict=True)
    plain = []
    for t in tgts:
        o = {}
        for k, v in t.items():
            o[k] = v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
        plain.append(o)
    return list(imgs), plain


def _seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def build_loaders(cfg: dict, ds: dict, device: torch.device) -> dict:
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = d["num_workers"]
    common = dict(num_workers=nw, collate_fn=collate, worker_init_fn=_seed_worker, persistent_workers=nw > 0,
                  pin_memory=bool(d.get("pin_memory", False)) and device.type == "cuda")
    ev_bs = max(1, d["batch_size"])
    out = {"train": DataLoader(ds["train"], batch_size=d["batch_size"], shuffle=True, drop_last=len(ds["train"]) >
                               d["batch_size"], generator=g, **common),
           "test": DataLoader(ds["test"], batch_size=ev_bs, shuffle=False, **common)}
    out["val"] = DataLoader(ds["val"], batch_size=ev_bs, shuffle=False, **common) if len(ds["val"]) else None
    n_tr = min(len(ds["train_eval"]), int(cfg["training"].get("eval_train_subset", 0)))
    sub = Subset(ds["train_eval"], list(range(n_tr)))
    out["train_eval"] = DataLoader(sub, batch_size=ev_bs, shuffle=False, **common) if n_tr else None
    return out
