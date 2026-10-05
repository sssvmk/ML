"""Detection data for DETR: VOC / COCO records, the paper's augmentation, padded batches with masks, loaders.

Training sample:   (image [3,H,W] normalised, {"boxes": [n,4] normalised (cx,cy,w,h), "labels": [n] in 0..K-1})
Evaluation sample: (image, same-style target for the loss, meta with ORIGINAL-pixel xyxy boxes, 1-based labels, difficult, size)
Batches are padded to the largest image; `mask` is True on the padding (paper section 3.2).
"""
import colorsys
import hashlib
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from detr_voc.data_download import coco_root, ensure_coco, ensure_voc, voc_root
from detr_voc.infer import aspect_size, to_normalised_tensor

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")


# ------------------------------------------------------------------ records
def parse_voc_xml(path) -> dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        d = o.find("difficult")
        objs.append({"name": o.find("name").text.strip(), "box": [x1 - 1.0, y1 - 1.0, x2, y2],   # VOC is 1-based
                     "difficult": bool(int(d.text)) if d is not None else False})
    return {"width": int(size.find("width").text), "height": int(size.find("height").text), "objects": objs}


def _digest(items) -> str:
    return hashlib.sha1("|".join(map(str, items)).encode()).hexdigest()[:10]


def voc_records(data_dir, set_key: str, ids: list[str] | None = None) -> list[dict]:
    """One record per image: path, size, boxes (px, xyxy), labels (1..20), difficult. Cached next to the data."""
    year, split = set_key.split("_")
    root = voc_root(data_dir, year)
    all_ids = [ln.strip() for ln in (root / "ImageSets" / "Main" / f"{split}.txt").read_text().splitlines() if ln.strip()]
    ids = all_ids if ids is None else ids
    cache = Path(data_dir) / f"index_voc{year}_{split}_{len(ids)}_{_digest(ids)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    cls = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}
    recs = []
    for i in ids:
        a = parse_voc_xml(root / "Annotations" / f"{i}.xml")
        recs.append({"id": f"{year}:{i}", "path": str(root / "JPEGImages" / f"{i}.jpg"), "width": a["width"],
                     "height": a["height"],
                     "boxes": np.asarray([o["box"] for o in a["objects"]], np.float32).reshape(-1, 4),
                     "labels": np.asarray([cls[o["name"]] for o in a["objects"]], np.int64),
                     "difficult": np.asarray([o["difficult"] for o in a["objects"]], bool)})
    cache.write_bytes(pickle.dumps(recs))
    return recs


def coco_records(data_dir, split: str) -> tuple[list[dict], list[str]]:
    """Compact per-image records from instances_<split>2017.json (the 450 MB JSON is parsed once, then cached)."""
    root = coco_root(data_dir)
    cache = Path(data_dir) / f"index_coco_{split}2017.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ann = json.loads((root / "annotations" / f"instances_{split}2017.json").read_text())
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    cid = {c["id"]: i + 1 for i, c in enumerate(cats)}
    per: dict[int, list] = {}
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        if w < 1 or h < 1:
            continue
        per.setdefault(a["image_id"], []).append((x, y, x + w, y + h, cid[a["category_id"]], bool(a.get("iscrowd", 0))))
    recs = []
    for im in sorted(ann["images"], key=lambda i: i["id"]):
        objs = per.get(im["id"])
        if not objs:
            continue
        arr = np.asarray([o[:4] for o in objs], np.float32)
        recs.append({"id": f"coco:{im['id']}", "path": str(root / f"{split}2017" / im["file_name"]),
                     "width": im["width"], "height": im["height"], "boxes": arr,
                     "labels": np.asarray([o[4] for o in objs], np.int64),
                     "difficult": np.asarray([o[5] for o in objs], bool)})   # crowd regions are ignored like 'difficult'
    names = [c["name"] for c in cats]
    cache.write_bytes(pickle.dumps((recs, names)))
    return recs, names


class RecordSource:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def get(self, i):
        r = self.records[i]
        return Image.open(r["path"]).convert("RGB"), r["boxes"], r["labels"], r["difficult"]


class SyntheticSource:
    """Coloured rectangles on noise; class = colour. Deterministic per index. Offline tests only."""

    def __init__(self, n, num_classes, image_size=128, seed=0):
        self.n, self.k, self.size, self.seed = n, num_classes, image_size, seed
        self.classes = [f"c{i}" for i in range(num_classes)]
        self.colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.9, 0.95))
                       for h in np.linspace(0, 1, num_classes, endpoint=False)]

    def __len__(self):
        return self.n

    def get(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        s = self.size
        img = Image.fromarray(rng.integers(60, 130, (s, s, 3), dtype=np.uint8))
        draw, boxes, labels = ImageDraw.Draw(img), [], []
        for _ in range(int(rng.integers(1, 3))):
            w, h = rng.integers(s // 4, s // 2, 2)
            x, y = rng.integers(0, s - w), rng.integers(0, s - h)
            c = int(rng.integers(0, self.k))
            draw.rectangle([int(x), int(y), int(x + w) - 1, int(y + h) - 1], fill=self.colors[c])
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(c + 1)
        return (img, np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64),
                np.zeros(len(labels), bool))


def split_ids(ids, val_fraction: float, seed: int):
    """Deterministic random split into (train, val)."""
    ids = sorted(ids)
    if val_fraction <= 0:
        return ids, []
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return sorted(ids[i] for i in perm[n_val:]), sorted(ids[i] for i in perm[:n_val])


# ------------------------------------------------------------------ augmentation (paper section 4), numpy + PIL, torch RNG
def _rand() -> float:
    return float(torch.rand(1))


def _randint(lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (inclusive)."""
    return int(torch.randint(lo, hi + 1, (1,)))


def hflip(img, boxes):
    w = img.width
    b = boxes.copy()
    if len(b):
        b[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img.transpose(Image.FLIP_LEFT_RIGHT), b


def resize_to(img, boxes, size, max_size):
    oh, ow = aspect_size(img.width, img.height, size, max_size)
    sx, sy = ow / img.width, oh / img.height
    return img.resize((ow, oh), Image.BILINEAR), boxes * np.array([sx, sy, sx, sy], dtype=np.float32)


def random_size_crop(img, boxes, labels, lo, hi):
    """Random rectangular patch; boxes are clipped to it and boxes that vanish are dropped."""
    w = _randint(min(lo, img.width), min(hi, img.width))
    h = _randint(min(lo, img.height), min(hi, img.height))
    x0, y0 = _randint(0, img.width - w), _randint(0, img.height - h)
    out = boxes - np.array([x0, y0, x0, y0], dtype=np.float32)
    out = np.clip(out, 0, [w, h, w, h]).astype(np.float32)
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1]) if len(out) else np.zeros(0, bool)
    return img.crop((x0, y0, x0 + w, y0 + h)), out[keep], labels[keep]


def train_augment(img, boxes, labels, aug: dict):
    """Horizontal flip; then with probability 1 - crop_prob a plain random resize, otherwise resize -> random crop -> random
    resize, exactly the official DETR recipe (scales 480..800, crop 384..600)."""
    if aug.get("flip", True) and _rand() < 0.5:
        img, boxes = hflip(img, boxes)
    scales = aug["scales"]
    if _rand() >= aug.get("crop_prob", 0.5):
        img, boxes = resize_to(img, boxes, scales[_randint(0, len(scales) - 1)], aug["max_size"])
    else:
        cr = aug["crop_resize"]
        img, boxes = resize_to(img, boxes, cr[_randint(0, len(cr) - 1)], aug["max_size"])
        img, boxes, labels = random_size_crop(img, boxes, labels, aug["crop_min"], aug["crop_max"])
        img, boxes = resize_to(img, boxes, scales[_randint(0, len(scales) - 1)], aug["max_size"])
    return img, boxes, labels


def normalise_boxes(boxes: np.ndarray, w: int, h: int) -> torch.Tensor:
    """xyxy pixels -> (cx, cy, w, h) relative to the image size, clipped to [0, 1]."""
    b = torch.from_numpy(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4)
    out = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h, (b[:, 2] - b[:, 0]) / w,
                       (b[:, 3] - b[:, 1]) / h], dim=1)
    return out.clamp(0, 1)


class DetrDataset(Dataset):
    def __init__(self, source, aug: dict, train: bool, include_difficult: bool = True):
        self.source, self.aug, self.train, self.include_difficult = source, aug, train, include_difficult

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        img, boxes, labels, difficult = self.source.get(i)
        if self.train:
            if not self.include_difficult:
                keep = ~difficult
                boxes, labels = boxes[keep], labels[keep]
            img, boxes, labels = train_augment(img, boxes, labels, self.aug)
            return to_normalised_tensor(img), {"boxes": normalise_boxes(boxes, img.width, img.height),
                                               "labels": torch.from_numpy(labels - 1).long()}
        w, h = img.size
        small, _ = resize_to(img, boxes, self.aug["test_size"], self.aug["max_size"])
        meta = {"boxes": boxes, "labels": labels, "difficult": difficult, "size": (w, h), "index": i}
        return (to_normalised_tensor(small), {"boxes": normalise_boxes(boxes, w, h), "labels": torch.from_numpy(labels - 1).long()},
                meta)


def pad_batch(images: list[torch.Tensor]):
    """Zero-pad to the largest image; mask is True on the padding."""
    h, w = max(i.shape[1] for i in images), max(i.shape[2] for i in images)
    batch = torch.zeros(len(images), 3, h, w)
    mask = torch.ones(len(images), h, w, dtype=torch.bool)
    for k, im in enumerate(images):
        batch[k, :, : im.shape[1], : im.shape[2]] = im
        mask[k, : im.shape[1], : im.shape[2]] = False
    return batch, mask


def collate_train(batch):
    images, mask = pad_batch([b[0] for b in batch])
    return images, mask, [b[1] for b in batch]


def collate_eval(batch):
    images, mask = pad_batch([b[0] for b in batch])
    return images, mask, [b[1] for b in batch], [b[2] for b in batch]


def build_data(cfg: dict, paths: dict) -> dict:
    d, seed = cfg["data"], cfg["seed"]
    rng = np.random.default_rng(seed)
    aug = cfg["aug"]
    if d["dataset"] == "voc2012":
        keys = list(d["voc_train_sets"]) + ([d["test_set"]] if d.get("test_set") else [])
        ensure_voc(paths["data"], paths["tmp"], keys, d.get("keep_archives", False), d.get("download", True))
        recs = [r for k in d["voc_train_sets"] for r in voc_records(paths["data"], k)]
        by_id = {r["id"]: r for r in recs}
        tr_ids, va_ids = split_ids(list(by_id), d["val_fraction"], seed)
        if d.get("subset_train"):
            tr_ids = sorted(rng.permutation(tr_ids)[: d["subset_train"]].tolist())
        train_recs, val_recs = [by_id[i] for i in tr_ids], [by_id[i] for i in va_ids]
        test_recs = voc_records(paths["data"], d["test_set"]) if d.get("test_set") else None
        classes, mk, inc = list(VOC_CLASSES), RecordSource, d.get("include_difficult_train", True)
    elif d["dataset"] == "coco2017":
        ensure_coco(paths["data"], paths["tmp"], ("train", "val"), d.get("keep_archives", False), d.get("download", True))
        train_recs, classes = coco_records(paths["data"], "train")
        val_all, _ = coco_records(paths["data"], "val")
        pick = sorted(rng.permutation(len(val_all))[: d["coco_val_images"]]) if d.get("coco_val_images") else range(len(val_all))
        val_recs, test_recs, mk, inc = [val_all[i] for i in pick], None, RecordSource, False
    else:
        s = d["synthetic"]
        k, n = int(s["classes"]), int(s["size"])
        n_val = max(2, int(n * d["val_fraction"]))
        train_recs, val_recs = SyntheticSource(n - n_val, k, s["image_size"], seed), SyntheticSource(n_val, k, s["image_size"], seed + 1)
        test_recs, classes = SyntheticSource(max(4, n // 4), k, s["image_size"], seed + 2), train_recs.classes
        mk, inc = (lambda x: x), True
    n_eval = min(len(train_recs), int(d.get("train_eval_images", 0)))
    train_src = mk(train_recs)
    out = {"classes": classes, "train": DetrDataset(train_src, aug, True, inc),
           "train_eval": DetrDataset(SubsetSource(train_src, n_eval), aug, False) if n_eval else None,
           "val": DetrDataset(mk(val_recs), aug, False) if len(val_recs) else None,
           "test": DetrDataset(mk(test_recs), aug, False) if test_recs is not None and len(test_recs) else None}
    ids = [train_src.records[i].get("id", i) for i in range(min(len(train_src), 5000))] if hasattr(train_src, "records") else [len(train_src)]
    out["data_version"] = f"{d['dataset']}-tr{len(train_src)}-va{len(val_recs)}-te{0 if test_recs is None else len(test_recs)}-{_digest(ids)}"
    return out


class SubsetSource:
    """First n items of another source (a fixed, un-augmented subset of the training images)."""

    def __init__(self, source, n):
        self.source, self.n = source, n

    def __len__(self):
        return self.n

    def get(self, i):
        return self.source.get(i)


def build_loaders(cfg: dict, data: dict, device: torch.device) -> dict:
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    nw, pin = int(d["num_workers"]), device.type == "cuda"

    def mk(ds, bs, shuffle, collate, workers, drop_last=False):
        kw = dict(num_workers=workers, collate_fn=collate, pin_memory=pin, persistent_workers=workers > 0)
        if workers > 0:
            kw["prefetch_factor"] = int(d.get("prefetch_factor", 4))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last, generator=g if shuffle else None, **kw)
    ev_bs, ev_nw = int(d["batch_size"]) * 2, min(nw, 8)
    out = {"train": mk(data["train"], d["batch_size"], True, collate_train, nw, drop_last=len(data["train"]) > d["batch_size"])}
    for name in ("val", "test", "train_eval"):
        out[name] = mk(data[name], ev_bs, False, collate_eval, ev_nw) if data.get(name) is not None else None
    return out
