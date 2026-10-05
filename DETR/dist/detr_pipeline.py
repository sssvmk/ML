#!/usr/bin/env python3
"""DETR (Carion et al., ECCV 2020) with an ImageNet-pretrained ResNet-50: the complete pipeline in one file.

  python detr_pipeline.py PATH [cpu|gpu]

PATH holds everything (data, caches, runs, models, MLflow). The device is asked for when omitted.
Generated from the package by tools/build_script.py; do not edit.
"""

# ====================================================================================================
# config.py
# ====================================================================================================
"""Configuration: the paper's recipe as the default, stage overrides, dotted overrides."""
import copy
import os

import yaml

DEFAULT_YAML = """
# DETR with an ImageNet-pretrained ResNet-50 (Carion et al., 2020), single A100 + 24 vCPUs.
# Every output, cache and temporary file lives under `dest` (a parameter, never hard-coded).
seed: 42

data:
  dataset: voc2012                 # voc2012 | coco2017 (the paper's dataset) | synthetic (offline tests only)
  voc_train_sets: ["2012_trainval"]
  test_set: "2007_test"            # untouched until `test`
  val_fraction: 0.10               # held-out part of the training set: early stopping and model selection
  coco_val_images: 2000
  subset_train: null               # int -> only N training images (debugging)
  include_difficult_train: true
  train_eval_images: 300           # un-augmented train images scored at every evaluation (train/val gap)
  batch_size: 16                   # paper: 64 (16 GPUs x 4); one A100 holds 16 with bf16
  num_workers: 20
  prefetch_factor: 4
  download: true
  keep_archives: false
  synthetic: {size: 64, classes: 3, image_size: 128}

model:
  backbone: resnet50               # resnet18 | resnet34 | resnet50 | resnet101
  pretrained: imagenet             # ImageNet weights from torchvision (cached under dest) | none
  dilation: false                  # true = DETR-DC5 (dilated last stage, 2x compute)
  freeze_bn: true                  # paper: frozen BatchNorm
  hidden_dim: 256                  # paper: d = 256
  nheads: 8
  enc_layers: 6
  dec_layers: 6
  dim_feedforward: 2048
  dropout: 0.1                     # paper: dropout 0.1 after every attention and FFN
  num_queries: 100                 # paper: N = 100
  aux_loss: true                   # paper: auxiliary decoding losses after every decoder layer

loss:
  cost_class: 1.0                  # matching costs (paper appendix A.4 / official code)
  cost_bbox: 5.0
  cost_giou: 2.0
  w_ce: 1.0                        # loss weights: class 1, L1 5, GIoU 2
  w_bbox: 5.0
  w_giou: 2.0
  eos_coef: 0.1                    # paper: down-weight the "no object" class by 10

aug:                               # paper section 4: scale augmentation + random crop (p = 0.5)
  scales: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
  max_size: 1333
  crop_prob: 0.5
  crop_resize: [400, 500, 600]
  crop_min: 384
  crop_max: 600
  flip: true
  test_size: 800

postprocess:
  max_detections: 100              # every query yields one detection; no NMS

schedule:                          # epochs; paper ablation schedule: 300 epochs, LR x0.1 after 200
  phases:
    - {lr: 0.0001, epochs: 200}
    - {lr: 0.00001, epochs: 100}
  scale: 1.0
  lr_backbone: 0.00001             # paper: backbone 1e-5, transformer 1e-4
  weight_decay: 0.0001             # paper: AdamW, 1e-4
  grad_clip: 0.1                   # paper: max gradient norm 0.1
  amp: bf16                        # bf16 | fp16 | none
  eval_every: 2                    # epochs between validation passes
  log_every: 20
  early_stopping:
    enabled: true
    patience: 8                    # evaluations without improvement inside the current LR phase
    min_delta: 0.002
    advance_phase_on_plateau: true # plateau -> next (lower) LR early; plateau in the last phase -> stop
    restore_best_on_decay: true
  ema:
    enabled: false                 # not in the paper; optional Polyak averaging
    decay: 0.9999
    tau: 2000
  overfit_gap_warn: 0.20

metric:
  name: map50                      # all-point AP at IoU 0.5 (VOC2012 protocol)
  target_value: 0.50               # ASSUMED, not confirmed
  higher_is_better: true

sanity:
  enabled: true
  initial_loss_tol: 0.30           # relative tolerance around ln(num_classes + 1) for the class loss
  overfit_images: 4
  overfit_steps: 80
  overfit_lr: 0.0003
  overfit_ratio: 0.85              # tiny-batch loss must fall below this fraction of its start
  fail_hard: true

mlflow:
  backend: auto
  experiment_name: detr-pipeline
  registered_model_name: null

resume_from: null
device: auto
"""

COCO = {"data.dataset": "coco2017", "data.test_set": None,
        "schedule.phases": [{"lr": 0.0001, "epochs": 400}, {"lr": 0.00001, "epochs": 100}],   # paper: 500 epochs, drop after 400
        "schedule.eval_every": 5}
STAGES = {
    "voc": ("DETR on VOC2012 from the ImageNet-pretrained ResNet (paper recipe), test on VOC2007 test", {}),
    "coco": ("DETR on COCO train2017 from the ImageNet-pretrained ResNet (the paper's dataset and 500-epoch schedule)", COCO),
}


def set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_overrides(items) -> dict:
    out = {}
    if isinstance(items, dict):
        return dict(items)
    for item in items or []:
        key, sep, raw = str(item).partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=value, got: {item}")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(stage: str = "voc", overrides=None, dest=None) -> dict:
    """DEFAULT_YAML -> stage overrides -> user overrides (list of 'a.b=value' strings or a dict)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
    cfg = yaml.safe_load(DEFAULT_YAML)
    for k, v in copy.deepcopy(STAGES[stage][1]).items():
        set_dotted(cfg, k, v)
    for k, v in parse_overrides(overrides).items():
        set_dotted(cfg, k, v)
    cfg["stage"] = stage
    if dest is not None:
        cfg["dest"] = str(dest)
    return resolve(cfg)


def resolve(cfg: dict) -> dict:
    d = cfg["data"]
    if d["dataset"] == "voc2012":
        d["num_foreground"] = 20
    elif d["dataset"] == "coco2017":
        d["num_foreground"] = 80
    elif d["dataset"] == "synthetic":
        d["num_foreground"] = int(d["synthetic"]["classes"])
    else:
        raise ValueError(f"unknown dataset {d['dataset']!r}; choose voc2012, coco2017 or synthetic")
    if cfg["model"]["hidden_dim"] % cfg["model"]["nheads"]:
        raise ValueError("model.hidden_dim must be divisible by model.nheads")
    return cfg


def registered_name(cfg: dict) -> str:
    return cfg["mlflow"].get("registered_model_name") or f"detr-{cfg['model']['backbone']}-{cfg['data']['dataset']}"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# ====================================================================================================
# env.py
# ====================================================================================================
"""Cache control: every cache, temporary and output path is placed under the destination folder."""
import os
import sys
import tempfile
from pathlib import Path


def make_paths(dest) -> dict:
    d = Path(str(dest)[len("dbfs:"):] if str(dest).startswith("dbfs:/") else dest)
    return {"dest": d, "data": d / "data", "runs": d / "runs", "checkpoints": d / "checkpoints", "tmp": d / "tmp",
            "mlflow": d / "mlflow", "mlflow_artifacts": d / "mlflow" / "artifacts",
            "cache": d / "cache", "torch_home": d / "cache" / "torch", "mpl": d / "cache" / "matplotlib",
            "cuda_cache": d / "cache" / "cuda", "xdg": d / "cache" / "xdg", "pycache": d / "cache" / "pycache"}


def setup_environment(dest) -> dict:
    """Create the folder layout and point every library cache at it. Call once, before heavy imports if possible."""
    p = make_paths(dest)
    if str(p["dest"]).startswith("/Volumes/"):
        parts = p["dest"].parts
        if len(parts) < 5:
            raise ValueError("a Volume path must look like /Volumes/<catalog>/<schema>/<volume>/<folder>")
        if not Path(*parts[:5]).exists():
            raise FileNotFoundError(f"{Path(*parts[:5])} does not exist: create the volume first (CREATE VOLUME ...)")
    for k, path in p.items():
        if k != "mlflow_artifacts":
            path.mkdir(parents=True, exist_ok=True)
    p["mlflow_artifacts"].mkdir(parents=True, exist_ok=True)
    env = {"TORCH_HOME": p["torch_home"], "XDG_CACHE_HOME": p["xdg"], "MPLCONFIGDIR": p["mpl"],
           "CUDA_CACHE_PATH": p["cuda_cache"], "TMPDIR": p["tmp"], "TEMP": p["tmp"], "TMP": p["tmp"],
           "TRITON_CACHE_DIR": p["cache"] / "triton", "HF_HOME": p["cache"] / "huggingface"}
    for k, v in env.items():
        os.environ[k] = str(v)
    tempfile.tempdir = str(p["tmp"])
    sys.pycache_prefix = str(p["pycache"])
    try:
        import torch
        torch.hub.set_dir(str(p["torch_home"]))
    except ImportError:
        pass
    return p

# ====================================================================================================
# data_download.py
# ====================================================================================================
"""Robust, resumable downloads of Pascal VOC and COCO into <dest>/data. Staging happens under <dest>/tmp only."""
import json
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

log = logging.getLogger("ssd_voc.download")

COCO_BASE = "http://images.cocodataset.org"
URLS = {
    "voc2012_trainval": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "voc2007_trainval": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "voc2007_test": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    "coco_val_images": f"{COCO_BASE}/zips/val2017.zip",
    "coco_train_images": f"{COCO_BASE}/zips/train2017.zip",
    "coco_annotations": f"{COCO_BASE}/annotations/annotations_trainval2017.zip",
}
VOC_SETS = {"2012_trainval": ("voc2012_trainval", "VOC2012", "trainval"),
            "2007_trainval": ("voc2007_trainval", "VOC2007", "trainval"),
            "2007_test": ("voc2007_test", "VOC2007", "test")}


def get_url(key: str) -> str:
    """Override with DATASET_URL_<KEY> (e.g. an internal mirror)."""
    return os.environ.get(f"DATASET_URL_{key.upper()}", URLS[key])


def _human(n: float) -> str:
    return f"{n / 1e6:,.0f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def ensure_space(path: Path, needed_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    if free < needed_gb:
        raise OSError(f"only {free:.1f} GB free under {path}, need about {needed_gb:.1f} GB")


def download(url: str, out, *, retries: int = 6, chunk: int = 1 << 20, timeout: int = 60,
             session: requests.Session | None = None, log_every: float = 15.0) -> Path:
    """Resumable download with retries; a finished file of the expected size is not downloaded again."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    total = None
    try:
        h = s.head(url, allow_redirects=True, timeout=timeout)
        if h.ok and "Content-Length" in h.headers:
            total = int(h.headers["Content-Length"])
    except requests.RequestException:
        pass
    if out.exists() and out.stat().st_size > 0 and (total is None or out.stat().st_size == total):
        return out
    part = out.with_name(out.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with s.get(url, stream=True, timeout=timeout, allow_redirects=True,
                       headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:
                    if total is not None and have == total:
                        break
                    part.unlink(missing_ok=True)
                    continue
                r.raise_for_status()
                resumed = bool(have) and r.status_code == 206
                if total is None:
                    cr = r.headers.get("Content-Range")
                    total = int(cr.rsplit("/", 1)[1]) if cr and "/" in cr else (
                        int(r.headers["Content-Length"]) + (have if resumed else 0) if "Content-Length" in r.headers
                        else None)
                done = have if resumed else 0
                last = time.time()
                with open(part, "ab" if resumed else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        done += len(block)
                        if time.time() - last >= log_every:
                            last = time.time()
                            log.info("  %s: %s%s", out.name, _human(done), f" ({100 * done / total:.0f}%)" if total else "")
            if total is not None and part.stat().st_size != total:
                raise OSError(f"incomplete download: {part.stat().st_size} of {total} bytes")
            break
        except (requests.RequestException, OSError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("download attempt %d/%d failed (%s); retrying in %ds", attempt, retries, e, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not download {url} after {retries} attempts: {last_err}. If the host is blocked from "
                           "this workspace, allow it, or set DATASET_URL_<KEY> to an internal mirror.")
    part.replace(out)
    log.info("downloaded %s (%s)", out.name, _human(out.stat().st_size))
    return out


def _inside(base: Path, target: Path) -> bool:
    return str(target.resolve()).startswith(str(base.resolve()) + os.sep)


def extract_tar(tpath, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tpath) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(out_dir, filter="data")
        else:
            for m in t.getmembers():
                if not _inside(out_dir, out_dir / m.name):
                    raise ValueError(f"unsafe path in archive: {m.name}")
            t.extractall(out_dir)


def extract_zip(zpath, out_dir, members: list[str] | None = None) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = members if members is not None else z.namelist()
        have = set(z.namelist())
        missing = [m for m in names if m not in have]
        if missing:
            raise KeyError(f"{missing[0]} not found in {Path(zpath).name}")
        for n in names:
            if not _inside(out_dir, out_dir / n):
                raise ValueError(f"unsafe path in archive: {n}")
        z.extractall(out_dir, names)
    return names


def voc_root(data_dir, year: str) -> Path:
    return Path(data_dir) / "VOCdevkit" / f"VOC{year}"


def voc_available(data_dir, key: str) -> bool:
    _, folder, split = VOC_SETS[key]
    root = Path(data_dir) / "VOCdevkit" / folder
    return (root / "Annotations").is_dir() and (root / "JPEGImages").is_dir() and \
        (root / "ImageSets" / "Main" / f"{split}.txt").exists()


def ensure_voc(data_dir, tmp_dir, keys, keep_archives=False, allow_download=True) -> dict:
    """Make sure the requested VOC sets exist under data_dir (download + extract if needed)."""
    done = {}
    for key in keys:
        if key not in VOC_SETS:
            raise ValueError(f"unknown VOC set {key!r}; choose from {sorted(VOC_SETS)}")
        if voc_available(data_dir, key):
            done[key] = "present"
            continue
        if not allow_download:
            raise FileNotFoundError(f"VOC set {key} not found under {data_dir} and data.download is false")
        url_key = VOC_SETS[key][0]
        ensure_space(Path(tmp_dir), 4.0)
        arc = download(get_url(url_key), Path(tmp_dir) / Path(get_url(url_key)).name)
        extract_tar(arc, data_dir)
        if not keep_archives:
            arc.unlink(missing_ok=True)
        if not voc_available(data_dir, key):
            raise RuntimeError(f"{key} was extracted but is incomplete under {data_dir}")
        done[key] = "downloaded"
    return done


def coco_root(data_dir) -> Path:
    return Path(data_dir) / "coco2017"


def ensure_coco(data_dir, tmp_dir, splits=("val",), keep_archives=False, allow_download=True) -> dict:
    """COCO 2017: annotations (instances only) plus val2017 and/or train2017 images (train is ~18 GB)."""
    root = coco_root(data_dir)
    done = {}
    for split in splits:
        ann = root / "annotations" / f"instances_{split}2017.json"
        imgs = root / f"{split}2017"
        marker = root / f".complete_{split}2017"
        if ann.exists() and marker.exists():
            done[split] = "present"
            continue
        if not allow_download:
            raise FileNotFoundError(f"COCO {split}2017 not found under {root} and data.download is false")
        ensure_space(Path(tmp_dir), 22.0 if split == "train" else 3.0)
        if not ann.exists():
            arc = download(get_url("coco_annotations"), Path(tmp_dir) / "annotations_trainval2017.zip")
            extract_zip(arc, root, [f"annotations/instances_{split}2017.json"])
            if not keep_archives:
                arc.unlink(missing_ok=True)
        arc = download(get_url(f"coco_{split}_images"), Path(tmp_dir) / f"{split}2017.zip", timeout=120)
        extract_zip(arc, root)
        if not keep_archives:
            arc.unlink(missing_ok=True)
        n = sum(1 for _ in imgs.glob("*.jpg"))
        marker.write_text(json.dumps({"images": n, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        done[split] = f"downloaded ({n} images)"
    return done

# ====================================================================================================
# ops.py
# ====================================================================================================
"""Box operations and the Hungarian assignment (scipy when it works, a pure numpy solver otherwise)."""
import numpy as np
import torch
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def giou_matrix(a_xyxy: torch.Tensor, b_xyxy: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (paper eq. 10) between every box of a and every box of b."""
    return generalized_box_iou(a_xyxy, b_xyxy)


def hungarian_numpy(cost: np.ndarray):
    """Minimum-cost assignment for a rectangular cost matrix (O(n^2 m), potentials method). Returns (rows, cols) like scipy."""
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    c = cost.T if transposed else cost
    n, m = c.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    inf = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=np.int64), np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


_SCIPY = {}


def solve_assignment(cost: np.ndarray):
    """scipy's linear_sum_assignment if scipy imports cleanly (a cluster with a numpy/scipy mismatch raises on import),
    otherwise the numpy solver above. Same optimal cost either way."""
    if "fn" not in _SCIPY:
        try:
            from scipy.optimize import linear_sum_assignment
            _SCIPY["fn"] = linear_sum_assignment
        except Exception:  # noqa: BLE001 - broken scipy installs raise AttributeError, not ImportError
            _SCIPY["fn"] = hungarian_numpy
    return _SCIPY["fn"](cost)

# ====================================================================================================
# model.py
# ====================================================================================================
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

# ====================================================================================================
# infer.py
# ====================================================================================================
"""Inference helpers: image preprocessing, post-processing (no NMS) and a checkpoint predictor."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def aspect_size(w: int, h: int, size: int, max_size: int | None) -> tuple[int, int]:
    """(height, width) after resizing the shorter side to `size`, the longer side capped at max_size (official DETR rule)."""
    if max_size is not None:
        lo, hi = float(min(w, h)), float(max(w, h))
        if hi / lo * size > max_size:
            size = int(round(max_size * lo / hi))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def to_normalised_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - m) / s


def denormalize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)


@torch.no_grad()
def postprocess(outputs: dict, sizes: torch.Tensor) -> list[dict]:
    """Paper sec. 4: every query yields one detection. Its class is the best REAL class (the 'no object' slot is overridden
    by the second-highest class, which the paper reports as +2 AP) and its score is that class's probability.
    sizes: [B, 2] original (width, height). Boxes are xyxy in original pixels; labels are 1..K."""
    prob = F.softmax(outputs["pred_logits"].float(), -1)[..., :-1]
    scores, labels = prob.max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"].float())
    w, h = sizes[:, 0].float(), sizes[:, 1].float()
    boxes = boxes * torch.stack([w, h, w, h], dim=1)[:, None, :].to(boxes.device)
    return [{"scores": s, "labels": lb + 1, "boxes": b} for s, lb, b in zip(scores, labels, boxes, strict=True)]


def load_checkpoint(path, device="cpu"):
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(st["state_dict"])
    return model.to(device).eval(), cfg, st["classes"]


class Predictor:
    """Run a trained checkpoint on images (paths, PIL images or HxWx3 uint8 arrays). No MLflow, no downloads."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.classes = load_checkpoint(checkpoint, self.device)

    @staticmethod
    def _pil(item) -> Image.Image:
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("RGB")
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        arr = np.asarray(item)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("array input must be uint8 with shape (H, W, 3)")
        return Image.fromarray(arr)

    @torch.inference_mode()
    def predict(self, images, score_threshold=0.5, max_detections=100) -> list[dict]:
        aug = self.cfg["aug"]
        out = []
        for item in images:
            img = self._pil(item)
            w, h = img.size
            oh, ow = aspect_size(w, h, aug["test_size"], aug["max_size"])
            x = to_normalised_tensor(img.resize((ow, oh), Image.BILINEAR))[None].to(self.device)
            res = postprocess(self.model(x, torch.zeros(1, oh, ow, dtype=torch.bool, device=self.device)),
                              torch.tensor([[w, h]]))[0]
            order = res["scores"].argsort(descending=True)[:max_detections]
            dets = [{"box": [round(float(v), 2) for v in res["boxes"][i]], "score": round(float(res["scores"][i]), 5),
                     "label_id": int(res["labels"][i]), "label": self.classes[int(res["labels"][i]) - 1]}
                    for i in order.tolist() if float(res["scores"][i]) >= score_threshold]
            out.append({"width": w, "height": h, "detections": dets})
        return out

# ====================================================================================================
# data.py
# ====================================================================================================
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

# ====================================================================================================
# loss.py
# ====================================================================================================
"""Hungarian matching and the set-prediction loss (paper section 3.1): L = class CE + 5 * L1 + 2 * GIoU per matched pair."""
import torch
import torch.nn as nn
import torch.nn.functional as F



class HungarianMatcher(nn.Module):
    """One-to-one assignment of predictions to ground truth minimising  w_c * (-p(class)) + w_b * L1 + w_g * (-GIoU)."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        bs, nq = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        if sum(sizes) == 0:
            e = torch.zeros(0, dtype=torch.int64)
            return [(e, e)] * bs
        prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)
        boxes = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([t["labels"] for t in targets]).to(prob.device)
        tgt_boxes = torch.cat([t["boxes"] for t in targets]).to(boxes.device).float()
        cost = (self.cost_bbox * torch.cdist(boxes, tgt_boxes, p=1) - self.cost_class * prob[:, tgt_ids]
                - self.cost_giou * giou_matrix(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        cost = cost.view(bs, nq, -1).cpu()
        out = []
        for i, c in enumerate(cost.split(sizes, -1)):
            r, col = solve_assignment(c[i].numpy())
            out.append((torch.as_tensor(r, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return out


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weights: dict, eos_coef: float):
        super().__init__()
        self.num_classes, self.matcher, self.weights = num_classes, matcher, weights
        w = torch.ones(num_classes + 1)
        w[-1] = eos_coef                                   # paper: "no object" down-weighted by a factor 10
        self.register_buffer("empty_weight", w, persistent=False)

    def _single(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"].float()
        idx = (torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)]), torch.cat([s for s, _ in indices]))
        classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
        classes[idx] = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices, strict=True)]).to(logits.device)
        loss_ce = F.cross_entropy(logits.transpose(1, 2), classes, self.empty_weight)
        src = outputs["pred_boxes"].float()[idx]
        tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices, strict=True)]).to(src.device).float()
        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / num_boxes
        giou = torch.diag(giou_matrix(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt))) if len(src) else src.sum() * 0
        loss_giou = (1 - giou).sum() / num_boxes if len(src) else src.sum() * 0
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        """Returns the weighted total plus every component (and the per-decoder-layer auxiliary losses)."""
        num_boxes = max(float(sum(len(t["labels"]) for t in targets)), 1.0)
        main = self._single(outputs, targets, self.matcher(outputs, targets), num_boxes)
        parts = dict(main)
        total = sum(self.weights[k] * v for k, v in main.items())
        for n, aux in enumerate(outputs.get("aux_outputs", [])):
            a = self._single(aux, targets, self.matcher(aux, targets), num_boxes)
            parts.update({f"{k}_{n}": v for k, v in a.items()})
            total = total + sum(self.weights[k] * v for k, v in a.items())
        return {"total": total, **{k: v.detach() for k, v in parts.items()}}

# ====================================================================================================
# metrics.py
# ====================================================================================================
"""Detection metrics: VOC-protocol AP (11-point VOC07 and all-point), COCO-style mAP@[.5:.95], AR@100, error breakdown.

Matching follows the Pascal VOC devkit: detections are sorted by score; each takes the ground-truth box of its class
with the highest IoU; if IoU >= threshold the detection is a true positive unless that GT was already matched
(duplicate -> false positive); 'difficult' GTs are ignored (neither TP nor FP, and not counted as positives).
"""

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for XYXY boxes (continuous coordinates, no +1)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def ap_all_points(rec: np.ndarray, prec: np.ndarray) -> float:
    """Area under the monotone precision envelope (VOC2010+ / COCO-like)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_11_point(rec: np.ndarray, prec: np.ndarray) -> float:
    """Official Pascal VOC2007 metric: mean of max precision at recall >= 0, 0.1, ..., 1.0."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return float(ap)


class DetectionEvaluator:
    def __init__(self, class_names: list[str], iou_thresholds=None):
        """iou_thresholds=[0.5] gives the fast mode used for periodic validation (map75 / map are then None)."""
        self.thresholds = np.round(np.asarray(IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds, float), 2)
        self.class_names = list(class_names)
        self.k = len(class_names)
        self.preds: list[dict] = []
        self.gts: list[dict] = []

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        """preds: dicts with boxes[N,4], scores[N], labels[N] (1..K); targets: boxes, labels, optional difficult."""
        for p, t in zip(preds, targets, strict=True):
            self.preds.append({"boxes": np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4),
                               "scores": np.asarray(p["scores"], dtype=np.float64).reshape(-1),
                               "labels": np.asarray(p["labels"], dtype=np.int64).reshape(-1)})
            n = len(np.asarray(t["labels"]))
            diff = np.asarray(t["difficult"], dtype=bool) if "difficult" in t else np.zeros(n, dtype=bool)
            self.gts.append({"boxes": np.asarray(t["boxes"], dtype=np.float64).reshape(-1, 4),
                             "labels": np.asarray(t["labels"], dtype=np.int64).reshape(-1), "difficult": diff})

    def _class_curves(self, c: int, thresholds):
        """For class c: per threshold -> (recall, precision) arrays, plus npos."""
        npos = sum(int(np.sum((g["labels"] == c) & ~g["difficult"])) for g in self.gts)
        dets = []  # (score, img, det_idx_in_class)
        per_img = []
        for i, (p, g) in enumerate(zip(self.preds, self.gts, strict=True)):
            m = p["labels"] == c
            gm = g["labels"] == c
            boxes_d, scores_d = p["boxes"][m], p["scores"][m]
            iou = box_iou_np(boxes_d, g["boxes"][gm])
            per_img.append((iou, g["difficult"][gm]))
            for j, s in enumerate(scores_d):
                dets.append((s, i, j))
        dets.sort(key=lambda x: -x[0])
        out = {}
        for t in thresholds:
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))
            matched = [np.zeros(iou.shape[1], dtype=bool) for iou, _ in per_img]
            for n, (_, i, j) in enumerate(dets):
                iou, diff = per_img[i]
                if iou.shape[1] == 0:
                    fp[n] = 1
                    continue
                row = iou[j]
                k = int(np.argmax(row))
                if row[k] >= t:
                    if diff[k]:
                        continue  # ignored
                    if not matched[i][k]:
                        tp[n] = 1
                        matched[i][k] = True
                    else:
                        fp[n] = 1
                else:
                    fp[n] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            rec = ctp / max(npos, 1)
            prec = ctp / np.maximum(ctp + cfp, 1e-12)
            out[t] = (rec, prec)
        return out, npos

    def compute(self) -> dict:
        th = self.thresholds
        ap_all = np.full((self.k, len(th)), np.nan)
        ap07 = np.full(self.k, np.nan)
        recall_final = np.full((self.k, len(th)), np.nan)
        for ci in range(self.k):
            curves, npos = self._class_curves(ci + 1, th)
            if npos == 0:
                continue
            for ti, t in enumerate(th):
                rec, prec = curves[t]
                if len(rec) == 0:
                    ap_all[ci, ti], recall_final[ci, ti] = 0.0, 0.0
                    if t == 0.5:
                        ap07[ci] = 0.0
                    continue
                ap_all[ci, ti] = ap_all_points(rec, prec)
                recall_final[ci, ti] = rec[-1]
                if t == 0.5:
                    ap07[ci] = ap_11_point(rec, prec)

        def nm(x):
            return float(np.nanmean(x)) if np.any(~np.isnan(x)) else 0.0

        def at(t):
            idx = np.flatnonzero(th == t)
            return nm(ap_all[:, idx[0]]) if len(idx) else None
        i50 = np.flatnonzero(th == 0.5)
        full = len(th) == len(IOU_THRESHOLDS)
        return {
            "map50_voc07": nm(ap07), "map50": at(0.5), "map75": at(0.75), "map": nm(ap_all) if full else None,
            "mar100": nm(recall_final) if full else None,
            "ap50_per_class": {n: (None if (not len(i50) or np.isnan(ap_all[i, i50[0]])) else float(ap_all[i, i50[0]]))
                               for i, n in enumerate(self.class_names)},
            "n_images": len(self.gts), "n_gt": int(sum(len(g["labels"]) for g in self.gts)),
        }


def error_breakdown(preds, targets, score_thr=0.5, iou_hi=0.5, iou_lo=0.1) -> dict:
    """Hoiem-style error analysis for detections with score >= score_thr (difficult GTs are ignored).

    correct | duplicate | localization (right class, IoU in [lo,hi)) | confusion (other class, IoU >= hi) | background,
    plus 'missed' = non-difficult GTs never matched.
    """
    cnt = dict(correct=0, duplicate=0, localization=0, confusion=0, background=0, missed=0, n_gt=0, n_det=0)
    for p, t in zip(preds, targets, strict=True):
        pb, ps, pl = np.asarray(p["boxes"]).reshape(-1, 4), np.asarray(p["scores"]), np.asarray(p["labels"])
        gb, gl = np.asarray(t["boxes"]).reshape(-1, 4), np.asarray(t["labels"])
        gd = np.asarray(t["difficult"], dtype=bool) if "difficult" in t else np.zeros(len(gl), dtype=bool)
        keep = ps >= score_thr
        order = np.argsort(-ps[keep])
        pb, pl = pb[keep][order], pl[keep][order]
        iou = box_iou_np(pb, gb)
        matched = np.zeros(len(gl), dtype=bool)
        cnt["n_gt"] += int(np.sum(~gd))
        cnt["n_det"] += len(pb)
        for i in range(len(pb)):
            same = np.flatnonzero(gl == pl[i])
            other = np.flatnonzero(gl != pl[i])
            best_same = (same[np.argmax(iou[i, same])], iou[i, same].max()) if len(same) else (None, 0.0)
            best_other = iou[i, other].max() if len(other) else 0.0
            if best_same[1] >= iou_hi:
                k = best_same[0]
                if gd[k]:
                    continue
                if matched[k]:
                    cnt["duplicate"] += 1
                else:
                    matched[k] = True
                    cnt["correct"] += 1
            elif best_same[1] >= iou_lo:
                cnt["localization"] += 1
            elif best_other >= iou_hi:
                cnt["confusion"] += 1
            else:
                cnt["background"] += 1
        cnt["missed"] += int(np.sum(~matched & ~gd))
    n_det = max(cnt["n_det"], 1)
    cnt["precision_at_thr"] = cnt["correct"] / n_det
    cnt["recall_at_thr"] = cnt["correct"] / max(cnt["n_gt"], 1)
    return cnt

# ====================================================================================================
# ema.py
# ====================================================================================================
"""Exponential moving average of weights (Polyak averaging with exponential decay, Deep Learning book 8.7.3)."""

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, tau: float = 500.0):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))  # ramp: early weights are not trustworthy
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]

# ====================================================================================================
# schedule.py
# ====================================================================================================
"""Iteration-based LR phases (the paper's step schedule), linear warmup, and phase-aware early stopping."""
import math


class PhaseSchedule:
    """Paper recipe: constant LR per phase (1e-3, then 1e-4, ...), measured in iterations. A phase can also be
    ended early (plateau) via advance(). Per-group `lr_mult` is respected."""

    def __init__(self, optimizer, phases, scale=1.0, warmup_iters=0, warmup_factor=0.1):
        self.opt = optimizer
        self.phases = [(float(lr), max(1, int(round(n * scale)))) for lr, n in phases]
        self.warmup_iters, self.warmup_factor = int(warmup_iters), float(warmup_factor)
        self.phase, self.it_in_phase, self.global_it = 0, 0, 0
        for g in optimizer.param_groups:
            g.setdefault("lr_mult", 1.0)

    @property
    def total_iters(self) -> int:
        return sum(n for _, n in self.phases)

    @property
    def is_last_phase(self) -> bool:
        return self.phase == len(self.phases) - 1

    def phase_done(self) -> bool:
        return self.it_in_phase >= self.phases[self.phase][1]

    def lr_now(self) -> float:
        base = self.phases[self.phase][0]
        if self.phase == 0 and self.global_it < self.warmup_iters:
            base *= self.warmup_factor + (1.0 - self.warmup_factor) * self.global_it / self.warmup_iters
        return base

    def step(self) -> float:
        lr = self.lr_now()
        for g in self.opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        self.global_it += 1
        self.it_in_phase += 1
        return lr

    def advance(self) -> bool:
        """Move to the next phase; False if this was the last one."""
        if self.is_last_phase:
            return False
        self.phase += 1
        self.it_in_phase = 0
        return True

    def state_dict(self):
        return {"phase": self.phase, "it_in_phase": self.it_in_phase, "global_it": self.global_it}

    def load_state_dict(self, s):
        self.phase, self.it_in_phase, self.global_it = s["phase"], s["it_in_phase"], s["global_it"]


class PhaseEarlyStopper:
    """Tracks the best validation metric (higher is better) and the number of evaluations without improvement in the
    CURRENT phase. `exhausted` tells the trainer to take the next LR step early or, in the last phase, to stop."""

    def __init__(self, patience: int, min_delta: float = 0.0, enabled: bool = True):
        self.patience, self.min_delta, self.enabled = int(patience), float(min_delta), bool(enabled)
        self.best, self.best_it, self.bad = -math.inf, -1, 0

    def update(self, value: float, iteration: int) -> bool:
        """Returns True when `value` is a new best."""
        if value > self.best + self.min_delta:
            self.best, self.best_it, self.bad = value, iteration, 0
            return True
        self.bad += 1
        return False

    @property
    def exhausted(self) -> bool:
        return self.enabled and self.bad >= self.patience

    def reset_phase(self) -> None:
        self.bad = 0

    def state_dict(self):
        return {"best": self.best, "best_it": self.best_it, "bad": self.bad}

    def load_state_dict(self, s):
        self.best, self.best_it, self.bad = s["best"], s["best_it"], s["bad"]

# ====================================================================================================
# curves.py
# ====================================================================================================
"""Learning-curve diagnosis (healthy / overfitting / underfitting / unstable ...) from a metrics CSV.

CSV columns: epoch (any x axis, here the iteration), train_loss, val_loss, optional train_metric, val_metric."""
import csv
import math
import sys


def curve_read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("metrics file is empty")
    needed = {"epoch", "train_loss", "val_loss"}
    missing = needed - set(rows[0].keys())
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return rows


def curve_col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        out.append(float(v) if v not in ("", None) else math.nan)
    return out


def curve_slope(ys):
    """Least-squares slope of ys against index; 0 for fewer than 2 points."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def diagnose_curves(rows, target=None, lower_is_better=False, patience=5):
    tl, vl = curve_col(rows, "train_loss"), curve_col(rows, "val_loss")
    n = len(rows)
    findings, verdicts = [], []

    if any(math.isnan(x) or math.isinf(x) for x in tl + vl):
        return {
            "verdict": ["unstable"],
            "findings": ["NaN or infinite loss found; check learning rate, input scaling, and numerical issues first."],
            "n_points": n,
        }

    best_i = min(range(n), key=lambda i: vl[i])
    since_best = n - 1 - best_i
    final_gap = vl[-1] - tl[-1]
    rel_gap = final_gap / max(abs(tl[-1]), 1e-12)
    tail = max(3, n // 4)
    train_tail_slope = curve_slope(tl[-tail:])
    val_tail_slope = curve_slope(vl[-tail:])
    scale = max(abs(tl[0]), abs(vl[0]), 1e-12)

    info = {
        "n_points": n,
        "best_val_loss": vl[best_i],
        "best_val_index": rows[best_i]["epoch"],
        "final_train_loss": tl[-1],
        "final_val_loss": vl[-1],
        "final_gap": final_gap,
        "relative_gap": rel_gap,
        "points_since_best_val": since_best,
    }

    # Divergence / oscillation
    if tl[-1] > tl[0] * 1.5 or vl[-1] > vl[0] * 1.5 and train_tail_slope > 0:
        verdicts.append("unstable")
        findings.append("Loss increased relative to the start; suspect learning rate too high, bad scaling, or a bug.")
    diffs = [abs(tl[i + 1] - tl[i]) for i in range(n - 1)]
    if n >= 6 and diffs and sum(diffs) / len(diffs) > 0.25 * scale:
        verdicts.append("unstable")
        findings.append("Training loss is oscillating strongly; lower the learning rate or clip gradients.")

    # Flat from the start (possible bug / learning rate far too low)
    if n >= 5 and abs(tl[-1] - tl[0]) < 0.01 * scale and abs(vl[-1] - vl[0]) < 0.01 * scale:
        verdicts.append("not_learning")
        findings.append("Neither loss moved. Run the tiny-batch overfit test and inspect gradients and data flow before tuning.")

    # Overfitting: val best earlier than last, and validation rising while train falls
    overfit = since_best >= patience and val_tail_slope > 0 and train_tail_slope <= 0
    big_gap = rel_gap > 0.3 and final_gap > 0.02 * scale
    if overfit or (big_gap and val_tail_slope >= 0 and train_tail_slope < 0):
        verdicts.append("overfitting")
        findings.append(
            f"Validation loss bottomed at index {rows[best_i]['epoch']} ({since_best} points ago) while train loss kept falling; "
            "use early stopping at the best point, add regularization or data, or reduce capacity."
        )
    elif big_gap:
        verdicts.append("overfitting")
        findings.append(f"Large train/validation gap (relative {rel_gap:.2f}); consider more data or regularization.")

    # Underfitting: small gap, and either target missed or both still high/flat
    metric_info = None
    if "val_metric" in rows[0] and "train_metric" in rows[0]:
        vm, tm = curve_col(rows, "val_metric"), curve_col(rows, "train_metric")
        if not any(math.isnan(x) for x in vm + tm):
            sign = -1.0 if lower_is_better else 1.0
            metric_info = {"final_train_metric": tm[-1], "final_val_metric": vm[-1]}
            if target is not None:
                train_meets = sign * tm[-1] >= sign * target
                val_meets = sign * vm[-1] >= sign * target
                metric_info.update({"target": target, "train_meets_target": train_meets, "val_meets_target": val_meets})
                if not train_meets and "overfitting" not in verdicts:
                    verdicts.append("underfitting")
                    findings.append(
                        "Training metric itself misses the target and there is no large gap; capacity, training time, or features are limiting. "
                        "Rule out a bug or data defect first (worst-error review)."
                    )
                elif train_meets and not val_meets and "overfitting" not in verdicts:
                    verdicts.append("overfitting")
                    findings.append("Training metric meets the target but validation does not: a generalization problem.")
                elif train_meets and val_meets:
                    findings.append("Both training and validation metrics meet the target; confirm once on the untouched test set.")
    if metric_info is None and not verdicts:
        still_improving = train_tail_slope < -0.01 * scale / tail
        if not big_gap and still_improving:
            verdicts.append("still_improving")
            findings.append("Both losses are still falling with a small gap; train longer before changing anything else.")
        elif not big_gap and abs(train_tail_slope) < 1e-3 * scale and tl[-1] > 0.5 * tl[0]:
            verdicts.append("underfitting")
            findings.append("Train loss has plateaued high with a small gap; likely capacity-limited. Check for bugs and data defects first.")

    if not verdicts:
        verdicts.append("healthy")
        findings.append("No red flags in the curves. Compare against the target and the baseline.")

    # Instability or a flat run points to a bug or bad optimization setting; fix that before reasoning about capacity.
    if "unstable" in verdicts or "not_learning" in verdicts:
        verdicts = [v for v in verdicts if v != "underfitting"]
        findings = [f for f in findings if not f.startswith("Training metric itself misses the target")]

    out = {"verdict": sorted(set(verdicts)), "findings": findings, "metrics": info}
    if metric_info:
        out["target_check"] = metric_info
    out["note"] = "Heuristic first pass; confirm with worst-error review and the debugging checklist."
    return out


def plot_curves(rows, path, target=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return False
    x = [r["epoch"] for r in rows]
    has_metric = "val_metric" in rows[0] and "train_metric" in rows[0]
    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(11 if has_metric else 6, 4), squeeze=False)
    ax = axes[0][0]
    ax.plot(x, curve_col(rows, "train_loss"), label="train")
    ax.plot(x, curve_col(rows, "val_loss"), label="validation")
    ax.set_xlabel("epoch / iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    if has_metric:
        ax2 = axes[0][1]
        ax2.plot(x, curve_col(rows, "train_metric"), label="train")
        ax2.plot(x, curve_col(rows, "val_metric"), label="validation")
        if target is not None:
            ax2.axhline(target, linestyle="--", color="gray", label="target")
        ax2.set_xlabel("epoch / iteration")
        ax2.set_ylabel("metric")
        ax2.set_title("Metric")
        ax2.legend()
    # keep tick labels readable for many points
    for a in axes[0]:
        if len(x) > 12:
            a.set_xticks(a.get_xticks()[:: max(1, len(a.get_xticks()) // 8)])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True

# ====================================================================================================
# diagnose.py
# ====================================================================================================
"""Sanity checks (Deep Learning book Ch. 11 debugging), qualitative review, and next-step advice for DETR."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn



def _to_dev(images, mask, targets, device):
    return images.to(device), mask.to(device), [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]


def check_initial_loss(model, criterion, batch, device, num_classes, tol):
    """Untrained heads: every query is nearly uniform, so the class loss is close to ln(K + 1)."""
    m = copy.deepcopy(model).to(device).eval()
    images, mask, targets = _to_dev(*batch, device)
    with torch.no_grad():
        ls = criterion(m(images, mask), targets)
    expected, got = math.log(num_classes + 1), float(ls["loss_ce"])
    return {"initial_loss_ce": got, "expected_ln_k": expected, "initial_total": float(ls["total"]),
            "passed": bool(abs(got - expected) <= tol * expected)}


def overfit_tiny_batch(model, criterion, batch, device, steps=80, ratio=0.85, lr=3e-4, grad_clip=0.1):
    """The network must be able to reduce its loss on a handful of images (dropout off); failure points to a bug."""
    m = copy.deepcopy(model).to(device).train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, nn.MultiheadAttention):
            mod.dropout = 0.0
    images, mask, targets = _to_dev(*batch, device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss = criterion(m(images, mask), targets)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        hist.append(float(loss.detach()))
    final = float(np.mean(hist[-5:]))
    return {"overfit_initial_loss": hist[0], "overfit_final_loss": final, "overfit_ratio": final / max(hist[0], 1e-9),
            "passed": bool(final < ratio * hist[0])}


def run_sanity_checks(cfg, model, criterion, batch, device):
    s = cfg["sanity"]
    n = s["overfit_images"]
    small = (batch[0][:n], batch[1][:n], batch[2][:n])
    init = check_initial_loss(model, criterion, small, device, cfg["data"]["num_foreground"], s["initial_loss_tol"])
    over = overfit_tiny_batch(model, criterion, small, device, s["overfit_steps"], s["overfit_ratio"], s["overfit_lr"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"}, "initial_loss_ok": init["passed"],
            "overfit_ok": over["passed"], "passed": bool(init["passed"] and over["passed"])}


@torch.inference_mode()
def qualitative_grid(model, dataset, classes, device, out_path, thr=0.5, n=8):
    """Predicted (red) vs ground-truth (green) boxes on the first n images of a dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes
    model.eval()
    cols = 4
    n = min(n, len(dataset))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(n), strict=False):
        x, tgt, meta = dataset[i]
        h, w = x.shape[1:]
        out = model(x[None].to(device), torch.zeros(1, h, w, dtype=torch.bool, device=device))
        det = postprocess(out, torch.tensor([[w, h]]))[0]
        keep = det["scores"] >= thr
        u8 = (denormalize(x) * 255).round().byte()
        if len(tgt["labels"]):
            gt = box_cxcywh_to_xyxy(tgt["boxes"]) * torch.tensor([w, h, w, h])
            u8 = draw_bounding_boxes(u8, gt, [classes[int(c)] for c in tgt["labels"]], colors="lime", width=2)
        if keep.any():
            u8 = draw_bounding_boxes(u8, det["boxes"][keep].cpu(),
                                     [f"{classes[int(c) - 1]} {s:.2f}" for c, s in zip(det["labels"][keep].cpu(), det["scores"][keep].cpu(), strict=True)],
                                     colors="red", width=2)
        ax.imshow(u8.permute(1, 2, 0).numpy())
    fig.suptitle(f"green = ground truth, red = predictions (score >= {thr})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


LEVERS = {
    "overfitting": [
        "Keep early stopping on: the saved model is the best-validation epoch, not the last one",
        "More data is the real fix for a transformer: add data.voc_train_sets=[\"2012_trainval\",\"2007_trainval\"] or train on COCO (stage coco)",
        "Stronger regularisation: model.dropout=0.2, schedule.weight_decay=0.0002, fewer queries model.num_queries=50, smaller model.enc_layers/dec_layers=4",
        "Enable weight averaging: schedule.ema.enabled=true",
        "Lower the backbone LR further: schedule.lr_backbone=0.000005"],
    "underfitting": [
        "DETR converges slowly: train longer (schedule.phases epochs up, schedule.scale=1.5); the paper used 300-500 epochs",
        "Check the learning rates: schedule.phases[0].lr=0.0002 for batch sizes above 32; keep schedule.grad_clip=0.1",
        "More capacity: model.backbone=resnet101 or model.dilation=true (DC5, 2x compute)",
        "Remove regularisation first: model.dropout=0, aug.crop_prob=0",
        "Check labels and boxes in qualitative_val.png"],
    "unstable": ["Lower the first-phase LR (x0.5), keep schedule.grad_clip=0.1, check for degenerate boxes"],
    "not_learning": ["DETR often shows near-zero mAP for many epochs on small data when trained from an ImageNet backbone: see "
                     "sanity.json, qualitative_val.png, and consider more data or a longer schedule"],
    "still_improving": ["Train longer: raise the phase lengths in schedule.phases; the validation metric had not plateaued"],
    "healthy": ["No red flags in the curves. Compare with the target, then evaluate once on the test set"],
}


def next_steps(diagnosis: dict) -> dict:
    verdicts = diagnosis.get("verdict", [])
    return {"verdict": verdicts, "levers": {v: LEVERS[v] for v in verdicts if v in LEVERS},
            "note": "A diagnosis is a hypothesis. Change ONE lever at a time and compare runs in MLflow."}

# ====================================================================================================
# train.py
# ====================================================================================================
"""DETR training (paper section 4 / A.4): AdamW, backbone LR 1e-5, transformer LR 1e-4, weight decay 1e-4, gradient clipping 0.1,
one LR step-down, plus validation-based early stopping, EMA option and MLflow logging."""
import contextlib
import csv
import json
import math
import shutil
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import mlflow
import numpy as np
import torch


CSV_FIELDS = ["epoch", "iteration", "phase", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "train_ce",
              "train_bbox", "train_giou", "val_ce", "val_bbox", "val_giou", "val_map50", "val_map50_voc07", "grad_norm",
              "imgs_per_s", "elapsed_s"]


def git_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        dev = torch.device(pref)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but PyTorch sees no GPU (CPU-only runtime?). On Databricks pick "
                               "the 'Machine Learning' GPU runtime.")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_mlflow(cfg: dict, paths: dict) -> str:
    """Tracking store: Databricks workspace (artifacts forced into <dest>) or SQLite/file under <dest>."""
    m = cfg["mlflow"]
    backend = m.get("backend", "auto")
    if backend == "auto":
        backend = "databricks" if on_databricks() else "sqlite"
    art = str(paths["mlflow_artifacts"])
    if backend == "databricks":
        uri = "databricks"
        art_loc = f"dbfs:{art}" if art.startswith("/Volumes/") else art
    elif backend == "sqlite":
        uri, art_loc = f"sqlite:///{paths['mlflow'] / 'mlflow.db'}", art
    elif backend == "file":
        uri, art_loc = f"file:{paths['mlflow'] / 'mlruns'}", art
    else:
        uri, art_loc = backend, art
    mlflow.set_tracking_uri(uri)
    name = m["experiment_name"]
    if backend == "databricks" and not name.startswith("/"):
        name = "/Shared/" + name
    if mlflow.get_experiment_by_name(name) is None:
        try:
            mlflow.create_experiment(name, artifact_location=art_loc)
        except Exception as e:  # e.g. the experiment path does not exist in the workspace
            raise RuntimeError(f"could not create MLflow experiment {name!r} at {uri}: {e}") from e
    mlflow.set_experiment(name)
    return uri


def amp_context(device, mode):
    if device.type != "cuda" or mode == "none":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)




def targets_to(targets, device):
    return [{"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)} for t in targets]


def make_criterion(cfg: dict, num_classes: int) -> SetCriterion:
    lc = cfg["loss"]
    w = {"loss_ce": lc["w_ce"], "loss_bbox": lc["w_bbox"], "loss_giou": lc["w_giou"]}
    return SetCriterion(num_classes, HungarianMatcher(lc["cost_class"], lc["cost_bbox"], lc["cost_giou"]), w, lc["eos_coef"])


def evaluate_model(model, loader, device, criterion, cfg, classes, fast=True, keep=False):
    """One pass: loss components + detections -> VOC-protocol metrics (IoU 0.5 only when fast)."""
    model.eval()
    ev = DetectionEvaluator(classes, [0.5] if fast else None)
    sums, n = defaultdict(float), 0
    preds, gts = [], []
    with torch.inference_mode():
        for images, mask, targets, metas in loader:
            images, mask = images.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with amp_context(device, cfg["schedule"]["amp"]):
                out = model(images, mask)
            ls = criterion(out, targets_to(targets, device))
            for k in ("total", "loss_ce", "loss_bbox", "loss_giou"):
                sums[k] += float(ls[k]) * len(metas)
            n += len(metas)
            dets = postprocess(out, torch.tensor([m["size"] for m in metas]))
            for d, m in zip(dets, metas, strict=True):
                p = {"boxes": d["boxes"].cpu().numpy(), "scores": d["scores"].cpu().numpy(), "labels": d["labels"].cpu().numpy()}
                g = {"boxes": m["boxes"], "labels": m["labels"], "difficult": m["difficult"]}
                ev.update([p], [g])
                if keep:
                    preds.append(p)
                    gts.append(g)
    res = ev.compute()
    res.update({f"loss_{k}" if not k.startswith("loss") else k: v / max(n, 1) for k, v in sums.items()})
    return (res, preds, gts) if keep else res


def save_checkpoint(path, model_state, cfg, classes, epoch, val_metric, ema_used):
    torch.save({"state_dict": model_state, "cfg": cfg, "classes": classes, "num_foreground": len(classes), "epoch": epoch,
                "val_metric": val_metric, "metric_name": cfg["metric"]["name"], "ema": ema_used, "stage": cfg["stage"]}, path)


def fit(cfg, data, loaders, model, criterion, device, out_dir: Path):
    s, es_cfg = cfg["schedule"], cfg["schedule"]["early_stopping"]
    classes, metric_name = data["classes"], cfg["metric"]["name"]
    phases = [(p["lr"], p["epochs"]) for p in s["phases"]]
    optimizer = torch.optim.AdamW(model.param_groups(phases[0][0], s["lr_backbone"], s["weight_decay"]))
    sch = PhaseSchedule(optimizer, phases, s["scale"], 0)
    has_val = loaders["val"] is not None
    es = PhaseEarlyStopper(es_cfg["patience"], es_cfg["min_delta"], bool(es_cfg["enabled"] and has_val))
    ema = ModelEMA(model, s["ema"]["decay"], s["ema"]["tau"]) if s["ema"]["enabled"] else None
    eval_model = ema.module if ema is not None else model
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and s["amp"] == "fp16") else None
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    best_path, last_path, csv_path = ck / "best.pt", ck / "last.pt", out_dir / "metrics.csv"
    epoch = iteration = 0
    if cfg.get("resume_from"):
        st = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["sched"])
        es.load_state_dict(st["es"])
        if ema is not None and st.get("ema"):
            ema.load_state_dict(st["ema"])
        epoch, iteration = st["epoch"], st["iteration"]
        prev = Path(cfg["resume_from"]).parent.parent   # carry the best checkpoint and the history into this run
        for name, dst in (("checkpoints/best.pt", best_path), ("metrics.csv", csv_path)):
            if (prev / name).exists() and not dst.exists():
                shutil.copy2(prev / name, dst)
    best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
    params = [p for p in model.parameters() if p.requires_grad]
    win, n_win, n_imgs, gn_sum = defaultdict(float), 0, 0, 0.0
    t_win = t_start = time.time()
    stop_reason, last_row = "schedule_complete", {}
    resumed = csv_path.exists()
    f = open(csv_path, "a" if resumed else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not resumed:
        writer.writeheader()
    try:
        while True:
            lr = sch.step()                       # LR for this epoch (the phase LR; backbone gets its own multiple)
            model.train()
            for images, mask, targets in loaders["train"]:
                images, mask, targets = images.to(device, non_blocking=True), mask.to(device, non_blocking=True), targets_to(targets, device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, s["amp"]):
                    out = model(images, mask)
                ls = criterion(out, targets)
                loss = ls["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at iteration {iteration}: lower schedule.phases[0].lr")
                clip = s["grad_clip"] or float("inf")
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(params, clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(params, clip)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
                iteration += 1
                win["total"] += float(loss.detach())
                for k in ("loss_ce", "loss_bbox", "loss_giou"):
                    win[k] += float(ls[k])
                gn_sum += float(gn)
                n_win += 1
                n_imgs += len(images)
                if iteration % s["log_every"] == 0:
                    mlflow.log_metrics({"train_loss_iter": win["total"] / n_win, "grad_norm": gn_sum / n_win}, step=iteration)
            epoch += 1
            mlflow.log_metrics({"lr": lr, "train_loss_epoch": win["total"] / max(n_win, 1)}, step=epoch)
            if not has_val:
                if sch.phase_done():
                    if sch.is_last_phase:
                        break
                    sch.advance()
                continue
            if epoch % s["eval_every"] != 0 and not sch.phase_done():
                continue
            dt = max(time.time() - t_win, 1e-9)
            row = {"epoch": epoch, "iteration": iteration, "phase": sch.phase, "lr": lr, "train_loss": win["total"] / n_win,
                   "train_ce": win["loss_ce"] / n_win, "train_bbox": win["loss_bbox"] / n_win, "train_giou": win["loss_giou"] / n_win,
                   "grad_norm": gn_sum / n_win, "imgs_per_s": n_imgs / dt, "elapsed_s": time.time() - t_start}
            v = evaluate_model(eval_model, loaders["val"], device, criterion, cfg, classes)
            value = v[metric_name]
            row.update({"val_loss": v["loss_total"], "val_ce": v["loss_ce"], "val_bbox": v["loss_bbox"], "val_giou": v["loss_giou"],
                        "val_metric": value, "val_map50": v["map50"], "val_map50_voc07": v["map50_voc07"]})
            if loaders.get("train_eval") is not None:
                row["train_metric"] = evaluate_model(eval_model, loaders["train_eval"], device, criterion, cfg, classes)[metric_name]
            writer.writerow(row)
            f.flush()
            mlflow.log_metrics({k: x for k, x in row.items() if k not in ("epoch", "iteration") and x is not None
                                and not (isinstance(x, float) and math.isnan(x))}, step=epoch)
            last_row = row
            gap = "" if "train_metric" not in row else f" | gap {row['train_metric'] - value:+.3f}"
            print(f"epoch {epoch:4d} ph {sch.phase} | loss {row['train_loss']:.3f} | val loss {row['val_loss']:.3f} | val {metric_name} "
                  f"{value:.4f}{gap} | lr {lr:.6f} | {row['imgs_per_s']:.0f} img/s", flush=True)
            improved = es.update(value, epoch)
            if improved:
                best_state = {k: x.detach().cpu().clone() for k, x in eval_model.state_dict().items()}
                save_checkpoint(best_path, best_state, cfg, classes, epoch, value, ema is not None)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sched": sch.state_dict(),
                        "es": es.state_dict(), "ema": ema.state_dict() if ema is not None else None, "epoch": epoch,
                        "iteration": iteration}, last_path)
            win.clear()
            n_win = n_imgs = 0
            gn_sum = 0.0
            t_win = time.time()
            if es.exhausted or sch.phase_done():
                plateau = es.exhausted
                if sch.is_last_phase:
                    stop_reason = "early_stop_plateau" if plateau else "schedule_complete"
                    break
                if plateau and not es_cfg["advance_phase_on_plateau"] and not sch.phase_done():
                    continue
                sch.advance()
                es.reset_phase()
                mlflow.log_metric("phase_change_epoch", epoch, step=epoch)
                if es_cfg["restore_best_on_decay"]:
                    model.load_state_dict(best_state)
                    if ema is not None:
                        ema.module.load_state_dict(best_state)
                    optimizer.state.clear()
                print(f"  -> LR phase {sch.phase} ({'plateau' if plateau else 'phase complete'}), lr {sch.phases[sch.phase][0]}", flush=True)
    finally:
        f.close()
    if not has_val:
        save_checkpoint(best_path, {k: x.detach().cpu() for k, x in eval_model.state_dict().items()}, cfg, classes, epoch, None, ema is not None)
    return {"best_path": best_path, "csv_path": csv_path, "epochs": epoch, "iterations": iteration, "stop_reason": stop_reason,
            "best_val_metric": es.best if has_val else None, "best_epoch": es.best_it, "last_row": last_row}


def run_training(cfg: dict, paths: dict) -> dict:
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    model = build_model(cfg).to(device)
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    uri = setup_mlflow(cfg, paths)
    with mlflow.start_run(run_name=cfg.get("run_label") or cfg["stage"]) as run:
        run_id = run.info.run_id
        out_dir = Path(paths["runs"]) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "running"}))
        mlflow.log_params({k: str(v)[:490] for k, v in flatten({kk: vv for kk, vv in cfg.items() if kk != "dest"}).items()})
        mlflow.log_params({"num_parameters": count_params(model), "num_trainable": count_params(model, True)})
        mlflow.set_tags({"stage": cfg["stage"], "task": "object-detection", "model": "DETR", "git_commit": git_commit(),
                         "data_version": data["data_version"], "metric": cfg["metric"]["name"],
                         "metric_target": str(cfg["metric"]["target_value"]), "test_evaluated": "false", "device": str(device),
                         "tracking_uri": uri, "dest": str(paths["dest"]), "pipeline_step": str(cfg.get("run_label", ""))})
        print(f"run {run_id} | stage {cfg['stage']} | device {device} | params {count_params(model):,} "
              f"(trainable {count_params(model, True):,}) | data {data['data_version']}")
        if cfg["sanity"]["enabled"]:
            rep = run_sanity_checks(cfg, model, criterion, next(iter(loaders["train"])), device)
            (out_dir / "sanity.json").write_text(json.dumps(rep, indent=2))
            mlflow.log_dict(rep, "diagnostics/sanity.json")
            mlflow.set_tag("sanity_passed", str(rep["passed"]).lower())
            print("sanity:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
            if cfg["sanity"]["fail_hard"] and not rep["passed"]:
                raise RuntimeError(f"sanity checks failed: {json.dumps(rep)}")
        set_seed(cfg["seed"])
        res = fit(cfg, data, loaders, model, criterion, device, out_dir)
        mlflow.set_tags({"stop_reason": res["stop_reason"], "best_epoch": str(res["best_epoch"])})
        mlflow.log_metrics({"total_epochs": res["epochs"], "total_iterations": res["iterations"]})
        summary = {"run_id": run_id, "stage": cfg["stage"], "epochs": res["epochs"], "iterations": res["iterations"],
                   "stop_reason": res["stop_reason"], "best_epoch": res["best_epoch"], "best_val_metric": res["best_val_metric"],
                   "out_dir": str(out_dir)}
        state = torch.load(res["best_path"], map_location="cpu", weights_only=False)
        best = build_model(cfg, pretrained=False)
        best.load_state_dict(state["state_dict"])
        best.to(device).eval()
        if loaders["val"] is not None:
            v, preds, gts = evaluate_model(best, loaders["val"], device, criterion, cfg, data["classes"], fast=False, keep=True)
            mlflow.log_metrics({f"final_val_{k}": v[k] for k in ("map50", "map50_voc07", "map75", "map", "mar100")})
            mlflow.log_metric("best_val_metric", res["best_val_metric"])
            (out_dir / "val_per_class_ap50.json").write_text(json.dumps(v["ap50_per_class"], indent=2))
            errs = error_breakdown(preds, gts, 0.5)
            (out_dir / "val_error_breakdown.json").write_text(json.dumps(errs, indent=2))
            mlflow.log_dict(errs, "diagnostics/val_error_breakdown.json")
            if loaders.get("train_eval") is not None:
                tm = evaluate_model(best, loaders["train_eval"], device, criterion, cfg, data["classes"])[cfg["metric"]["name"]]
                gap = tm - v[cfg["metric"]["name"]]
                flag = bool(gap > cfg["schedule"]["overfit_gap_warn"])
                mlflow.log_metrics({"final_train_metric": tm, "generalisation_gap": gap})
                mlflow.set_tag("overfit_warning", str(flag).lower())
                summary.update({"train_metric_at_best": tm, "generalisation_gap": gap, "overfit_warning": flag})
                if flag:
                    print(f"WARNING: train - val {cfg['metric']['name']} gap {gap:.3f} exceeds {cfg['schedule']['overfit_gap_warn']}: "
                          "see next_steps.json (overfitting levers)")
            try:
                qualitative_grid(best, data["val"], data["classes"], device, out_dir / "qualitative_val.png")
            except Exception as e:  # plotting must never fail a finished training run
                print("qualitative grid skipped:", e)
            rows = curve_read_rows(res["csv_path"])
            diag = diagnose_curves(rows, target=cfg["metric"]["target_value"])
            (out_dir / "diagnosis.json").write_text(json.dumps(diag, indent=2))
            (out_dir / "next_steps.json").write_text(json.dumps(next_steps(diag), indent=2))
            try:
                plot_curves(rows, str(out_dir / "curves.png"), cfg["metric"]["target_value"])
            except Exception as e:
                print("curve plot skipped:", e)
            summary["diagnosis"] = diag["verdict"]
            print("diagnosis:", diag["verdict"])
        (out_dir / "class_names.json").write_text(json.dumps(data["classes"]))
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "complete"}))
        stable = Path(paths["checkpoints"]) / f"{cfg['stage']}_best.pt"
        shutil.copy2(res["best_path"], stable)
        for p in out_dir.glob("*"):
            if p.is_file():
                mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(res["best_path"]), "checkpoints")
        mlflow.log_dict({p: metadata.version(p) for p in ("torch", "torchvision")}, "config/versions.json")
        summary["stable_checkpoint"] = str(stable)
        summary["target_met"] = bool(res["best_val_metric"] is not None and res["best_val_metric"] >= cfg["metric"]["target_value"])
        print(json.dumps(summary, indent=2))
        return summary


def benchmark(cfg: dict, paths: dict, steps: int = 6, loader_batches: int = 12) -> dict:
    """Model step time and augmentation throughput on THIS machine (random images, no downloads)."""
    device = select_device(cfg["device"])
    set_seed(cfg["seed"])
    model = build_model(cfg, pretrained=False).to(device).train()
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    opt = torch.optim.AdamW(model.param_groups(1e-4, 1e-5, 1e-4))
    bs, size, mx = cfg["data"]["batch_size"], cfg["aug"]["test_size"], cfg["aug"]["max_size"]
    h, w = size, min(int(size * 4 / 3), mx)
    x, mask = torch.randn(bs, 3, h, w, device=device), torch.zeros(bs, h, w, dtype=torch.bool, device=device)
    tg = [{"boxes": torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.3, 0.2]], device=device), "labels": torch.tensor([0, 1], device=device)}
          for _ in range(bs)]

    def step():
        opt.zero_grad()
        with amp_context(device, cfg["schedule"]["amp"]):
            out = model(x, mask)
        criterion(out, tg)["total"].backward()
        opt.step()
    for _ in range(2):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - t0) / steps
    ds = DetrDataset(SyntheticSource(max(bs * 6, 48), 20, 500, cfg["seed"]), cfg["aug"], True)
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=int(cfg["data"]["num_workers"]), collate_fn=collate_train)
    t1, n = None, 0
    for i, _ in enumerate(ld):
        if i == 2:
            t1 = time.time()
        if i >= 2:
            n += bs
        if i >= loader_batches:
            break
    loader_ips = n / max(time.time() - (t1 or time.time()), 1e-9) if t1 else float("nan")
    ips_model = bs / sec
    n_train = {"voc2012": 10386, "coco2017": 118287}.get(cfg["data"]["dataset"], cfg["data"]["synthetic"]["size"])
    epoch_s = n_train / min(ips_model, loader_ips if loader_ips == loader_ips else ips_model)
    sch = cfg["schedule"]
    epochs = sum(round(p["epochs"] * sch["scale"]) for p in sch["phases"])
    out = {"device": str(device), "batch_size": bs, "image_hw": [h, w], "model_step_s": round(sec, 3), "model_img_per_s": round(ips_model, 1),
           "augmentation_img_per_s": round(loader_ips, 1), "bottleneck": "data loading" if loader_ips < ips_model else "model",
           "est_minutes_per_epoch": round(epoch_s / 60, 2), "schedule_epochs_max": epochs,
           "est_hours_if_run_to_the_end": round(epoch_s * epochs / 3600, 2),
           "note": "estimate; early stopping usually ends a run sooner; evaluation time is extra"}
    if device.type == "cuda":
        out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out

# ====================================================================================================
# evaluate.py
# ====================================================================================================
"""One-time evaluation of the best checkpoint on the untouched test set (logged into the training run)."""
import json
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient



def run_test_evaluation(cfg: dict, paths: dict, run_id: str, force: bool = False) -> dict:
    setup_mlflow(cfg, paths)
    client = MlflowClient()
    count = int(client.get_run(run_id).data.tags.get("test_evaluation_count", "0"))
    if count >= 1 and not force:
        raise RuntimeError(f"run {run_id} was already evaluated on the test set {count} time(s). Tuning on the test set "
                           "invalidates it; pass force=True only to re-verify (the count is recorded).")
    if not cfg["data"].get("test_set") and cfg["data"]["dataset"] != "synthetic":
        raise ValueError("this stage has no test set (COCO stage: use the validation metrics)")
    device = select_device(cfg["device"])
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]))))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    res, preds, gts = evaluate_model(model, loaders["test"], device, criterion, cfg, data["classes"], fast=False, keep=True)
    errs = error_breakdown(preds, gts, 0.5)
    out_dir = Path(paths["runs"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_per_class_ap50.json").write_text(json.dumps(res["ap50_per_class"], indent=2))
    metric, target = cfg["metric"]["name"], cfg["metric"]["target_value"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": res[k] for k in ("map50", "map50_voc07", "map75", "map", "mar100", "loss_total")})
        mlflow.set_tags({"test_evaluated": "true", "test_evaluation_count": str(count + 1),
                         "test_target_met": str(res[metric] >= target).lower()})
        mlflow.log_artifact(str(out_dir / "test_per_class_ap50.json"), "test")
        mlflow.log_dict(errs, "test/error_breakdown.json")
    worst = sorted(((k, v) for k, v in res["ap50_per_class"].items() if v is not None), key=lambda kv: kv[1])[:5]
    summary = {"run_id": run_id, "test_set": cfg["data"].get("test_set") or "synthetic", f"test_{metric}": res[metric],
               "test_map50_voc07": res["map50_voc07"], "test_map": res["map"], "test_map75": res["map75"], "test_mar100": res["mar100"],
               "target": target, "target_met": res[metric] >= target, "evaluation_count": count + 1,
               "worst_classes_ap50": worst, "error_breakdown": errs}
    print(json.dumps(summary, indent=2))
    return summary

# ====================================================================================================
# serve.py
# ====================================================================================================
"""Packaging: a model folder (checkpoint + pyfunc), an MLflow pyfunc logged in the run, and a fresh-process reload check."""
import base64
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema
from PIL import Image


# MLflow "model from code": loading it needs this project's code, which MLflow copies next to the model (code_paths).
PYFUNC_TEMPLATE = '''
import base64
import io
import json

import mlflow
import pandas as pd
import torch
from PIL import Image

from {module} import Predictor


class DetrDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = Predictor(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        params = params or {{}}
        thr, k = float(params.get("score_threshold", 0.5)), int(params.get("max_detections", 100))
        rows = []
        for b64 in model_input["image_b64"].tolist():
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            rows.append(json.dumps(self.predictor.predict([img], thr, k)[0]))
        return pd.DataFrame({{"detections": rows}})


mlflow.models.set_model(DetrDetector())
'''

VERIFY_SNIPPET = '''
import json, sys
import mlflow, numpy as np, pandas as pd
uri, tracking, example_csv, expected_json, atol = sys.argv[1:6]
mlflow.set_tracking_uri(tracking)
model = mlflow.pyfunc.load_model(uri)
got = json.loads(model.predict(pd.read_csv(example_csv), params={"score_threshold": 0.0})["detections"].iloc[0])["detections"]
want = json.load(open(expected_json))["detections"]
if len(got) != len(want):
    print("FAIL: %d detections vs %d expected" % (len(got), len(want))); sys.exit(1)
if not got:
    print("OK (no detections on either side)"); sys.exit(0)
err = max(float(np.abs(np.array([d["box"] for d in got]) - np.array([d["box"] for d in want])).max()),
          float(np.abs(np.array([d["score"] for d in got]) - np.array([d["score"] for d in want])).max()))
print("max abs diff %.2e over %d detections" % (err, len(got)))
sys.exit(0 if err <= float(atol) else 1)
'''


def code_reference() -> tuple[str, str]:
    """(path MLflow must ship, module name that provides Predictor): the package folder, or the single-file script."""
    here = Path(__file__).resolve()
    if here.name == "serve.py" and (here.parent / "infer.py").exists():
        return str(here.parent), "detr_voc.infer"
    return str(here), here.stem


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def package_model(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, out_dir) -> dict:
    """Save checkpoint + pyfunc source into out_dir, log the pyfunc model to the run, verify it in a FRESH process."""
    uri = setup_mlflow(cfg, paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]) / f"pkg_{run_id}")))
    shutil.copy2(ckpt, out / "checkpoint.pt")
    code_path, module = code_reference()
    (out / "detr_pyfunc.py").write_text(PYFUNC_TEMPLATE.format(module=module))
    b64 = _png_b64(example_image)
    pd.DataFrame({"image_b64": [b64]}).to_csv(out / "example.csv", index=False)
    (out / "expected.json").write_text(json.dumps(Predictor(out / "checkpoint.pt").predict([example_image], score_threshold=0.0)[0]))
    signature = ModelSignature(inputs=Schema([ColSpec("string", "image_b64")]), outputs=Schema([ColSpec("string", "detections")]),
                               params=ParamSchema([ParamSpec("score_threshold", "float", 0.5), ParamSpec("max_detections", "integer", 100)]))
    import torchvision
    reqs = [f"torch=={torch.__version__.split('+')[0]}",
            f"torchvision=={torchvision.__version__.split('+')[0]}", f"mlflow=={mlflow.__version__}", "numpy", "pandas", "pillow"]
    code_dir = str(Path(code_path).parent)              # MLflow imports the model file while logging: make the module importable
    added = code_dir not in sys.path
    if added:
        sys.path.insert(0, code_dir)
    try:
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(name="model", python_model=str(out / "detr_pyfunc.py"),
                                           artifacts={"checkpoint": str(out / "checkpoint.pt")}, code_paths=[code_path],
                                           signature=signature, input_example=pd.DataFrame({"image_b64": [b64]}), pip_requirements=reqs)
    finally:
        if added:
            sys.path.remove(code_dir)
    proc = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET, info.model_uri, uri, str(out / "example.csv"),
                           str(out / "expected.json"), "0.05"], cwd=str(out), capture_output=True, text=True)
    ok = proc.returncode == 0
    print(proc.stdout.strip(), proc.stderr.strip()[-400:] if not ok else "")
    (out / "README.txt").write_text(
        "DETR model folder\n  checkpoint.pt  full checkpoint (config + classes): Predictor('checkpoint.pt').predict([image])\n"
        f"  MLflow pyfunc  logged in run {run_id} as 'model' (input column image_b64, output column detections)\n")
    return {"model_dir": str(out), "model_uri": info.model_uri, "load_verified": ok, "tracking_uri": uri}


def register_candidate(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, model_name: str | None = None) -> dict:
    """package_model + registry: set the alias `candidate` if the fresh-process check passed."""
    setup_mlflow(cfg, paths)
    if on_databricks() and cfg["mlflow"].get("backend", "auto") in ("auto", "databricks"):
        mlflow.set_registry_uri("databricks-uc")
    name = model_name or registered_name(cfg)
    if on_databricks() and name.count(".") != 2:
        raise ValueError(f"Unity Catalog needs a three-level model name catalog.schema.model, got {name!r}")
    pkg = package_model(cfg, paths, run_id, example_image, Path(paths["tmp"]) / f"register_{run_id}")
    mv = mlflow.register_model(pkg["model_uri"], name)
    client = MlflowClient()
    client.set_model_version_tag(name, mv.version, "load_verified", str(pkg["load_verified"]).lower())
    client.set_model_version_tag(name, mv.version, "source_run_id", run_id)
    if pkg["load_verified"]:
        client.set_registered_model_alias(name, "candidate", mv.version)
    return {"name": name, "version": mv.version, "model_uri": pkg["model_uri"], "load_verified": pkg["load_verified"]}

# ====================================================================================================
# pipeline.py
# ====================================================================================================
"""The whole job in order: load datasets -> DETR training (single phase, early stopping) -> test once -> save the model.
Inputs: a path and a device (cpu | gpu). An interrupted run resumes where it stopped."""
import json
import os
import sys
import time
from pathlib import Path

import torch


DEVICE_MODES = ("cpu", "gpu")
DEVICE_HELP = {"cpu": "everything on the CPU (DETR is very slow on a CPU: use it for small checks)",
               "gpu": "the model trains on the GPU; CPU workers load and augment images and the CPU solves the Hungarian matching"}
STEPS = [{"key": "detr", "title": "DETR training (single phase) -> test -> save", "test": True, "save": "detr"}]


def choose_device_mode(value=None) -> str:
    """Validate the answer, or ask for it when running in a terminal."""
    names = {"1": "cpu", "2": "gpu"}
    if value:
        v = names.get(str(value).strip(), str(value).strip().lower())
        if v not in DEVICE_MODES:
            raise ValueError(f"device must be one of {DEVICE_MODES}, got {value!r}")
        return v
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("Choose where to train: pass cpu or gpu as the second argument.")
    print("Where should training run?")
    for i, m in enumerate(DEVICE_MODES, 1):
        print(f"  {i}) {m:5s} {DEVICE_HELP[m]}")
    while True:
        ans = input("Enter 1 or 2: ").strip()
        if ans in names or ans in DEVICE_MODES:
            return names.get(ans, ans)


def device_overrides(mode: str, cpu_count: int | None = None) -> dict:
    n = cpu_count or os.cpu_count() or 4
    if mode == "cpu":
        return {"device": "cpu", "schedule.amp": "none", "data.num_workers": max(2, n // 4)}
    return {"device": "cuda", "schedule.amp": "bf16", "data.num_workers": min(20, max(2, n - 4))}


def check_device(mode: str) -> str:
    if mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("You chose 'gpu' but PyTorch sees no GPU. Choose 'cpu', or use a GPU cluster with the Machine Learning runtime.")
        return torch.cuda.get_device_name(0)
    return "cpu"


class PipelineContext:
    def __init__(self, paths, mode, stage, overrides, state, state_path):
        self.paths, self.mode, self.stage, self.overrides, self.state, self.state_path = paths, mode, stage, overrides, state, state_path

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def cfg(self) -> dict:
        cfg = load_config(self.stage, {**device_overrides(self.mode), **parse_overrides(self.overrides)}, dest=self.paths["dest"])
        cfg["run_label"] = f"1_detr_{cfg['data']['dataset']}"
        return cfg


def start_pipeline(path, device_mode=None, stage="voc", overrides=None) -> PipelineContext:
    mode = choose_device_mode(device_mode)
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
    paths = setup_environment(path)
    gpu_name = check_device(mode)
    if mode == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - device_overrides("cpu")["data.num_workers"]))
    state_path = Path(paths["dest"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    ctx = PipelineContext(paths, mode, stage, overrides, state, state_path)
    state["device_mode"], state["stage"] = mode, stage
    ctx.save()
    ctx.log(f"path: {paths['dest']} | device: {mode} ({DEVICE_HELP[mode]}) | {gpu_name} | stage: {stage}")
    if state["stages"].get("detr", {}).get("train_done"):
        ctx.log("found earlier progress, finished steps will be skipped")
    return ctx


def step_load_datasets(ctx: PipelineContext) -> dict:
    ctx.log("step 1: loading datasets")
    data = build_data(ctx.cfg(), ctx.paths)
    info = {"data_version": data["data_version"], "train": len(data["train"]), "val": len(data["val"] or []), "test": len(data["test"] or [])}
    ctx.state["datasets"] = info
    ctx.save()
    ctx.log(f"datasets ready: {info}")
    return info


def step_estimate_time(ctx: PipelineContext) -> dict:
    cfg = ctx.cfg()
    b = benchmark(cfg, ctx.paths, steps=4, loader_batches=6)
    ctx.log(f"time estimate (upper bound; early stopping usually ends sooner, evaluation is extra): {b}")
    ctx.state["estimate"] = b
    ctx.save()
    return b


def find_resumable(paths: dict, stage: str) -> Path | None:
    best, best_t = None, -1.0
    for f in Path(paths["runs"]).glob("*/stage.json"):
        try:
            meta = json.loads(f.read_text())
        except ValueError:
            continue
        last = f.parent / "checkpoints" / "last.pt"
        if meta.get("stage") == stage and meta.get("status") == "running" and last.exists() and last.stat().st_mtime > best_t:
            best, best_t = last, last.stat().st_mtime
    return best


def _supersede_unfinished(paths: dict, stage: str, keep_run: str) -> None:
    for f in Path(paths["runs"]).glob("*/stage.json"):
        meta = json.loads(f.read_text())
        if meta.get("stage") == stage and meta.get("status") == "running" and f.parent.name != keep_run:
            f.write_text(json.dumps({**meta, "status": "superseded"}))


def step_train(ctx: PipelineContext) -> dict:
    """Train (resuming an interrupted run), test once on the held-out test set, save the model folder."""
    spec, st = STEPS[0], ctx.state["stages"].setdefault("detr", {})
    cfg = ctx.cfg()
    ctx.log("step 2: " + spec["title"])
    if not st.get("train_done"):
        resume = find_resumable(ctx.paths, cfg["stage"])
        if resume is not None:
            cfg["resume_from"] = str(resume)
            ctx.log(f"resuming the interrupted run from {resume}")
        summary = run_training(cfg, ctx.paths)
        _supersede_unfinished(ctx.paths, cfg["stage"], summary["run_id"])
        st.update({"train_done": True, "run_id": summary["run_id"], "summary": summary})
        ctx.save()
    else:
        ctx.log("training already done, skipping")
    if cfg["data"].get("test_set") or cfg["data"]["dataset"] == "synthetic":
        if "test" not in st:
            st["test"] = run_test_evaluation(cfg, ctx.paths, st["run_id"])
            ctx.save()
    if not st.get("model_dir"):
        data = build_data(cfg, ctx.paths)
        example = (data["val"] or data["test"]).source.get(0)[0]
        pkg = package_model(cfg, ctx.paths, st["run_id"], example, Path(ctx.paths["dest"]) / "models" / f"{spec['save']}_{cfg['data']['dataset']}")
        if not pkg["load_verified"]:
            raise RuntimeError(f"the saved model failed its reload check: {pkg}")
        st["model_dir"] = pkg["model_dir"]
        ctx.save()
        ctx.log(f"saved model: {pkg['model_dir']}")
    return st


def pipeline_summary(ctx: PipelineContext) -> dict:
    st = ctx.state["stages"].get("detr", {})
    sm, t = st.get("summary", {}), st.get("test", {})
    row = {"stage": ctx.stage, "epochs": sm.get("epochs"), "stop_reason": sm.get("stop_reason"), "best_epoch": sm.get("best_epoch"),
           "best_val_map50": sm.get("best_val_metric"), "train_minus_val": sm.get("generalisation_gap"),
           "overfit_warning": sm.get("overfit_warning"), "test_map50": t.get("test_map50"), "saved_model": st.get("model_dir")}
    out = {"device_mode": ctx.mode, "dest": str(ctx.paths["dest"]), "result": row}
    (Path(ctx.paths["dest"]) / "pipeline_summary.json").write_text(json.dumps(out, indent=2, default=str))
    ctx.log("summary:")
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return out


def run_pipeline(path, device_mode=None, stage="voc", overrides=None) -> dict:
    """Everything, in order. `stage` and `overrides` exist for tests and the package CLI; the script takes only path and device."""
    ctx = start_pipeline(path, device_mode, stage, overrides)
    step_load_datasets(ctx)
    step_estimate_time(ctx)
    step_train(ctx)
    return pipeline_summary(ctx)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python {Path(sys.argv[0]).name} PATH [cpu|gpu]\n  PATH: folder for everything (data, caches, runs, models)\n"
              "  device: where to train; asked interactively when omitted")
        return 0
    run_pipeline(argv[0], argv[1] if len(argv) > 1 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
