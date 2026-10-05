#!/usr/bin/env python3
"""SSD (Liu et al., ECCV 2016) with a ResNet-50 base network: the complete pipeline in one file.

  python ssd_pipeline.py PATH [cpu|gpu|combination]

PATH holds everything (data, caches, runs, models, MLflow). The device is asked for when omitted.
Generated from the package by tools/build_notebook.py; do not edit.
"""

# ====================================================================================================
# config.py
# ====================================================================================================
"""Configuration: one default YAML (the paper's recipe adapted to a ResNet), stage overrides, dotted CLI overrides."""
import copy
import os

import yaml

DEFAULT_YAML = """
# SSD300 with a ResNet-50 base network (Liu et al., ECCV 2016, arXiv 1512.02325), single A100 + 24 vCPUs.
# Every output, cache and temporary file lives under `dest` (a parameter, never hard-coded).
seed: 42

data:
  dataset: voc2012                 # voc2012 | coco2017 | synthetic (offline tests only)
  voc_train_sets: ["2012_trainval"]   # add "2007_trainval" for the paper's 07+12 style training data
  test_set: "2007_test"            # untouched until `test`; VOC2012 test labels are not public, VOC2007 test labels are
  val_fraction: 0.10               # held-out part of the training set: early stopping and model selection
  coco_val_images: 2000            # COCO stage: val2017 images used for early stopping
  coco_train_subset: null          # int -> only N COCO train images (debugging)
  subset_train: null               # int -> only N VOC train images (debugging)
  include_difficult_train: true    # the paper's implementation trains on difficult objects; evaluation ignores them
  train_eval_images: 500           # un-augmented train images scored at every evaluation (train/val gap)
  batch_size: 32                   # paper
  num_workers: 20                  # 24 vCPUs: leave a few for the main process
  prefetch_factor: 4
  download: true
  keep_archives: false             # delete downloaded archives after extraction
  synthetic: {size: 64, classes: 3, image_size: 128}

model:
  backbone: resnet50               # resnet18 | resnet34 | resnet50 | resnet101
  pretrained: imagenet             # imagenet (torchvision weights, cached under dest) | none
  input_size: 300                  # 300 -> SSD300 (8732 boxes); 512 -> SSD512 (24564 boxes)
  freeze_up_to: layer1             # stem + layer1 frozen: fewer trainable parameters, less overfitting
  freeze_bn: true                  # frozen BatchNorm statistics
  l2norm_first_map: true           # paper: L2-normalise the first (conv4_3-like) map, learnable scale 20
  l2norm_scale: 20.0
  extras: [[256, 512, 2, 1], [128, 256, 2, 1], [128, 256, 1, 0], [128, 256, 1, 0]]   # paper conv8_2..conv11_2
  boxes_per_loc: [4, 6, 6, 6, 4, 4]
  priors: {first_scale: 0.1, min_scale: 0.2, max_scale: 0.9}
  dropout: 0.0                     # the paper removes all dropout layers
  conf_init_gain: 0.1              # class heads N(0, (gain/sqrt(fan_in))^2); extras and box heads use the paper's Xavier init

loss:
  iou_threshold: 0.5               # paper matching threshold
  neg_pos_ratio: 3                 # paper hard negative mining
  alpha: 1.0                       # paper localisation weight
  variances: [0.1, 0.2]

aug:                               # paper section 2.2 (+ the zoom-out expansion of section 3.6)
  photometric: true
  crop: true                       # random patch with min IoU in {0.1,0.3,0.5,0.7,0.9} | whole image | random patch
  crop_min_scale: 0.3              # torchvision default; the paper text says patches of 0.1 to 1 of the image size
  expansion: false                 # stage "long": place the image on a canvas up to 16x its area (4x per side)
  expansion_max_side: 4.0
  flip: true

postprocess:                       # paper section 3.7
  score_thresh: 0.01
  nms_iou: 0.45
  max_detections: 200
  pre_nms_top_k: 2000

schedule:                          # iterations at batch 32; paper VOC2012: 1e-3 for 60k, then 1e-4 for 20k
  phases:
    - {lr: 0.001, iters: 60000}
    - {lr: 0.0001, iters: 20000}
  scale: 1.0                       # multiplies every phase length (stage "long" uses 2.0, as in the paper)
  warmup_iters: 500                # deviation from the paper: protects the freshly initialised layers
  warmup_factor: 0.1
  momentum: 0.9                    # paper
  weight_decay: 0.0005             # paper (weights only; biases and the L2Norm scale are not decayed)
  nesterov: false
  grad_clip: 10.0
  amp: bf16                        # bf16 | fp16 | none (A100: bf16)
  channels_last: true
  eval_every: 1000                 # iterations between validation passes
  log_every: 50
  early_stopping:
    enabled: true
    patience: 8                    # evaluations without improvement inside the current LR phase
    min_delta: 0.002
    advance_phase_on_plateau: true # plateau -> take the next (lower) LR step early; plateau in the last phase -> stop
    restore_best_on_decay: true    # reload the best weights before each LR drop
  ema:
    enabled: true                  # Polyak averaging; the averaged weights are evaluated and saved
    decay: 0.9995
    tau: 1000
  overfit_gap_warn: 0.20           # warn if train mAP - val mAP exceeds this at the best checkpoint

metric:
  name: map50                      # all-point AP at IoU 0.5 (VOC2012 protocol), mean over classes
  target_value: 0.65               # ASSUMED, not confirmed
  higher_is_better: true

sanity:
  enabled: true
  initial_loss_tol: 0.30           # relative tolerance around (1 + neg_pos_ratio) * ln(num_classes)
  overfit_images: 8
  overfit_steps: 60
  overfit_ratio: 0.70              # tiny-batch loss must fall below this fraction of its start
  fail_hard: true

mlflow:
  backend: auto                    # auto (Databricks workspace on Databricks, else SQLite under dest) | sqlite | file | databricks | <uri>
  experiment_name: ssd-pipeline    # on Databricks a name without a leading / is created under /Shared/
  registered_model_name: null      # null -> ssd-<backbone>-<dataset> (Unity Catalog needs catalog.schema.name)

init_from: null                    # checkpoint to start from (stage voc_from_coco fills this in automatically)
resume_from: null                  # last.pt of an interrupted run
device: auto
"""

COCO_MODEL = {"model.priors": {"first_scale": 0.07, "min_scale": 0.15, "max_scale": 0.9}}  # paper section 3.4
COCO_SCHEDULE = [{"lr": 0.001, "iters": 160000}, {"lr": 0.0001, "iters": 40000}, {"lr": 0.00001, "iters": 40000}]
COCO_DATA = {"data.dataset": "coco2017", "data.test_set": None, "schedule.phases": COCO_SCHEDULE,
             "schedule.eval_every": 4000, "schedule.early_stopping": None, **COCO_MODEL}
LONG = {"aug.expansion": True, "schedule.scale": 2.0}

# name -> (description, dotted overrides applied on top of DEFAULT_YAML)
STAGES = {
    "voc": ("Detection training on VOC2012 from the ImageNet-pretrained ResNet (paper recipe)", {}),
    "coco": ("Optional second round, part 1: train on COCO train2017 from the ImageNet-pretrained ResNet",
             dict(COCO_DATA)),
    "voc_from_coco": ("Optional second round, part 2: fine-tune on VOC2012 starting from the COCO model",
                      {"init_from": "auto:coco"}),
    "voc_long": ("Longer schedule on VOC2012: zoom-out expansion augmentation and 2x iterations", dict(LONG)),
    "coco_long": ("Longer schedule on COCO: zoom-out expansion augmentation and 2x iterations",
                  {**COCO_DATA, **LONG}),
    "voc_from_coco_long": ("Longer schedule, fine-tune on VOC2012 from the long COCO model",
                           {**LONG, "init_from": "auto:coco_long"}),
}
# early stopping defaults differ for COCO (118k images, fewer evaluations)
COCO_ES = {"enabled": True, "patience": 5, "min_delta": 0.001, "advance_phase_on_plateau": True,
           "restore_best_on_decay": True}


def set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_overrides(items) -> dict:
    out = {}
    for item in items or []:
        key, sep, raw = str(item).partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=value, got: {item}")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(stage: str = "voc", overrides=None, dest=None, yaml_text: str | None = None) -> dict:
    """DEFAULT_YAML -> stage overrides -> user overrides (list of 'a.b=value' strings or a dict)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
    cfg = yaml.safe_load(yaml_text or DEFAULT_YAML)
    stage_over = copy.deepcopy(STAGES[stage][1])
    for k, v in stage_over.items():
        if k == "schedule.early_stopping" and v is None:
            v = copy.deepcopy(COCO_ES)
        set_dotted(cfg, k, v)
    user = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
    for k, v in user.items():
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
    if len(cfg["model"]["boxes_per_loc"]) != len(cfg["model"]["extras"]) + 2:
        raise ValueError("model.boxes_per_loc needs one entry per feature map (2 backbone maps + one per extra layer)")
    return cfg


def registered_name(cfg: dict) -> str:
    return cfg["mlflow"].get("registered_model_name") or f"ssd-{cfg['model']['backbone']}-{cfg['data']['dataset']}"


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
# boxes.py
# ====================================================================================================
"""SSD box machinery: default boxes, matching, MultiBox loss with hard negative mining, decoding + NMS (paper sec. 2)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], dim=-1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2]], dim=-1)


def box_iou_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [B,G,4] xyxy, b [P,4] xyxy -> IoU [B,G,P]."""
    lt = torch.max(a[:, :, None, :2], b[None, None, :, :2])
    rb = torch.min(a[:, :, None, 2:], b[None, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, :, None] + area_b[None, None, :] - inter).clamp(min=1e-9)


def prior_scales(num_maps: int, first_scale: float, min_scale: float, max_scale: float) -> list[float]:
    """Paper eq. (4): s_k = s_min + (s_max - s_min)/(m - 1) * (k - 1) for the later maps; the first map gets its own
    smaller scale; one extra scale (same step) serves the sqrt(s_k * s_k+1) box of the last map."""
    later = num_maps - 1
    step = (max_scale - min_scale) / max(later - 1, 1)
    return [first_scale] + [min_scale + step * k for k in range(later)] + [max_scale + step]


def generate_priors(feature_sizes, boxes_per_loc, first_scale=0.1, min_scale=0.2, max_scale=0.9, clip=True):
    """Default boxes as (cx, cy, w, h) in [0, 1]. 6 boxes per cell use aspect ratios {1, 1', 2, 1/2, 3, 1/3}; 4 omit 3, 1/3."""
    scales = prior_scales(len(feature_sizes), first_scale, min_scale, max_scale)
    out = []
    for k, (f, nb) in enumerate(zip(feature_sizes, boxes_per_loc, strict=True)):
        s, s_next = scales[k], scales[k + 1]
        r2, r3 = math.sqrt(2), math.sqrt(3)
        shapes = [(s, s), (math.sqrt(s * s_next),) * 2, (s * r2, s / r2), (s / r2, s * r2)]
        if nb == 6:
            shapes += [(s * r3, s / r3), (s / r3, s * r3)]
        elif nb != 4:
            raise ValueError("boxes_per_loc entries must be 4 or 6")
        for i in range(f):
            for j in range(f):
                for w, h in shapes:
                    out.append([(j + 0.5) / f, (i + 0.5) / f, w, h])
    t = torch.tensor(out, dtype=torch.float32)
    return t.clamp_(0.0, 1.0) if clip else t


def encode(matched_xyxy: torch.Tensor, priors: torch.Tensor, variances) -> torch.Tensor:
    m = xyxy_to_cxcywh(matched_xyxy)
    d_cxcy = (m[..., :2] - priors[..., :2]) / (variances[0] * priors[..., 2:])
    d_wh = torch.log(m[..., 2:].clamp(min=1e-6) / priors[..., 2:]) / variances[1]
    return torch.cat([d_cxcy, d_wh], dim=-1)


def decode(loc: torch.Tensor, priors: torch.Tensor, variances) -> torch.Tensor:
    cxcy = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
    wh = priors[..., 2:] * torch.exp((loc[..., 2:] * variances[1]).clamp(max=4.0))
    return cxcywh_to_xyxy(torch.cat([cxcy, wh], dim=-1))


def pad_targets(targets: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """list of {boxes [n,4] normalised xyxy, labels [n]} -> padded tensors and a validity mask."""
    g = max((len(t["labels"]) for t in targets), default=0)
    b = len(targets)
    boxes = torch.zeros(b, g, 4, device=device)
    labels = torch.zeros(b, g, dtype=torch.long, device=device)
    valid = torch.zeros(b, g, dtype=torch.bool, device=device)
    for i, t in enumerate(targets):
        n = len(t["labels"])
        if n:
            boxes[i, :n], labels[i, :n], valid[i, :n] = t["boxes"].to(device), t["labels"].to(device), True
    return boxes, labels, valid


def match_priors(gt_boxes, gt_labels, valid, priors_xyxy, threshold=0.5):
    """Paper sec. 2.2: every ground truth takes its best-overlapping default box, then every default box with
    IoU >= threshold to some ground truth is positive. Returns labels [B,P] (0 = background) and boxes [B,P,4]."""
    b, g, _ = gt_boxes.shape
    p = priors_xyxy.shape[0]
    if g == 0:
        return torch.zeros(b, p, dtype=torch.long, device=priors_xyxy.device), torch.zeros(b, p, 4,
                                                                                             device=priors_xyxy.device)
    iou = box_iou_batched(gt_boxes, priors_xyxy).masked_fill(~valid[:, :, None], -1.0)
    best_prior_idx = iou.argmax(dim=2)                      # [B,G]
    best_gt_iou, best_gt_idx = iou.max(dim=1)               # [B,P]
    bi, gi = torch.nonzero(valid, as_tuple=True)
    pi = best_prior_idx[bi, gi]
    best_gt_idx[bi, pi] = gi                                 # force each ground truth to keep its best prior
    best_gt_iou[bi, pi] = 2.0
    labels = gt_labels.gather(1, best_gt_idx)
    labels = torch.where(best_gt_iou >= threshold, labels, torch.zeros_like(labels))
    boxes = gt_boxes.gather(1, best_gt_idx[:, :, None].expand(-1, -1, 4))
    return labels, boxes


class MultiBoxLoss(nn.Module):
    """L = (L_conf + alpha * L_loc) / N, softmax confidence with 3:1 hard negative mining, smooth-L1 localisation."""

    def __init__(self, priors_cxcywh: torch.Tensor, iou_threshold=0.5, neg_pos_ratio=3, alpha=1.0,
                 variances=(0.1, 0.2)):
        super().__init__()
        self.register_buffer("priors", priors_cxcywh.clone(), persistent=False)
        self.register_buffer("priors_xyxy", cxcywh_to_xyxy(priors_cxcywh), persistent=False)
        self.iou_threshold, self.neg_pos_ratio, self.alpha, self.variances = iou_threshold, neg_pos_ratio, alpha, variances

    def forward(self, loc_p: torch.Tensor, conf_p: torch.Tensor, targets: list[dict]) -> dict:
        bsz, n_priors, n_cls = conf_p.shape
        gt_boxes, gt_labels, valid = pad_targets(targets, conf_p.device)
        labels, matched = match_priors(gt_boxes, gt_labels, valid, self.priors_xyxy, self.iou_threshold)
        pos = labels > 0
        loc_t = encode(matched, self.priors, self.variances)
        loc_loss = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction="sum")
        ce = F.cross_entropy(conf_p.reshape(-1, n_cls), labels.reshape(-1), reduction="none").view(bsz, n_priors)
        with torch.no_grad():
            neg_score = ce.detach().clone()
            neg_score[pos] = 0.0
            rank = neg_score.argsort(dim=1, descending=True).argsort(dim=1)
            num_pos = pos.sum(dim=1, keepdim=True)
            num_neg = (self.neg_pos_ratio * num_pos).clamp(max=n_priors - 1)
            neg = (rank < num_neg) & ~pos
        conf_loss = ce[pos | neg].sum()
        n = num_pos.sum().clamp(min=1).float()
        return {"total": (conf_loss + self.alpha * loc_loss) / n, "conf": conf_loss / n, "loc": loc_loss / n,
                "num_pos": num_pos.sum()}


@torch.no_grad()
def postprocess(loc, conf, priors_cxcywh, variances=(0.1, 0.2), score_thresh=0.01, nms_iou=0.45, max_det=200,
                pre_nms_top_k=2000) -> list[dict]:
    """Decode, drop scores below the threshold, per-class NMS (IoU 0.45), keep the best `max_det` (paper sec. 3.7).
    Boxes are normalised xyxy; labels are 1..C-1."""
    probs = conf.float().softmax(-1)[..., 1:]
    boxes_all = decode(loc.float(), priors_cxcywh, variances).clamp(0.0, 1.0)
    results = []
    for b in range(probs.shape[0]):
        sc = probs[b]
        pi, ci = (sc > score_thresh).nonzero(as_tuple=True)
        s = sc[pi, ci]
        if s.numel() > pre_nms_top_k:
            top = s.topk(pre_nms_top_k).indices
            pi, ci, s = pi[top], ci[top], s[top]
        bx = boxes_all[b][pi]
        keep = batched_nms(bx, s, ci, nms_iou)[:max_det]
        results.append({"boxes": bx[keep], "scores": s[keep], "labels": ci[keep] + 1})
    return results

# ====================================================================================================
# data.py
# ====================================================================================================
"""Detection data: VOC / COCO records, the paper's augmentation, datasets and loaders (+ a synthetic stand-in).

Training samples:  (image [3,S,S] normalised, {"boxes": [n,4] normalised xyxy in [0,1], "labels": [n] in 1..K})
Evaluation samples: (image, same-style target for the loss, meta with ORIGINAL-pixel boxes, difficult flags, size)
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
from torchvision import tv_tensors
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
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


# ------------------------------------------------------------------ transforms (paper section 2.2 and 3.6)
def train_transform(size: int, aug: dict):
    fill = [int(round(m * 255)) for m in IMAGENET_MEAN]   # paper: expansion canvas is filled with the mean
    steps = [v2.ToImage()]
    if aug.get("photometric", True):
        steps.append(v2.RandomPhotometricDistort(p=0.5))
    if aug.get("expansion", False):
        steps.append(v2.RandomZoomOut(fill=fill, side_range=(1.0, float(aug.get("expansion_max_side", 4.0))), p=0.5))
    if aug.get("crop", True):  # original image | patch with min IoU 0.1/0.3/0.5/0.7/0.9 | random patch
        steps.append(v2.RandomIoUCrop(min_scale=float(aug.get("crop_min_scale", 0.3)), max_scale=1.0,
                                      min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                                      sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], trials=40))
    steps.append(v2.Resize((size, size), antialias=True))
    if aug.get("flip", True):
        steps.append(v2.RandomHorizontalFlip(0.5))
    steps += [v2.SanitizeBoundingBoxes(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


def eval_transform(size: int):
    return v2.Compose([v2.Resize((size, size), antialias=True), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)


class DetectionDataset(Dataset):
    def __init__(self, source, size: int, train: bool, aug: dict | None = None, include_difficult: bool = True):
        self.source, self.size, self.train, self.include_difficult = source, size, train, include_difficult
        self.train_tf = train_transform(size, aug or {}) if train else None
        self.eval_tf = eval_transform(size)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        img, boxes, labels, difficult = self.source.get(i)
        if self.train:
            if not self.include_difficult:
                keep = ~difficult
                boxes, labels = boxes[keep], labels[keep]
            if len(labels) == 0:  # nothing to crop around: resize only
                return self.eval_tf(img), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
            tgt = {"boxes": tv_tensors.BoundingBoxes(torch.from_numpy(boxes), format="XYXY",
                                                     canvas_size=(img.height, img.width)),
                   "labels": torch.from_numpy(labels)}
            x, tgt = self.train_tf(img, tgt)
            b = (tgt["boxes"].as_subclass(torch.Tensor) / self.size).clamp(0, 1).float()
            return x, {"boxes": b, "labels": tgt["labels"].long()}
        w, h = img.size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        norm = (torch.from_numpy(boxes) / scale).clamp(0, 1)
        meta = {"boxes": boxes, "labels": labels, "difficult": difficult, "size": (w, h), "index": i}
        return self.eval_tf(img), {"boxes": norm, "labels": torch.from_numpy(labels)}, meta


def collate_train(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def collate_eval(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def split_ids(ids, val_fraction: float, seed: int):
    """Deterministic random split into (train, val)."""
    ids = sorted(ids)
    if val_fraction <= 0:
        return ids, []
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return sorted(ids[i] for i in perm[n_val:]), sorted(ids[i] for i in perm[:n_val])


def build_data(cfg: dict, paths: dict) -> dict:
    d, seed, size = cfg["data"], cfg["seed"], cfg["model"]["input_size"]
    rng = np.random.default_rng(seed)
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
        classes, mk = list(VOC_CLASSES), RecordSource
        inc = d.get("include_difficult_train", True)
    elif d["dataset"] == "coco2017":
        ensure_coco(paths["data"], paths["tmp"], ("train", "val"), d.get("keep_archives", False), d.get("download", True))
        train_recs, classes = coco_records(paths["data"], "train")
        if d.get("coco_train_subset"):
            train_recs = [train_recs[i] for i in sorted(rng.permutation(len(train_recs))[: d["coco_train_subset"]])]
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
    out = {"classes": classes,
           "train": DetectionDataset(train_src, size, True, cfg["aug"], inc),
           "train_eval": DetectionDataset(SubsetSource(train_src, n_eval), size, False) if n_eval else None,
           "val": DetectionDataset(mk(val_recs), size, False) if len(val_recs) else None,
           "test": DetectionDataset(mk(test_recs), size, False) if test_recs is not None and len(test_recs) else None}
    ids = [getattr(train_src, "records", [{}])[i].get("id", i) for i in range(min(len(train_src), 5000))] \
        if hasattr(train_src, "records") else [len(train_src)]
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
    nw = int(d["num_workers"])
    pin = device.type == "cuda"

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
# model.py
# ====================================================================================================
"""SSD with a ResNet base network: the paper's multi-map design (conv4_3 -> layer2, fc7 -> layer3, paper extras)."""
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d


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
"""Sanity checks (Deep Learning book Ch. 11 debugging), qualitative review, and next-step advice."""
import copy
import math

import numpy as np
import torch



def check_initial_loss(model, loss_fn, batch, device, num_classes, tol):
    """Untrained model: every sampled box (positives + 3x hard negatives) costs ~ln(C), so the confidence part of the
    loss is close to (1 + neg_pos_ratio) * ln(C)."""
    images, targets = batch[0].to(device), batch[1]
    m = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        loc, conf = m(images)
        losses = loss_fn(loc.float(), conf.float(), targets)
    expected = (1 + loss_fn.neg_pos_ratio) * math.log(num_classes)
    got = float(losses["conf"])
    return {"initial_conf_loss": got, "expected_conf_loss": expected, "initial_total": float(losses["total"]),
            "num_positives": int(losses["num_pos"]), "passed": bool(abs(got - expected) <= tol * expected)}


def overfit_tiny_batch(model, loss_fn, batch, device, steps=60, ratio=0.7, lr=0.01):
    """The network must be able to memorise a handful of images; failure points to a bug in data, loss or optimiser."""
    m = copy.deepcopy(model).to(device).train()
    images, targets = batch[0].to(device), batch[1]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loc, conf = m(images)
        loss = loss_fn(loc.float(), conf.float(), targets)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        hist.append(float(loss.detach()))
    final = float(np.mean(hist[-5:]))
    return {"overfit_initial_loss": hist[0], "overfit_final_loss": final, "overfit_ratio": final / max(hist[0], 1e-9),
            "passed": bool(final < ratio * hist[0])}


def run_sanity_checks(cfg, model, loss_fn, batch, device):
    s = cfg["sanity"]
    n = s["overfit_images"]
    small = (batch[0][:n], batch[1][:n])
    init = check_initial_loss(model, loss_fn, small, device, cfg["data"]["num_foreground"] + 1, s["initial_loss_tol"])
    over = overfit_tiny_batch(model, loss_fn, small, device, s["overfit_steps"], s["overfit_ratio"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"}, "initial_loss_ok": init["passed"],
            "overfit_ok": over["passed"], "passed": bool(init["passed"] and over["passed"])}


@torch.inference_mode()
def qualitative_grid(model, dataset, classes, device, out_path, thr=0.5, n=8, post_cfg=None, variances=(0.1, 0.2)):
    """Predicted (red) vs ground-truth (green) boxes on the first n images of a dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes

    model.eval()
    post_cfg = post_cfg or {}
    cols = 4
    n = min(n, len(dataset))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(n), strict=False):
        x, tgt, _ = dataset[i]
        loc, conf = model(x[None].to(device))
        det = postprocess(loc, conf, model.priors, variances, post_cfg.get("score_thresh", 0.01),
                          post_cfg.get("nms_iou", 0.45), post_cfg.get("max_detections", 200))[0]
        keep = det["scores"] >= thr
        s = x.shape[-1]
        u8 = (denormalize(x) * 255).round().byte()
        if len(tgt["labels"]):
            u8 = draw_bounding_boxes(u8, tgt["boxes"] * s, [classes[int(c) - 1] for c in tgt["labels"]], colors="lime",
                                     width=2)
        if keep.any():
            u8 = draw_bounding_boxes(u8, (det["boxes"][keep] * s).cpu(),
                                     [f"{classes[int(c) - 1]} {sc:.2f}" for c, sc in
                                      zip(det["labels"][keep].cpu(), det["scores"][keep].cpu(), strict=True)],
                                     colors="red", width=2)
        ax.imshow(u8.permute(1, 2, 0).numpy())
    fig.suptitle(f"green = ground truth, red = predictions (score >= {thr})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


LEVERS = {
    "overfitting": [
        "Stronger augmentation: aug.expansion=true (zoom-out), schedule.scale=2 (the paper's longer schedule), or lower aug.crop_min_scale",
        "More weight decay: schedule.weight_decay=0.001; freeze more of the backbone: model.freeze_up_to=layer2",
        "Add training data: data.voc_train_sets=[\"2012_trainval\",\"2007_trainval\"] or start from a COCO model (stage voc_from_coco)",
        "Smaller input or a smaller backbone: model.backbone=resnet34",
        "Keep early stopping and EMA on; the best-validation checkpoint is the one that is saved and tested"],
    "underfitting": [
        "Unfreeze more: model.freeze_up_to=\"\" (nothing frozen) or model.freeze_bn=false",
        "More capacity: model.input_size=512 (SSD512) or model.backbone=resnet101",
        "Train longer / lift the first-phase LR: schedule.scale=2, schedule.phases[0].lr=0.002",
        "Remove regularisation first: aug.expansion=false, aug.crop_min_scale=0.5, schedule.weight_decay=0.0001",
        "Check labels and boxes in qualitative_val.png: broken boxes look like underfitting"],
    "unstable": ["Lower the first-phase LR (x0.5), keep grad_clip=10, raise schedule.warmup_iters"],
    "not_learning": ["Check sanity.json, qualitative_val.png, class ids and box coordinates (VOC is 1-based)"],
    "still_improving": ["Train longer: schedule.scale=2; the validation metric had not plateaued"],
    "healthy": ["No red flags in the curves. Compare with the target, then evaluate once on the test set"],
}


def next_steps(diagnosis: dict) -> dict:
    verdicts = diagnosis.get("verdict", [])
    return {"verdict": verdicts, "levers": {v: LEVERS[v] for v in verdicts if v in LEVERS},
            "note": "A diagnosis is a hypothesis. Change ONE lever at a time and compare runs in MLflow."}

# ====================================================================================================
# serve.py
# ====================================================================================================
"""Inference and packaging: checkpoint predictor, TorchScript export, self-contained MLflow pyfunc, registry."""
import base64
import copy
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema
from PIL import Image


# A standalone MLflow "model from code" file: only torch / torchvision / numpy / pandas / PIL / mlflow are needed to
# load the registered model, never this package. Written to <dest>/tmp when registering.
PYFUNC_SOURCE = '''
import base64
import io
import json

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import batched_nms


class SSDDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.net = torch.jit.load(context.artifacts["torchscript"], map_location="cpu").eval()
        self.meta = json.load(open(context.artifacts["meta"]))
        self.priors = torch.from_numpy(np.load(context.artifacts["priors"])).float()
        self.mean = torch.tensor(self.meta["mean"]).view(3, 1, 1)
        self.std = torch.tensor(self.meta["std"]).view(3, 1, 1)

    def _decode(self, loc):
        v0, v1 = self.meta["variances"]
        p = self.priors
        cxcy = p[:, :2] + loc[:, :2] * v0 * p[:, 2:]
        wh = p[:, 2:] * torch.exp((loc[:, 2:] * v1).clamp(max=4.0))
        return torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1).clamp(0, 1)

    def predict(self, context, model_input, params=None):
        params = params or {}
        thr = float(params.get("score_threshold", self.meta["score_thresh"]))
        size, rows = self.meta["input_size"], []
        with torch.inference_mode():
            for b64 in model_input["image_b64"].tolist():
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                w, h = img.size
                x = torch.from_numpy(np.asarray(img.resize((size, size), Image.BILINEAR)).copy()).permute(2, 0, 1)
                x = ((x.float() / 255.0 - self.mean) / self.std)[None]
                loc, conf = self.net(x)
                probs = conf[0].float().softmax(-1)[:, 1:]
                boxes = self._decode(loc[0].float())
                pi, ci = (probs > thr).nonzero(as_tuple=True)
                sc = probs[pi, ci]
                if sc.numel() > self.meta["pre_nms_top_k"]:
                    top = sc.topk(self.meta["pre_nms_top_k"]).indices
                    pi, ci, sc = pi[top], ci[top], sc[top]
                bx = boxes[pi]
                keep = batched_nms(bx, sc, ci, self.meta["nms_iou"])[: self.meta["max_detections"]]
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                dets = [{"box": [round(float(v), 2) for v in (b * scale)], "score": round(float(s), 5),
                         "label_id": int(c) + 1, "label": self.meta["classes"][int(c)]}
                        for b, s, c in zip(bx[keep], sc[keep], ci[keep])]
                rows.append(json.dumps({"width": w, "height": h, "detections": dets}))
        return pd.DataFrame({"detections": rows})


mlflow.models.set_model(SSDDetector())
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
        self.tf = eval_transform(self.cfg["model"]["input_size"])

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
    def predict(self, images, score_threshold=0.5, max_detections=200) -> list[dict]:
        pp, var = self.cfg["postprocess"], tuple(self.cfg["loss"]["variances"])
        out = []
        for item in images:
            img = self._pil(item)
            w, h = img.size
            loc, conf = self.model(self.tf(img)[None].to(self.device))
            d = postprocess(loc, conf, self.model.priors, var, min(score_threshold, pp["score_thresh"]), pp["nms_iou"],
                            max_detections, pp["pre_nms_top_k"])[0]
            keep = d["scores"] >= score_threshold
            scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=d["boxes"].device)
            dets = [{"box": [round(float(v), 2) for v in (b * scale)], "score": round(float(s), 5), "label_id": int(c),
                     "label": self.classes[int(c) - 1]}
                    for b, s, c in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep], strict=True)]
            out.append({"width": w, "height": h, "detections": dets})
        return out


def export_torchscript(model, path) -> None:
    """Trace the raw network (images -> loc, conf); box decoding and NMS live in the pyfunc."""
    m = copy.deepcopy(model).cpu().eval()
    size = m.input_size
    with torch.no_grad():
        ts = torch.jit.trace(m, torch.zeros(1, 3, size, size), check_trace=False)
    ts.save(str(path))


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def package_model(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, out_dir) -> dict:
    """Save a self-contained model folder (checkpoint, TorchScript, priors, meta, pyfunc source), log it to the run as an
    MLflow pyfunc model and verify that it reloads and predicts identically in a FRESH process."""
    uri = setup_mlflow(cfg, paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]) / f"pkg_{run_id}")))
    shutil.copy2(ckpt, out / "checkpoint.pt")
    model, ccfg, classes = load_checkpoint(out / "checkpoint.pt")
    export_torchscript(model, out / "torchscript.pt")
    np.save(out / "priors.npy", model.priors.numpy())
    pp = ccfg["postprocess"]
    (out / "meta.json").write_text(json.dumps({
        "classes": classes, "input_size": ccfg["model"]["input_size"], "variances": ccfg["loss"]["variances"],
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "score_thresh": pp["score_thresh"],
        "nms_iou": pp["nms_iou"], "max_detections": pp["max_detections"], "pre_nms_top_k": pp["pre_nms_top_k"]}))
    (out / "ssd_pyfunc.py").write_text(PYFUNC_SOURCE)
    b64 = _png_b64(example_image)
    pd.DataFrame({"image_b64": [b64]}).to_csv(out / "example.csv", index=False)
    (out / "expected.json").write_text(json.dumps(Predictor(out / "checkpoint.pt").predict([example_image], score_threshold=0.0)[0]))
    signature = ModelSignature(inputs=Schema([ColSpec("string", "image_b64")]),
                               outputs=Schema([ColSpec("string", "detections")]),
                               params=ParamSchema([ParamSpec("score_threshold", "float", float(pp["score_thresh"]))]))
    import torchvision
    reqs = [f"torch=={torch.__version__.split('+')[0]}", f"torchvision=={torchvision.__version__.split('+')[0]}",
            f"mlflow=={mlflow.__version__}", "numpy", "pandas", "pillow"]
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model", python_model=str(out / "ssd_pyfunc.py"),
            artifacts={"torchscript": str(out / "torchscript.pt"), "priors": str(out / "priors.npy"),
                       "meta": str(out / "meta.json")},
            signature=signature, input_example=pd.DataFrame({"image_b64": [b64]}), pip_requirements=reqs)
    proc = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET, info.model_uri, uri, str(out / "example.csv"),
                           str(out / "expected.json"), "0.05"], cwd=str(out), capture_output=True, text=True)
    ok = proc.returncode == 0
    print(proc.stdout.strip(), proc.stderr.strip()[-400:] if not ok else "")
    (out / "README.txt").write_text(
        "SSD model folder\n  checkpoint.pt   full checkpoint (config + classes); Predictor('checkpoint.pt').predict([image])\n"
        "  torchscript.pt  traced network (images -> loc, conf); priors.npy are the default boxes; meta.json the decoding settings\n"
        f"  MLflow pyfunc   logged in run {run_id} as 'model' (input column image_b64, output column detections)\n")
    return {"model_dir": str(out), "model_uri": info.model_uri, "load_verified": ok, "tracking_uri": uri}


def register_candidate(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, model_name: str | None = None) -> dict:
    """package_model + registry: set the alias `candidate` if the fresh-process check passed."""
    setup_mlflow(cfg, paths)
    if on_databricks() and cfg["mlflow"].get("backend", "auto") in ("auto", "databricks"):
        mlflow.set_registry_uri("databricks-uc")
    name = model_name or registered_name(cfg)
    if on_databricks() and name.count(".") != 2:
        raise ValueError(f"Unity Catalog needs a three-level model name catalog.schema.model, got {name!r}: set "
                         "mlflow.registered_model_name")
    pkg = package_model(cfg, paths, run_id, example_image, Path(paths["tmp"]) / f"register_{run_id}")
    client = MlflowClient()
    mv = mlflow.register_model(pkg["model_uri"], name)
    run = client.get_run(run_id)
    for key, val in {"source_run_id": run_id, "stage": cfg["stage"], "data_version": run.data.tags.get("data_version", ""),
                     "load_verified": str(pkg["load_verified"]).lower(),
                     "best_val_metric": f"{run.data.metrics.get('best_val_metric', float('nan')):.4f}",
                     "test_metric": f"{run.data.metrics[f'test_{cfg['metric']['name']}']:.4f}"
                     if f"test_{cfg['metric']['name']}" in run.data.metrics else "not evaluated"}.items():
        client.set_model_version_tag(name, mv.version, key, val)
    if pkg["load_verified"]:
        client.set_registered_model_alias(name, "candidate", mv.version)
    else:
        print("load-and-predict check FAILED: alias `candidate` not set")
    out = {"name": name, "version": mv.version, "model_uri": pkg["model_uri"], "load_verified": pkg["load_verified"]}
    print(out)
    return out

# ====================================================================================================
# train.py
# ====================================================================================================
"""SSD training: paper recipe (SGD, step LR phases, weight decay, hard negative mining) + early stopping, EMA, MLflow."""
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


CSV_FIELDS = ["epoch", "iteration", "phase", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "train_conf",
              "train_loc", "val_conf", "val_loc", "val_map50", "val_map50_voc07", "grad_norm", "imgs_per_s", "elapsed_s"]


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
    return [{"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)}
            for t in targets]


def infinite(loader):
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def evaluate_model(model, loader, device, loss_fn, cfg, classes, fast=True, keep=False):
    """One pass: loss components + detections -> VOC-protocol metrics (IoU 0.5 only when fast)."""
    model.eval()
    ev = DetectionEvaluator(classes, [0.5] if fast else None)
    pp, var = cfg["postprocess"], cfg["loss"]["variances"]
    sums, n = defaultdict(float), 0
    preds, gts = [], []
    mode = cfg["schedule"]["amp"]
    with torch.inference_mode():
        for images, targets, metas in loader:
            images = images.to(device, non_blocking=True)
            with amp_context(device, mode):
                loc, conf = model(images)
            ls = loss_fn(loc.float(), conf.float(), targets_to(targets, device))
            for k in ("total", "conf", "loc"):
                sums[k] += float(ls[k]) * len(metas)
            n += len(metas)
            dets = postprocess(loc, conf, model.priors, var, pp["score_thresh"], pp["nms_iou"], pp["max_detections"],
                               pp["pre_nms_top_k"])
            for d, m in zip(dets, metas, strict=True):
                w, h = m["size"]
                p = {"boxes": d["boxes"].cpu().numpy() * np.array([w, h, w, h], np.float32),
                     "scores": d["scores"].cpu().numpy(), "labels": d["labels"].cpu().numpy()}
                g = {"boxes": m["boxes"], "labels": m["labels"], "difficult": m["difficult"]}
                ev.update([p], [g])
                if keep:
                    preds.append(p)
                    gts.append(g)
    res = ev.compute()
    res.update({f"loss_{k}": v / max(n, 1) for k, v in sums.items()})
    return (res, preds, gts) if keep else res


def resolve_init(cfg: dict, paths: dict):
    v = cfg.get("init_from")
    if not v:
        return None
    if str(v).startswith("auto:"):
        p = Path(paths["checkpoints"]) / f"{str(v).split(':', 1)[1]}_best.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run the stage that produces it first "
                                    f"({str(v).split(':', 1)[1]}), or set init_from to a checkpoint path")
        return p
    return Path(v)


def make_optimizer(model, s: dict):
    return torch.optim.SGD(model.param_groups(s["weight_decay"]), lr=s["phases"][0]["lr"], momentum=s["momentum"],
                           nesterov=s.get("nesterov", False))


def save_checkpoint(path, model_state, cfg, classes, it, val_metric, ema_used):
    torch.save({"state_dict": model_state, "cfg": cfg, "classes": classes, "num_foreground": len(classes),
                "iteration": it, "val_metric": val_metric, "metric_name": cfg["metric"]["name"], "ema": ema_used,
                "stage": cfg["stage"]}, path)


def fit(cfg, data, loaders, model, loss_fn, device, out_dir: Path, run_id: str):
    s, es_cfg = cfg["schedule"], cfg["schedule"]["early_stopping"]
    classes, metric_name = data["classes"], cfg["metric"]["name"]
    optimizer = make_optimizer(model, s)
    sch = PhaseSchedule(optimizer, [(p["lr"], p["iters"]) for p in s["phases"]], s["scale"], s["warmup_iters"],
                        s["warmup_factor"])
    has_val = loaders["val"] is not None
    es = PhaseEarlyStopper(es_cfg["patience"], es_cfg["min_delta"], bool(es_cfg["enabled"] and has_val))
    ema_cfg = s["ema"]
    ema = ModelEMA(model, ema_cfg["decay"], ema_cfg["tau"]) if ema_cfg["enabled"] else None
    eval_model = ema.module if ema is not None else model
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and s["amp"] == "fp16") else None
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    best_path, last_path, csv_path = ck / "best.pt", ck / "last.pt", out_dir / "metrics.csv"
    it = 0
    if cfg.get("resume_from"):
        st = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["sched"])
        es.load_state_dict(st["es"])
        if ema is not None and st.get("ema"):
            ema.load_state_dict(st["ema"])
        it = st["iteration"]
        prev = Path(cfg["resume_from"]).parent.parent  # carry the best checkpoint and the history into this run
        for name, dst in (("checkpoints/best.pt", best_path), ("metrics.csv", csv_path)):
            if (prev / name).exists() and not dst.exists():
                shutil.copy2(prev / name, dst)
    best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
    use_cl = device.type == "cuda" and s.get("channels_last", True)
    if use_cl:
        model.to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    gen = infinite(loaders["train"])
    win = defaultdict(float)
    n_win = n_imgs = 0
    gn_sum = 0.0
    t_win = t_start = time.time()
    resumed = csv_path.exists()
    stop_reason, last_row = "schedule_complete", {}
    f = open(csv_path, "a" if resumed else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not resumed:
        writer.writeheader()
    model.train()
    try:
        while True:
            epoch, (images, targets) = next(gen)
            images = images.to(device, non_blocking=True)
            if use_cl:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets_to(targets, device)
            lr = sch.step()
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, s["amp"]):
                loc, conf = model(images)
            losses = loss_fn(loc.float(), conf.float(), targets)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {it}: lower schedule.phases[0].lr or check the boxes")
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
            it += 1
            for k in ("total", "conf", "loc"):
                win[k] += float(losses[k].detach())
            gn_sum += float(gn)
            n_win += 1
            n_imgs += len(images)
            if it % s["log_every"] == 0:
                dt = max(time.time() - t_win, 1e-9)
                mlflow.log_metrics({"train_loss_iter": win["total"] / n_win, "lr": lr, "grad_norm": gn_sum / n_win,
                                    "imgs_per_s": n_imgs / dt}, step=it)
            if it % s["eval_every"] == 0 or sch.phase_done():
                dt = max(time.time() - t_win, 1e-9)
                row = {"epoch": it, "iteration": it, "phase": sch.phase, "lr": lr,
                       "train_loss": win["total"] / n_win, "train_conf": win["conf"] / n_win,
                       "train_loc": win["loc"] / n_win, "grad_norm": gn_sum / n_win, "imgs_per_s": n_imgs / dt,
                       "elapsed_s": time.time() - t_start}
                value = None
                if has_val:
                    v = evaluate_model(eval_model, loaders["val"], device, loss_fn, cfg, classes)
                    value = v[metric_name]
                    row.update({"val_loss": v["loss_total"], "val_conf": v["loss_conf"], "val_loc": v["loss_loc"],
                                "val_metric": value, "val_map50": v["map50"], "val_map50_voc07": v["map50_voc07"]})
                if loaders.get("train_eval") is not None:
                    row["train_metric"] = evaluate_model(eval_model, loaders["train_eval"], device, loss_fn, cfg,
                                                         classes)[metric_name]
                model.train()
                writer.writerow(row)
                f.flush()
                mlflow.log_metrics({k: v for k, v in row.items() if k not in ("epoch", "iteration") and v is not None
                                    and not (isinstance(v, float) and math.isnan(v))}, step=it)
                last_row = row
                gap = "" if "train_metric" not in row or value is None else f" | gap {row['train_metric'] - value:+.3f}"
                print(f"it {it:6d} ep {epoch:3d} ph {sch.phase} | loss {row['train_loss']:.3f}"
                      + (f" | val loss {row['val_loss']:.3f} | val {metric_name} {value:.4f}" if has_val else "")
                      + gap + f" | lr {lr:.5f} | {row['imgs_per_s']:.0f} img/s", flush=True)
                improved = es.update(value, it) if has_val else True
                if improved:
                    best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
                    save_checkpoint(best_path, best_state, cfg, classes, it, value, ema is not None)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sched": sch.state_dict(),
                            "es": es.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                            "iteration": it}, last_path)
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
                    mlflow.log_metric("phase_change_iteration", it, step=it)
                    if es_cfg["restore_best_on_decay"] and has_val:
                        model.load_state_dict(best_state)
                        if ema is not None:
                            ema.module.load_state_dict(best_state)
                        optimizer.state.clear()
                    print(f"  -> LR phase {sch.phase} ({'plateau' if plateau else 'phase complete'}), "
                          f"lr {sch.phases[sch.phase][0]}", flush=True)
    finally:
        f.close()
    if not has_val:  # no validation split: the final weights are the model
        save_checkpoint(best_path, {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}, cfg, classes, it,
                        None, ema is not None)
    return {"best_path": best_path, "csv_path": csv_path, "iterations": it, "stop_reason": stop_reason,
            "best_val_metric": es.best if has_val else None, "best_iteration": es.best_it, "last_row": last_row}


def run_training(cfg: dict, paths: dict) -> dict:
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    model = build_model(cfg).to(device)
    init_report = None
    init_path = resolve_init(cfg, paths)
    if init_path is not None:
        st = torch.load(init_path, map_location="cpu", weights_only=False)
        same = st["num_foreground"] == cfg["data"]["num_foreground"]
        init_report = load_compatible(model, st["state_dict"], () if same else ("conf_heads.",))
        print(f"init from {init_path}: loaded {init_report['loaded']} tensors, skipped {len(init_report['skipped'])}")
    loss_fn = MultiBoxLoss(model.priors, cfg["loss"]["iou_threshold"], cfg["loss"]["neg_pos_ratio"], cfg["loss"]["alpha"],
                           tuple(cfg["loss"]["variances"])).to(device)
    uri = setup_mlflow(cfg, paths)
    with mlflow.start_run(run_name=cfg.get("run_label") or cfg["stage"]) as run:
        run_id = run.info.run_id
        out_dir = Path(paths["runs"]) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "running"}))   # lets a rerun resume
        mlflow.log_params({k: str(v)[:490] for k, v in flatten({kk: vv for kk, vv in cfg.items() if kk != "dest"}).items()})
        mlflow.log_params({"num_parameters": count_params(model), "num_trainable": count_params(model, True),
                           "num_priors": model.num_priors, "feature_sizes": str(model.feature_sizes)})
        mlflow.set_tags({"stage": cfg["stage"], "task": "object-detection", "git_commit": git_commit(),
                         "data_version": data["data_version"], "metric": cfg["metric"]["name"],
                         "metric_target": str(cfg["metric"]["target_value"]), "test_evaluated": "false",
                         "device": str(device), "tracking_uri": uri, "dest": str(paths["dest"]),
                         "init_from": str(init_path) if init_path else "imagenet-pretrained backbone",
                         "pipeline_step": str(cfg.get("run_label", ""))})
        print(f"run {run_id} | stage {cfg['stage']} | device {device} | params {count_params(model):,} "
              f"(trainable {count_params(model, True):,}) | priors {model.num_priors} | data {data['data_version']}")
        if cfg["sanity"]["enabled"]:
            batch = next(iter(loaders["train"]))
            rep = run_sanity_checks(cfg, model, loss_fn, batch, device)
            (out_dir / "sanity.json").write_text(json.dumps(rep, indent=2))
            mlflow.log_dict(rep, "diagnostics/sanity.json")
            mlflow.set_tag("sanity_passed", str(rep["passed"]).lower())
            print("sanity:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
            if cfg["sanity"]["fail_hard"] and not rep["passed"]:
                raise RuntimeError(f"sanity checks failed: {json.dumps(rep)}")
        set_seed(cfg["seed"])
        res = fit(cfg, data, loaders, model, loss_fn, device, out_dir, run_id)
        mlflow.set_tags({"stop_reason": res["stop_reason"], "best_iteration": str(res["best_iteration"])})
        mlflow.log_metric("total_iterations", res["iterations"])
        summary = {"run_id": run_id, "stage": cfg["stage"], "iterations": res["iterations"],
                   "stop_reason": res["stop_reason"], "best_iteration": res["best_iteration"],
                   "best_val_metric": res["best_val_metric"], "out_dir": str(out_dir)}

        state = torch.load(res["best_path"], map_location="cpu", weights_only=False)
        best = build_model(cfg, pretrained=False)
        best.load_state_dict(state["state_dict"])
        best.to(device).eval()
        if loaders["val"] is not None:
            v, preds, gts = evaluate_model(best, loaders["val"], device, loss_fn, cfg, data["classes"], fast=False, keep=True)
            mlflow.log_metrics({f"final_val_{k}": v[k] for k in ("map50", "map50_voc07", "map75", "map", "mar100")})
            mlflow.log_metric("best_val_metric", res["best_val_metric"])
            (out_dir / "val_per_class_ap50.json").write_text(json.dumps(v["ap50_per_class"], indent=2))
            errs = error_breakdown(preds, gts, 0.5)
            (out_dir / "val_error_breakdown.json").write_text(json.dumps(errs, indent=2))
            mlflow.log_dict(errs, "diagnostics/val_error_breakdown.json")
            if loaders.get("train_eval") is not None:
                tm = evaluate_model(best, loaders["train_eval"], device, loss_fn, cfg, data["classes"])[cfg["metric"]["name"]]
                gap = tm - v[cfg["metric"]["name"]]
                flag = bool(gap > cfg["schedule"]["overfit_gap_warn"])
                mlflow.log_metrics({"final_train_metric": tm, "generalisation_gap": gap})
                mlflow.set_tag("overfit_warning", str(flag).lower())
                summary.update({"train_metric_at_best": tm, "generalisation_gap": gap, "overfit_warning": flag})
                if flag:
                    print(f"WARNING: train - val {cfg['metric']['name']} gap {gap:.3f} exceeds "
                          f"{cfg['schedule']['overfit_gap_warn']}: see next_steps.json (overfitting levers)")
            try:
                qualitative_grid(best, data["val"], data["classes"], device, out_dir / "qualitative_val.png",
                                 post_cfg=cfg["postprocess"], variances=tuple(cfg["loss"]["variances"]))
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
        stable = Path(paths["checkpoints"]) / f"{cfg['stage']}_best.pt"
        shutil.copy2(res["best_path"], stable)   # stable name the next stage can start from
        for p in out_dir.glob("*"):
            if p.is_file():
                mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(res["best_path"]), "checkpoints")
        mlflow.log_dict({p: metadata.version(p) for p in ("torch", "torchvision")}, "config/versions.json")
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "complete"}))
        summary["stable_checkpoint"] = str(stable)
        summary["target_met"] = bool(res["best_val_metric"] is not None and res["best_val_metric"] >= cfg["metric"]["target_value"])
        print(json.dumps(summary, indent=2))
        return summary


def benchmark(cfg: dict, paths: dict, steps: int = 20, loader_batches: int = 20) -> dict:
    """Model step time and augmentation-pipeline throughput on THIS machine (synthetic images, no downloads)."""
    device = select_device(cfg["device"])
    set_seed(cfg["seed"])
    model = build_model(cfg, pretrained=False).to(device)
    loss_fn = MultiBoxLoss(model.priors, cfg["loss"]["iou_threshold"], cfg["loss"]["neg_pos_ratio"]).to(device)
    opt = make_optimizer(model, cfg["schedule"])
    bs, size = cfg["data"]["batch_size"], cfg["model"]["input_size"]
    x = torch.randn(bs, 3, size, size, device=device)
    tg = [{"boxes": torch.tensor([[0.1, 0.1, 0.5, 0.6], [0.4, 0.3, 0.9, 0.9]], device=device),
           "labels": torch.tensor([1, 2], device=device)} for _ in range(bs)]
    model.train()

    def step():
        opt.zero_grad()
        with amp_context(device, cfg["schedule"]["amp"]):
            loc, conf = model(x)
        loss_fn(loc.float(), conf.float(), tg)["total"].backward()
        opt.step()
    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec_step = (time.time() - t0) / steps
    src = SyntheticSource(max(bs * 8, 64), 20, 500, cfg["seed"])  # VOC-sized images exercise the real augmentation cost
    ds = DetectionDataset(src, size, True, cfg["aug"])
    nw = int(cfg["data"]["num_workers"])
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_train,
                                     persistent_workers=False)
    t0, n = None, 0
    for i, _ in enumerate(ld):
        if i == 2:
            t0 = time.time()
        if i >= 2:
            n += bs
        if i >= loader_batches:
            break
    loader_ips = n / max(time.time() - (t0 or time.time()), 1e-9) if t0 else float("nan")
    sch = cfg["schedule"]
    total = sum(round(p["iters"] * sch["scale"]) for p in sch["phases"])
    ips_model = bs / sec_step
    it_s = min(1 / sec_step, loader_ips / bs) if loader_ips == loader_ips else 1 / sec_step
    out = {"device": str(device), "batch_size": bs, "model_step_s": round(sec_step, 3), "model_img_per_s": round(ips_model, 1),
           "augmentation_img_per_s": round(loader_ips, 1), "bottleneck": "data loading" if loader_ips < ips_model else "model",
           "schedule_iterations_max": total, "est_hours_if_run_to_the_end": round(total / it_s / 3600, 3),
           "est_minutes_if_run_to_the_end": round(total / it_s / 60, 2),
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
    run = client.get_run(run_id)
    count = int(run.data.tags.get("test_evaluation_count", "0"))
    if count >= 1 and not force:
        raise RuntimeError(f"run {run_id} was already evaluated on the test set {count} time(s). Tuning on the test set "
                           "invalidates it; pass force=True only to re-verify (the count is recorded).")
    if not cfg["data"].get("test_set") and cfg["data"]["dataset"] != "synthetic":
        raise ValueError("this stage has no test set (COCO stage: use the validation metrics; test on VOC after fine-tuning)")
    device = select_device(cfg["device"])
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    ckpt = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoints/best.pt",
                                               dst_path=str(Path(paths["tmp"])))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    loss_fn = MultiBoxLoss(model.priors, cfg["loss"]["iou_threshold"], cfg["loss"]["neg_pos_ratio"]).to(device)
    res, preds, gts = evaluate_model(model, loaders["test"], device, loss_fn, cfg, data["classes"], fast=False, keep=True)
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
               "test_map50_voc07": res["map50_voc07"], "test_map": res["map"], "test_map75": res["map75"],
               "test_mar100": res["mar100"], "target": target, "target_met": res[metric] >= target,
               "evaluation_count": count + 1, "worst_classes_ap50": worst, "error_breakdown": errs}
    print(json.dumps(summary, indent=2))
    return summary

# ====================================================================================================
# pipeline.py
# ====================================================================================================
"""The whole job as one sequence: load datasets -> detection training (+test, save) -> second round (+save) -> longer schedule
(+save). Inputs: a path and a device mode (cpu | gpu | combination). Interrupted runs resume where they stopped."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch


DEVICE_MODES = ("cpu", "gpu", "combination")
DEVICE_HELP = {
    "cpu": "everything runs on the CPU (very slow for SSD: use it for small checks)",
    "gpu": "the model trains on the GPU; CPU workers only load and augment images; all datasets are prepared first",
    "combination": "like gpu, and the CPU also works in parallel: the COCO download for the later stages runs in the "
                   "background while the GPU trains the first stage",
}

# key, config stage, title, start from, test on VOC2007 test, saved model name
STEPS = [
    {"key": "detection", "stage": "voc", "title": "Detection training on VOC2012", "init": None, "test": True,
     "save": "01_detection_voc"},
    {"key": "coco", "stage": "coco", "title": "Second round, part 1: training on COCO", "init": None, "test": False,
     "save": None},
    {"key": "second_round", "stage": "voc_from_coco", "title": "Second round, part 2: fine-tuning on VOC2012 (final model)",
     "init": "auto", "test": True, "save": "02_final_second_round"},
    {"key": "coco_long", "stage": "coco_long", "title": "Longer schedule, part 1: COCO, starting from the saved final model",
     "init": "saved:second_round", "test": False, "save": None},
    {"key": "long_final", "stage": "voc_from_coco_long", "title": "Longer schedule, part 2: VOC2012 (final model)",
     "init": "auto", "test": True, "save": "03_final_longer_schedule"},
]
STEP = {s["key"]: s for s in STEPS}


def choose_device_mode(value=None) -> str:
    """Validate the answer, or ask for it when running in a terminal."""
    names = {"1": "cpu", "2": "gpu", "3": "combination"}
    if value:
        v = names.get(str(value).strip(), str(value).strip().lower())
        if v not in DEVICE_MODES:
            raise ValueError(f"device must be one of {DEVICE_MODES}, got {value!r}")
        return v
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("Choose where to train: pass cpu, gpu or combination as the second argument.")
    print("Where should training run?")
    for i, m in enumerate(DEVICE_MODES, 1):
        print(f"  {i}) {m:12s} {DEVICE_HELP[m]}")
    while True:
        ans = input("Enter 1, 2 or 3: ").strip()
        if ans in names or ans in DEVICE_MODES:
            return names.get(ans, ans)


def device_overrides(mode: str, cpu_count: int | None = None) -> dict:
    n = cpu_count or os.cpu_count() or 4
    if mode == "cpu":
        workers = max(2, n // 4)
        return {"device": "cpu", "schedule.amp": "none", "schedule.channels_last": False, "data.num_workers": workers}
    return {"device": "cuda", "schedule.amp": "bf16", "schedule.channels_last": True,
            "data.num_workers": min(20, max(2, n - 4))}


def check_device(mode: str) -> str:
    if mode in ("gpu", "combination"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"You chose '{mode}' but PyTorch sees no GPU. Choose 'cpu', or use a GPU cluster with the "
                               "Machine Learning runtime.")
        return torch.cuda.get_device_name(0)
    return "cpu"


class PipelineContext:
    def __init__(self, paths, mode, overrides, stage_overrides, state, state_path):
        self.paths, self.mode, self.overrides = paths, mode, overrides
        self.stage_overrides, self.state, self.state_path = stage_overrides or {}, state, state_path
        self.background = None

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def cfg(self, key: str, need_init: bool = True) -> dict:
        spec = STEP[key]
        over = {**device_overrides(self.mode), **parse_overrides(self.overrides), **parse_overrides(self.stage_overrides.get(key))}
        if spec["init"] == "saved:second_round":
            model_dir = self.state["stages"].get("second_round", {}).get("model_dir")
            if model_dir:
                over["init_from"] = str(Path(model_dir) / "checkpoint.pt")
            elif need_init:
                raise RuntimeError("the longer schedule starts from the saved final model of the second round, which is missing")
        cfg = load_config(spec["stage"], over, dest=self.paths["dest"])
        cfg["run_label"] = f"{[s['key'] for s in STEPS].index(key) + 1}_{key}"
        return cfg


def start_pipeline(path, device_mode=None, overrides=None, stage_overrides=None) -> PipelineContext:
    """Create the folder layout under `path`, route every cache there, check the device, load (or start) the state file."""
    mode = choose_device_mode(device_mode)
    paths = setup_environment(path)
    gpu_name = check_device(mode)
    if mode == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - device_overrides("cpu")["data.num_workers"]))
    state_path = Path(paths["dest"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    ctx = PipelineContext(paths, mode, overrides, stage_overrides, state, state_path)
    state["device_mode"] = mode
    ctx.save()
    ctx.log(f"path: {paths['dest']} | device mode: {mode} ({DEVICE_HELP[mode]}) | {gpu_name}")
    done = [k for k, v in state["stages"].items() if v.get("train_done")]
    if done:
        ctx.log(f"found earlier progress, will skip finished steps: {done}")
    return ctx


# ------------------------------------------------------------------ step 1: datasets
def prepare_second_round_data(ctx: PipelineContext, download: bool = True, index: bool = True) -> None:
    cfg = ctx.cfg("coco")
    if cfg["data"]["dataset"] != "coco2017":
        return
    d = cfg["data"]
    if download:
        ensure_coco(ctx.paths["data"], ctx.paths["tmp"], ("val", "train"), d.get("keep_archives", False), d.get("download", True))
    if index:
        for split in ("val", "train"):
            coco_records(ctx.paths["data"], split)


class BackgroundPrep:
    """Runs a function in a thread (the CPU side of 'combination' mode) and re-raises its error at join()."""

    def __init__(self, fn, *args, **kwargs):
        self.fn, self.args, self.kwargs, self.error, self.started, self.finished = fn, args, kwargs, None, time.time(), None
        self.thread = threading.Thread(target=self._run, daemon=True, name="ssd-background-prep")
        self.thread.start()

    def _run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except BaseException as e:  # noqa: BLE001 - reported at join()
            self.error = e
        finally:
            self.finished = time.time()

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"background data preparation failed: {self.error}") from self.error


def step_load_datasets(ctx: PipelineContext) -> dict:
    """VOC2012 trainval + VOC2007 test first. COCO next (gpu/cpu), or in the background while the GPU trains (combination)."""
    ctx.log("step 1: loading datasets")
    data = build_data(ctx.cfg("detection"), ctx.paths)
    info = {"voc_data_version": data["data_version"], "voc_train": len(data["train"]), "voc_val": len(data["val"] or []),
            "voc_test": len(data["test"] or [])}
    if ctx.mode == "combination":
        ctx.background = BackgroundPrep(prepare_second_round_data, ctx, True, False)   # download only, in the background
        ctx.log("COCO is downloading in the background while the GPU trains the first stage")
    else:
        prepare_second_round_data(ctx)
    ctx.state["datasets"] = info
    ctx.save()
    ctx.log(f"VOC ready: {info}")
    return info


def step_estimate_time(ctx: PipelineContext) -> dict:
    """Measure this machine for a moment and print the longest the whole pipeline could take."""
    cfg = ctx.cfg("detection")
    b = benchmark(cfg, ctx.paths, steps=5, loader_batches=6)
    per_it = 1 / min(1 / b["model_step_s"], b["augmentation_img_per_s"] / cfg["data"]["batch_size"]) if b["augmentation_img_per_s"] == b["augmentation_img_per_s"] else b["model_step_s"]
    total = sum(round(p["iters"] * c["schedule"]["scale"]) for k in STEP for c in [ctx.cfg(k, need_init=False)] for p in c["schedule"]["phases"])
    est = {"seconds_per_iteration": round(per_it, 3), "bottleneck": b["bottleneck"], "max_iterations_all_steps": total,
           "max_hours_all_steps": round(total * per_it / 3600, 1)}
    ctx.log(f"time estimate (upper bound, early stopping usually ends steps sooner, evaluation is extra): {est}")
    ctx.state["estimate"] = est
    ctx.save()
    return est


# ------------------------------------------------------------------ steps 2-4
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


def _example_image(ctx: PipelineContext, cfg: dict):
    data = build_data(cfg, ctx.paths)
    ds = data["val"] or data["test"]
    return ds.source.get(0)[0]


def run_step(ctx: PipelineContext, key: str) -> dict:
    """Train one stage (resuming it if it was interrupted), then test and save it where the plan says so."""
    spec, st = STEP[key], ctx.state["stages"].setdefault(key, {})
    cfg = ctx.cfg(key)
    ctx.log(f"{spec['title']}  [stage {spec['stage']}]")
    if key == "coco" and ctx.background is not None:
        ctx.log("waiting for the background COCO download to finish ...")
        ctx.background.join()
        ctx.background = None
    if key == "coco":
        prepare_second_round_data(ctx)      # builds the compact COCO index if it does not exist yet
    if not st.get("train_done"):
        resume = find_resumable(ctx.paths, spec["stage"])
        if resume is not None:
            cfg["resume_from"] = str(resume)
            ctx.log(f"resuming the interrupted run from {resume}")
        summary = run_training(cfg, ctx.paths)
        _supersede_unfinished(ctx.paths, spec["stage"], summary["run_id"])
        st.update({"train_done": True, "run_id": summary["run_id"], "summary": summary})
        ctx.save()
    else:
        ctx.log("training already done, skipping")
    if spec["test"] and "test" not in st:
        st["test"] = run_test_evaluation(cfg, ctx.paths, st["run_id"])
        ctx.save()
    if spec["save"] and not st.get("model_dir"):
        pkg = package_model(cfg, ctx.paths, st["run_id"], _example_image(ctx, cfg), Path(ctx.paths["dest"]) / "models" / spec["save"])
        if not pkg["load_verified"]:
            raise RuntimeError(f"the saved model {spec['save']} failed its reload check: {pkg}")
        st["model_dir"] = pkg["model_dir"]
        ctx.save()
        ctx.log(f"saved model: {pkg['model_dir']}")
    return st


def step_detection(ctx: PipelineContext) -> dict:
    ctx.log("step 2: detection training -> test -> save")
    return run_step(ctx, "detection")


def step_second_round(ctx: PipelineContext) -> dict:
    ctx.log("step 3: optional second round (COCO, then fine-tune on VOC2012) -> save final model")
    run_step(ctx, "coco")
    return run_step(ctx, "second_round")


def step_longer_schedule(ctx: PipelineContext) -> dict:
    ctx.log("step 4: longer schedules (zoom-out augmentation, 2x iterations) from the saved model, with the respective datasets")
    run_step(ctx, "coco_long")
    return run_step(ctx, "long_final")


def pipeline_summary(ctx: PipelineContext) -> dict:
    rows = []
    for spec in STEPS:
        st = ctx.state["stages"].get(spec["key"], {})
        sm = st.get("summary", {})
        t = st.get("test", {})
        rows.append({"step": spec["key"], "stage": spec["stage"], "iterations": sm.get("iterations"),
                     "stop_reason": sm.get("stop_reason"), "best_val_map50": sm.get("best_val_metric"),
                     "train_minus_val": sm.get("generalisation_gap"), "overfit_warning": sm.get("overfit_warning"),
                     "test_map50": t.get("test_map50"), "saved_model": st.get("model_dir")})
    out = {"device_mode": ctx.mode, "dest": str(ctx.paths["dest"]), "steps": rows}
    (Path(ctx.paths["dest"]) / "pipeline_summary.json").write_text(json.dumps(out, indent=2, default=str))
    ctx.log("summary (choose the final model by VALIDATION mAP; the test set is for reporting):")
    for r in rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return out


def run_pipeline(path, device_mode=None, overrides=None, stage_overrides=None) -> dict:
    """Everything, in order. `overrides` / `stage_overrides` exist for tests; the notebook and script never pass them."""
    ctx = start_pipeline(path, device_mode, overrides, stage_overrides)
    step_load_datasets(ctx)
    step_estimate_time(ctx)
    step_detection(ctx)
    step_second_round(ctx)
    step_longer_schedule(ctx)
    return pipeline_summary(ctx)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python {Path(sys.argv[0]).name} PATH [cpu|gpu|combination]\n  PATH: folder for everything (data, caches, runs, models)\n"
              "  device: where to train; asked interactively when omitted")
        return 0
    run_pipeline(argv[0], argv[1] if len(argv) > 1 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
