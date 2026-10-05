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
