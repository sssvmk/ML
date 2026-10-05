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
