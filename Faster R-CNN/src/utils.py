"""Config loading, seeding, device selection, environment info."""
from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path, overrides=None) -> dict:
    """Load YAML and apply dotted overrides such as ``training.lr=0.01`` or ``model.min_size=[256,320]``."""
    cfg = load_yaml(path)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=value, got: {item}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return resolve_data(cfg)


def resolve_data(cfg: dict) -> dict:
    """data.num_classes (foreground classes) follows the dataset: voc2007 -> 20; synthetic -> synthetic_classes."""
    d = cfg["data"]
    if d["dataset"] == "voc2007":
        d["num_classes"] = 20
    elif d["dataset"] == "synthetic":
        d["num_classes"] = d.get("synthetic_classes", 3)
    else:
        raise ValueError(f"unknown dataset {d['dataset']!r}; choose voc2007 or synthetic")
    return cfg


def registered_name(cfg: dict, env: dict) -> str:
    """Explicit env / REGISTERED_MODEL_NAME wins, otherwise <arch>-<dataset> (e.g. fasterrcnn_..._fpn-voc2007)."""
    return env.get("registered_model_name") or f"{cfg['model']['arch']}-{cfg['data']['dataset']}".replace("_", "-")


def load_env(path=None) -> dict:
    path = Path(path) if path else ROOT / "config" / "env.yaml"
    env = load_yaml(path) if path.exists() else {}
    env["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", env.get("tracking_uri", "sqlite:///mlflow.db"))
    env["registered_model_name"] = os.environ.get("REGISTERED_MODEL_NAME", env.get("registered_model_name"))
    return env


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        dev = torch.device(pref)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda requested but PyTorch sees no GPU (CPU-only build, or AMD "
                               "ROCm / NVIDIA driver missing). Use training.device=cpu, or fix the install.")
        return dev
    if torch.cuda.is_available():  # ROCm builds of PyTorch also report cuda
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_info(device: torch.device) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
