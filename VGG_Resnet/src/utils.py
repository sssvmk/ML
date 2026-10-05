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
    """Load YAML and apply dotted overrides such as ``training.lr=0.1``."""
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
    """num_classes always follows the dataset (cifar10 -> 10, cifar100 -> 100); synthetic keeps the configured value."""
    from src.data import DATASETS
    name = cfg["data"]["dataset"]
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    if DATASETS[name]["num_classes"] is not None:
        cfg["data"]["num_classes"] = DATASETS[name]["num_classes"]
    return cfg


def registered_name(cfg: dict, env: dict) -> str:
    """Registry name: explicit env/REGISTERED_MODEL_NAME wins, otherwise <arch>-<dataset> (e.g. resnet20-cifar10)."""
    return env.get("registered_model_name") or f"{cfg['model']['arch']}-{cfg['data']['dataset']}"


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
            raise RuntimeError(
                "training.device=cuda was requested but PyTorch sees no GPU. This is a CPU-only build or the AMD "
                "driver/ROCm runtime is missing. Run `python run.py gpu-check` for a diagnosis "
                "(AMD GPUs on Windows need the ROCm build of PyTorch, see README).")
        return dev
    if torch.cuda.is_available():  # also true for ROCm builds of PyTorch
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["hip"] = str(getattr(torch.version, "hip", None))
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        if arch:
            info["gpu_arch"] = arch
    return info
