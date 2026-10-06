"""Configuration loading. All behaviour is driven by YAML (config/default.yaml) plus optional --set overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply 'a.b.c=value' style overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(value)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    s = cfg["split"]
    total = s["train"] + s["validation"] + s["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    if cfg["metric"]["primary"] not in {"roc_auc", "pr_auc", "f1", "accuracy", "neg_log_loss"}:
        raise ValueError(f"unsupported primary metric {cfg['metric']['primary']}")
    if cfg["output"]["label_style"] not in {"logical", "binary", "text"}:
        raise ValueError("output.label_style must be logical | binary | text")
