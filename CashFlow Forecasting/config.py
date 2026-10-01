"""
Loads config.json (the single source of truth for orchestration
parameters and per-algorithm hyperparameters) and builds the candidate
factory dict the Orchestrator needs -- no algorithm hyperparameters are
hard-coded in demo.py or orchestrator.py.
"""
from __future__ import annotations
from pathlib import Path
import json

from algorithms.catalog import load_class


def load_config(path: str | Path = "config.json") -> dict:
    return json.loads(Path(path).read_text())


def build_candidates(config: dict) -> dict:
    """id -> zero-arg factory returning a fresh AlgorithmModule instance,
    for every algorithm marked enabled in config.json."""
    candidates = {}
    for algo_id, spec in config["algorithms"].items():
        if not spec.get("enabled", True):
            continue
        cls = load_class(algo_id)
        forced = bool(spec.get("force_enabled", False))
        if getattr(cls, "requires_admission", False) and not forced:
            # custom model (G-42): only a model with a passing admission report for its CURRENT source competes
            from admission import is_admitted
            if not is_admitted(algo_id, cls, config)[0]:
                continue
        hp = dict(spec.get("hyperparameters", {}))
        hp.setdefault("frequency", str(config.get("orchestration", {}).get("frequency", "D")))   # this request: no hard-coded daily assumption
        if getattr(cls, "nf_model_name", None) or getattr(cls, "needs_horizon", False):
            # neural models are built for a fixed forecast length h: take it from the configured backtest horizon
            hp.setdefault("horizon", int(config.get("orchestration", {}).get("backtest_horizon", 4)))
        pk = getattr(cls, "prebuilt_key", None)
        if pk:
            hp["prebuilt"] = dict(config.get("prebuilt_models", {}).get(pk) or {})   # this module's pretrained-model entry
            hp["environment"] = (config.get("deployment") or {}).get("environment")   # for licence-restricted weights
        factory = (lambda cls=cls, hp=hp: cls(dict(hp)))
        factory.force_enabled = forced      # G-42 override: this algorithm is a config-forced winner, not a competitor (see orchestrator.py)
        candidates[algo_id] = factory
    return candidates
