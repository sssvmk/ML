"""
Pipeline (bridge): config-driven adapter instantiation, extraction, validation and handoff to the
Orchestrator. This is the ONLY place that knows which adapter to use; downstream code (Orchestrator,
algorithms) never sees adapter internals. Adapter changes never propagate downstream.

Config.json -> data_source block drives everything:
    {
        "data_source": {
            "adapter": "synthetic" | "csv_fixture" | "generic_multivariate" | "sap" | "<custom>",
            "params": { ...adapter-specific kwargs passed to extract() },
            "segments": ["seg-a", "seg-b", ...]   // optional filter after extraction
        }
    }

The Orchestrator decides single vs pooled based on which algorithms are enabled (pooling_capable flag
on each class). The bridge does NOT make that decision — it only supplies validated frames.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pandas as pd

from contract import validate_contract


# ------------------------------------------------------------------------------- adapter registry
_ADAPTER_REGISTRY: dict[str, str] = {
    "synthetic":            "adapters.synthetic.SyntheticAdapter",
    "csv_fixture":          "adapters.csv_fixture.CSVFixtureAdapter",
    "generic_multivariate": "adapters.generic_multivariate.GenericMultivariateAdapter",
    "exchange_rate":        "adapters.exchange_rate.ExchangeRateAdapter",
    "sap":                  "adapters.sap_stub.SAPAdapter",
}


def register_adapter(name: str, dotted_class_path: str) -> None:
    """Register a custom adapter class so the bridge can instantiate it by name from config.
    Call this before Pipeline.run() in any entry-point that uses a custom adapter.
    Example: register_adapter("my_source", "my_package.adapters.MyAdapter")
    """
    _ADAPTER_REGISTRY[name] = dotted_class_path


def _load_adapter_class(name: str):
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"unknown adapter {name!r}; registered adapters: {sorted(_ADAPTER_REGISTRY)}. "
            f"Call pipeline.register_adapter(name, dotted_class_path) to add a custom one."
        )
    module_path, class_name = _ADAPTER_REGISTRY[name].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


# ------------------------------------------------------------------------------- result
class PipelineResult:
    """Everything the caller needs after a pipeline run."""
    def __init__(self, segments: dict[str, pd.DataFrame], validation: dict, elapsed_s: float,
                 adapter_name: str, entries: dict | None = None):
        self.segments = segments          # {segment_id: contract DataFrame}
        self.validation = validation      # {segment_id: ValidationResult}
        self.elapsed_s = elapsed_s
        self.adapter_name = adapter_name
        self.entries = entries or {}      # {segment_id: RegistryEntry} — filled after orchestrator runs

    @property
    def valid_segments(self) -> dict[str, pd.DataFrame]:
        return {sid: df for sid, df in self.segments.items() if self.validation[sid].ok}

    @property
    def invalid_segments(self) -> dict[str, list[str]]:
        return {sid: self.validation[sid].errors for sid, r in self.validation.items() if not r.ok}

    def summary(self) -> dict:
        return {"adapter": self.adapter_name, "total_segments": len(self.segments),
                "valid": len(self.valid_segments), "invalid": len(self.invalid_segments),
                "elapsed_s": round(self.elapsed_s, 2)}


# ------------------------------------------------------------------------------- main class
class Pipeline:
    """
    Config-driven bridge between any source adapter and the Orchestrator.

    The Orchestrator (not the bridge) decides whether to run single-segment or pooled training
    based on which algorithms are enabled and their pooling_capable flags. The bridge hands
    it validated frames and steps aside.
    """

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.config = config
        ds_cfg = config.get("data_source", {})
        self.adapter_name: str = ds_cfg.get("adapter", "synthetic")
        self.adapter_params: dict = dict(ds_cfg.get("params", {}))
        self.segment_filter: list | None = ds_cfg.get("segments") or None
        self.min_observations: int = int(config.get("orchestration", {}).get("min_observations", 30))
        self.output_dir = Path(output_dir or config.get("orchestration", {}).get("output_dir", "logs"))
        self._adapter = None

    # ---- adapter ----------------------------------------------------------
    def _get_adapter(self):
        if self._adapter is None:
            cls = _load_adapter_class(self.adapter_name)
            self._adapter = cls()
        return self._adapter

    def extract(self) -> dict[str, pd.DataFrame]:
        """Run the configured adapter and split its output into per-segment DataFrames."""
        adapter = self._get_adapter()
        df = adapter.extract(**self.adapter_params)
        segments = {}
        for sid, group in df.groupby("segment_id"):
            segments[str(sid)] = group.reset_index(drop=True)
        if self.segment_filter:
            segments = {k: v for k, v in segments.items() if k in self.segment_filter}
        return segments

    # ---- validate ---------------------------------------------------------
    def validate(self, segments: dict[str, pd.DataFrame]) -> dict[str, object]:
        return {sid: validate_contract(df, min_observations=self.min_observations)
                for sid, df in segments.items()}

    # ---- orchestrate ------------------------------------------------------
    def _needs_pooled(self, candidates: dict) -> bool:
        """True if any enabled candidate is pooling-capable — the Orchestrator knows, not the bridge."""
        return any(getattr(type(f()), "pooling_capable", False) for f in candidates.values())

    def run(self, orchestrator, candidates: dict, horizon: int,
            rule_version: str = "pipeline-v1") -> PipelineResult:
        """
        Full pipeline run: extract → validate → orchestrate → return results.
        Single-segment and pooled paths are chosen by the Orchestrator based on which
        algorithms are enabled (pooling_capable flag). The bridge routes accordingly.
        """
        t0 = time.time()
        segments = self.extract()
        validation = self.validate(segments)
        valid = {sid: df for sid, df in segments.items() if validation[sid].ok}

        invalid = {sid: validation[sid].errors for sid in segments if not validation[sid].ok}
        if invalid:
            print(f"[pipeline] {len(invalid)} segments failed validation and are excluded: {invalid}")

        if not valid:
            raise ValueError("No segments passed contract validation — cannot proceed to training.")

        entries = {}
        needs_pooled = self._needs_pooled(candidates)

        if needs_pooled and len(valid) >= 2:
            # Orchestrator trains pooling-capable algorithms across all valid segments at once,
            # and single-segment algorithms per segment — the orchestrator manages both paths.
            print(f"[pipeline] {len(valid)} valid segments → full_train_pooled "
                  f"(pooling-capable candidates detected)")
            entries = orchestrator.full_train_pooled(valid, horizon=horizon, rule_version=rule_version)
        elif needs_pooled and len(valid) < 2:
            print(f"[pipeline] only {len(valid)} valid segment(s) — pooled algorithms need >= 2; "
                  f"running single-segment path for all")
            for sid, df in valid.items():
                entries[sid] = orchestrator.full_train(sid, df, horizon=horizon, rule_version=rule_version)
        else:
            print(f"[pipeline] {len(valid)} valid segments → full_train (single-segment path)")
            for sid, df in valid.items():
                entries[sid] = orchestrator.full_train(sid, df, horizon=horizon, rule_version=rule_version)

        result = PipelineResult(segments, validation, time.time() - t0, self.adapter_name, entries)
        self._write_pipeline_log(result, horizon, rule_version)
        return result

    def infer(self, orchestrator, horizon: int) -> dict[str, pd.DataFrame]:
        """
        Inference pipeline: extract → validate → infer from the Production registry.
        Uses daily_infer_pool for joint models, daily_infer for everything else.
        """
        segments = self.extract()
        validation = self.validate(segments)
        valid = {sid: df for sid, df in segments.items() if validation[sid].ok}
        if not valid:
            raise ValueError("No segments passed contract validation for inference.")
        return orchestrator.daily_infer_pool(valid, horizon)

    # ---- output -----------------------------------------------------------
    def _write_pipeline_log(self, result: PipelineResult, horizon: int, rule_version: str):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log = {
            "adapter": result.adapter_name, "adapter_params": self.adapter_params,
            "total_segments": len(result.segments), "valid_segments": sorted(result.valid_segments),
            "invalid_segments": result.invalid_segments, "elapsed_s": round(result.elapsed_s, 2),
            "horizon": horizon, "rule_version": rule_version,
            "winners": {sid: {"algorithm": e.algorithm_name, "window": e.window,
                               "metrics": e.metrics, "holdout_metrics": e.holdout_metrics}
                        for sid, e in result.entries.items()},
        }
        import json
        (self.output_dir / "pipeline_run.json").write_text(json.dumps(log, indent=2, default=str))
