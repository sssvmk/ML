"""Step 11 (MLOps): experiment tracking. MLflow when available, always mirrored to a plain JSONL log so the
analysis never depends on a tracking server being reachable."""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

from .utils import NumpyEncoder


def _clean_key(k: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-\. :/]", "_", str(k))[:240]


class Tracker:
    def __init__(self, cfg: dict, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.log_path = self.run_dir / "experiment_log.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow = None
        self.stack: list[str] = []
        tcfg = cfg["tracking"]
        if tcfg.get("enabled", True):
            try:
                import mlflow

                uri = tcfg.get("uri") or f"sqlite:///{(self.run_dir / 'mlflow.db').resolve()}"
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(tcfg["experiment_name"])
                self.mlflow = mlflow
                self.uri = uri
            except Exception as exc:  # tracking must never break training
                self._event("tracking_disabled", {"reason": repr(exc)})

    def _event(self, kind: str, data: dict):
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "run": "/".join(self.stack), "event": kind, "data": data},
                                cls=NumpyEncoder) + "\n")

    @contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None):
        self.stack.append(name)
        self._event("run_start", {"name": name, "tags": tags or {}})
        ctx = None
        if self.mlflow:
            try:
                ctx = self.mlflow.start_run(run_name=name, nested=nested, tags={k: str(v) for k, v in (tags or {}).items()})
            except Exception as exc:
                self._event("mlflow_error", {"op": "start_run", "error": repr(exc)})
        try:
            yield self
        finally:
            if ctx is not None:
                try:
                    self.mlflow.end_run()
                except Exception:
                    pass
            self._event("run_end", {"name": name})
            self.stack.pop()

    @property
    def active_run_id(self) -> str | None:
        if self.mlflow and self.mlflow.active_run():
            return self.mlflow.active_run().info.run_id
        return None

    def log_params(self, params: dict):
        self._event("params", params)
        if self.mlflow and self.mlflow.active_run():
            try:
                self.mlflow.log_params({_clean_key(k): str(v)[:500] for k, v in params.items()})
            except Exception as exc:
                self._event("mlflow_error", {"op": "log_params", "error": repr(exc)})

    def log_metrics(self, metrics: dict, step: int | None = None):
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        self._event("metrics", {"step": step, **clean})
        if self.mlflow and self.mlflow.active_run():
            try:
                self.mlflow.log_metrics({_clean_key(k): v for k, v in clean.items()}, step=step)
            except Exception as exc:
                self._event("mlflow_error", {"op": "log_metrics", "error": repr(exc)})

    def set_tags(self, tags: dict):
        if self.mlflow and self.mlflow.active_run():
            try:
                self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
            except Exception:
                pass

    def log_artifact(self, path: str | Path, artifact_path: str | None = None):
        self._event("artifact", {"path": str(path)})
        if self.mlflow and self.mlflow.active_run() and Path(path).exists():
            try:
                self.mlflow.log_artifact(str(path), artifact_path)
            except Exception as exc:
                self._event("mlflow_error", {"op": "log_artifact", "error": repr(exc)})
