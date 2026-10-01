"""
MLflow integration (PRD v10 §3.3.10, §4.4).

  * `run()`               -- the per-segment/process/algorithm run (full_training,
                             daily_inference), fluent API, as before.
  * explicit-client API   -- thread-safe run/param/metric/artifact calls by run id,
                             used by search.py to open ONE NESTED RUN PER
                             HYPERPARAMETER TRIAL under a parent search run (G-02).
  * registry API          -- one registered model per segment, versions moved
                             through Staging -> Production -> Archived (G-01, §4.4).

Everything degrades to a silent no-op when mlflow isn't installed or
tracking is disabled in config.json, so the rest of the pipeline never
depends on it (mirrors §3.7's "not a mandatory dependency" stance for Ray);
the JSON log folder stays the human-readable supplement either way.

NOTE on stages: MLflow has deprecated Model Registry *stages* in favour of
aliases, but PRD §4.4 specifies Staging/Production/Archived and MLflow 3.16
still implements them; if a later MLflow removes stage transitions the
`transition_stage` call below is the one place to swap for aliases.
"""

from __future__ import annotations
from contextlib import contextmanager
import warnings


def _clip(v, n=480) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


class MLflowLogger:
    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("mlflow", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.tracking_uri = cfg.get("tracking_uri")
        self.registry_uri = cfg.get("registry_uri")
        self.experiment_name = cfg.get("experiment_name", "stc_cashflow")
        self._mlflow = None
        self._client = None
        self._experiment_id = None
        self._disabled_reason = None
        if self.enabled:
            try:
                import mlflow
                from mlflow import MlflowClient
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                if self.registry_uri:
                    mlflow.set_registry_uri(self.registry_uri)
                exp = mlflow.set_experiment(self.experiment_name)
                self._experiment_id = exp.experiment_id
                self._client = MlflowClient()
                self._mlflow = mlflow
            except Exception as exc:
                # mlflow not installed, or tracking server unreachable -- degrade to no-op.
                self._mlflow = None
                self._client = None
                self._disabled_reason = str(exc)

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    # ---- fluent per-run context (full_training / daily_inference) --------------------------
    @contextmanager
    def run(self, *, segment_id: str, process: str, algorithm: str, rule_version: str,
            hyperparameters: dict | None = None, metrics: dict | None = None, extra_tags: dict | None = None):
        if not self.active:
            yield None
            return
        mlflow = self._mlflow
        with mlflow.start_run(run_name=f"{segment_id}/{process}/{algorithm}") as run:
            mlflow.set_tags({
                "segment_id": segment_id, "process": process, "algorithm": algorithm,
                "rule_version": rule_version, **{k: _clip(v) for k, v in (extra_tags or {}).items()},
            })
            if hyperparameters:
                mlflow.log_params({k: _clip(v) for k, v in hyperparameters.items()})  # mlflow rejects non-scalars
            if metrics:
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v == v})
            yield run

    def log_artifact_in_run(self, local_path, artifact_path: str | None = None) -> None:
        """Attach a file (chart, JSON log) to the CURRENT fluent run."""
        if self.active:
            try:
                self._mlflow.log_artifact(str(local_path), artifact_path)
            except Exception:
                pass

    # ---- explicit, thread-safe API (nested per-trial runs) ---------------------------------
    def start_run_explicit(self, name: str, tags: dict | None = None, parent_run_id: str | None = None) -> str | None:
        if not self.active:
            return None
        t = {k: _clip(v) for k, v in (tags or {}).items()}
        t["mlflow.runName"] = name
        if parent_run_id:
            t["mlflow.parentRunId"] = parent_run_id
        return self._client.create_run(self._experiment_id, tags=t).info.run_id

    def log_params_explicit(self, run_id: str | None, params: dict) -> None:
        if run_id and self.active:
            for k, v in params.items():
                try:
                    self._client.log_param(run_id, k, _clip(v))
                except Exception:
                    pass

    def log_metrics_explicit(self, run_id: str | None, metrics: dict, step: int | None = None) -> None:
        if run_id and self.active:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf")):
                    try:
                        self._client.log_metric(run_id, k, float(v), step=step or 0)
                    except Exception:
                        pass

    def set_tags_explicit(self, run_id: str | None, tags: dict) -> None:
        if run_id and self.active:
            for k, v in tags.items():
                self._client.set_tag(run_id, k, _clip(v))

    def log_artifact_explicit(self, run_id: str | None, local_path, artifact_path: str | None = None) -> None:
        if run_id and self.active:
            try:
                self._client.log_artifact(run_id, str(local_path), artifact_path)
            except Exception:
                pass

    def log_dict_explicit(self, run_id: str | None, d: dict, artifact_file: str) -> None:
        if run_id and self.active:
            try:
                self._client.log_dict(run_id, d, artifact_file)
            except Exception:
                pass

    def end_run_explicit(self, run_id: str | None, status: str = "FINISHED") -> None:
        if run_id and self.active:
            self._client.set_terminated(run_id, status)

    # ---- model registry (one registered model per segment) ---------------------------------
    @staticmethod
    def model_name(segment_id: str) -> str:
        return f"stc-{segment_id}"

    def register_version(self, segment_id: str, run_id: str | None, artifact_dir, tags: dict | None = None) -> int | None:
        """Log the model artifact into the run and create a registered-model version (stage None)."""
        if not self.active or not run_id:
            return None
        name = self.model_name(segment_id)
        try:
            try:
                self._client.create_registered_model(name)
            except Exception:
                pass  # already exists
            self._client.log_artifacts(run_id, str(artifact_dir), "model")
            mv = self._client.create_model_version(
                name, source=f"runs:/{run_id}/model", run_id=run_id,
                tags={k: _clip(v) for k, v in (tags or {}).items()},
            )
            return int(mv.version)
        except Exception as exc:
            self._disabled_reason = f"register_version failed: {exc}"
            return None

    def transition_stage(self, segment_id: str, mlflow_version: int | None, stage: str) -> None:
        if not self.active or mlflow_version is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # MLflow's stage deprecation warning (see module docstring)
            self._client.transition_model_version_stage(self.model_name(segment_id), str(mlflow_version), stage)

    def stage_of(self, segment_id: str, mlflow_version: int) -> str | None:
        if not self.active:
            return None
        return self._client.get_model_version(self.model_name(segment_id), str(mlflow_version)).current_stage
