"""MLflow packaging + registry (aliases). Best-effort: a registry problem never blocks training."""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from .bundle import ModelBundle

try:
    import mlflow
    import mlflow.pyfunc

    _Base = mlflow.pyfunc.PythonModel
except Exception:  # pragma: no cover
    mlflow = None
    _Base = object


class AeclfPyfunc(_Base):
    """raw passenger rows in -> probability + label out (requires `aeclf` installed where it is served)."""

    def load_context(self, context):
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        p = self.bundle.predict_proba(pd.DataFrame(model_input))
        return pd.DataFrame({"probability": p, "prediction": (p >= self.bundle.threshold).astype(int)})


def register_candidate(run_dir: Path, bundle_dir: Path, cfg: dict, example_rows: pd.DataFrame, tracking_uri: str | None) -> dict:
    if mlflow is None or not cfg["tracking"].get("enabled", True):
        return {"registered": False, "reason": "mlflow unavailable or tracking disabled"}
    try:
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{(Path(run_dir) / 'mlflow.db').resolve()}")
        mlflow.set_experiment(cfg["tracking"]["experiment_name"])
        b = ModelBundle.load(bundle_dir)
        sample = example_rows.head(20).copy()
        sig = infer_signature(sample, pd.DataFrame({"probability": b.predict_proba(sample), "prediction": b.predict(sample)}))
        key = "name" if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters else "artifact_path"
        with mlflow.start_run(run_name="champion_packaging") as run:
            mlflow.set_tags({"phase": "champion", "candidate": str(b.metadata.get("candidate"))})
            info = mlflow.pyfunc.log_model(**{key: "model"}, python_model=AeclfPyfunc(), artifacts={"bundle": str(bundle_dir)},
                                           signature=sig, input_example=sample.head(3), pip_requirements=["aeclf"])
            uri = getattr(info, "model_uri", f"runs:/{run.info.run_id}/model")
        name = cfg["tracking"]["registered_model_name"]
        mv = mlflow.register_model(uri, name)
        c = MlflowClient()
        c.set_registered_model_alias(name, "candidate", mv.version)
        c.set_model_version_tag(name, mv.version, "validated", "false")
        return {"registered": True, "name": name, "version": mv.version, "alias": "candidate", "model_uri": uri}
    except Exception as exc:
        return {"registered": False, "reason": f"{type(exc).__name__}: {exc}"}
