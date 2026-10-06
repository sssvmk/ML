"""MLflow model packaging + registry (aliases, not deprecated stages). Best-effort: never blocks training."""
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


class AirsatPyfunc(_Base):
    """pyfunc wrapper: raw passenger rows in -> probability + label out. Requires `airsat` installed where it is served."""

    def load_context(self, context):
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        p = self.bundle.predict_proba(pd.DataFrame(model_input))
        return pd.DataFrame({"probability": p, "prediction": (p >= self.bundle.threshold).astype(int)})


def register_candidate(run_dir: Path, bundle_dir: Path, cfg: dict, example_rows: pd.DataFrame, metadata: dict, tracking_uri: str | None) -> dict:
    """Log the champion bundle as an MLflow pyfunc model, register it and tag it with the `candidate` alias."""
    if mlflow is None or not cfg["tracking"].get("enabled", True):
        return {"registered": False, "reason": "mlflow unavailable or tracking disabled"}
    try:
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient

        uri = tracking_uri or f"sqlite:///{(Path(run_dir) / 'mlflow.db').resolve()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg["tracking"]["experiment_name"])
        bundle = ModelBundle.load(bundle_dir)
        sample = example_rows.head(20).copy()
        sig_out = pd.DataFrame({"probability": bundle.predict_proba(sample), "prediction": bundle.predict(sample)})
        signature = infer_signature(sample, sig_out)
        kwargs = dict(python_model=AirsatPyfunc(), artifacts={"bundle": str(bundle_dir)}, signature=signature,
                      input_example=sample.head(3), pip_requirements=["airsat"])
        key = "name" if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters else "artifact_path"
        with mlflow.start_run(run_name="champion_packaging") as run:
            mlflow.set_tags({"phase": "champion", "algorithm": bundle.algorithm})
            mlflow.log_params({k: str(v)[:200] for k, v in metadata.items() if not isinstance(v, (dict, list))})
            info = mlflow.pyfunc.log_model(**{key: "model"}, **kwargs)
            model_uri = getattr(info, "model_uri", f"runs:/{run.info.run_id}/model")
        name = cfg["tracking"]["registered_model_name"]
        mv = mlflow.register_model(model_uri, name)
        client = MlflowClient()
        client.set_registered_model_alias(name, "candidate", mv.version)
        client.set_model_version_tag(name, mv.version, "validated", "false")
        client.set_model_version_tag(name, mv.version, "algorithm", bundle.algorithm)
        return {"registered": True, "name": name, "version": mv.version, "alias": "candidate", "model_uri": model_uri}
    except Exception as exc:
        return {"registered": False, "reason": f"{type(exc).__name__}: {exc}"}
