"""MLflow pyfunc wrapper: preprocessing + network in one deployable unit (uint8 NHWC in, probabilities out)."""
from __future__ import annotations

import mlflow.pyfunc
import numpy as np


class CifarClassifier(mlflow.pyfunc.PythonModel):
    BATCH = 256

    def load_context(self, context):
        import torch

        from src.model import build_model

        state = torch.load(context.artifacts["checkpoint"], map_location="cpu", weights_only=True)
        self.model = build_model(state["model_cfg"], state["num_classes"])
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        from src.data import MEAN, STD
        norm = state.get("norm") or {"mean": MEAN, "std": STD}  # checkpoints from before multi-dataset support
        self.mean, self.std = tuple(norm["mean"]), tuple(norm["std"])
        self.class_names = state["classes"]
        self.epoch = state["epoch"]

    def predict(self, context, model_input, params=None):
        import torch

        from src.data import preprocess_uint8_batch

        x = model_input.to_numpy() if hasattr(model_input, "to_numpy") else np.asarray(model_input)
        if x.ndim == 3:
            x = x[None]
        batches = []
        with torch.inference_mode():
            for i in range(0, len(x), self.BATCH):
                logits = self.model(preprocess_uint8_batch(x[i:i + self.BATCH], self.mean, self.std))
                batches.append(logits.softmax(1).numpy().astype(np.float32))
        return np.concatenate(batches) if batches else np.zeros((0, len(self.class_names)), np.float32)
