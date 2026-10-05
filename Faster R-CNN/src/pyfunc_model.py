"""MLflow pyfunc wrapper: base64 image(s) in, JSON detections out. Weights come from the checkpoint (no downloads)."""
from __future__ import annotations

import base64
import io
import json

import mlflow.pyfunc
import numpy as np
import pandas as pd


class DetectorModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch

        from src.model import build_model

        state = torch.load(context.artifacts["checkpoint"], map_location="cpu", weights_only=True)
        self.model = build_model(state["model_cfg"], state["num_foreground"], pretrained="none")
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.classes = state["classes"]
        self.default_floor = float(state["model_cfg"].get("score_floor", 0.05))

    def predict(self, context, model_input, params=None):
        import torch
        from PIL import Image

        params = params or {}
        self.model.roi_heads.score_thresh = float(params.get("score_floor", self.default_floor))
        out_rows = []
        with torch.inference_mode():
            for b64 in model_input["image_b64"].tolist():
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255.0)
                out = self.model([x])[0]
                dets = [{"box": [round(float(v), 2) for v in b], "score": round(float(s), 5), "label_id": int(lb),
                         "label": self.classes[int(lb) - 1]}
                        for b, s, lb in zip(out["boxes"], out["scores"], out["labels"], strict=True)]
                out_rows.append(json.dumps({"width": img.width, "height": img.height, "detections": dets}))
        return pd.DataFrame({"detections": out_rows})
