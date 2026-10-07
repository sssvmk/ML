"""Real-time inference service (FastAPI).

    AECLF_MODEL_DIR=runs/<run>/champion uvicorn aeclf.serving.app:app --host 0.0.0.0 --port 8000

POST /predict      {"records": [ {raw passenger row}, ... ]}  -> satisfied flag + probability
POST /embed        same input -> latent codes from the pretrained encoder   (not available for the no-pretraining control)
POST /reconstruct  same input -> decoded rows in original units + per-row reconstruction error (anomaly score)
GET  /health, /model-info
Input is validated against the schema stored in the bundle (HTTP 422 on missing columns; unseen categories/ranges are warnings).
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import pandas as pd

from ..bundle import ModelBundle
from ..validation import InputValidationError, validate_input

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    FastAPI = None

STATE: dict = {}


def load_model(model_dir: str | None = None) -> ModelBundle:
    model_dir = model_dir or os.environ.get("AECLF_MODEL_DIR", "champion")
    STATE["bundle"] = ModelBundle.load(model_dir)
    STATE["meta"] = ModelBundle.read_meta(model_dir)
    return STATE["bundle"]


if FastAPI is not None:

    class Records(BaseModel):
        records: list[dict] = Field(..., min_length=1, max_length=10000, description="raw passenger rows")
        include_probability: bool = True

    @asynccontextmanager
    async def lifespan(app):
        if "bundle" not in STATE:
            load_model()
        yield

    app = FastAPI(title="Airline satisfaction (autoencoder encoder + classifier)", version="0.1.0", lifespan=lifespan)

    def _frame(req: Records) -> tuple[pd.DataFrame, list[str]]:
        df = pd.DataFrame(req.records)
        try:
            _, warns = validate_input(df, STATE["bundle"].schema)
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return df, warns

    @app.get("/health")
    def health():
        return {"status": "ok" if "bundle" in STATE else "model_not_loaded"}

    @app.get("/model-info")
    def model_info():
        b = STATE["bundle"]
        return {"candidate": b.metadata.get("candidate"), "kind": b.kind, "mode": b.mode, "threshold": b.threshold, "head_type": b.head_type, "latent_dim": (b.ae_arch["latent"] if b.ae_arch else None),
                "has_pretrained_autoencoder": b.ae_state is not None, "required_columns": b.schema.required_columns, "metadata": b.metadata}

    @app.post("/predict")
    def predict(req: Records):
        t0 = time.perf_counter()
        df, warns = _frame(req)
        b = STATE["bundle"]
        proba = b.predict_proba(df)
        rows = [{"satisfied": bool(p >= b.threshold), **({"probability": round(float(p), 6)} if req.include_probability else {})} for p in proba]
        return {"predictions": rows, "threshold": b.threshold, "warnings": warns, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/embed")
    def embed(req: Records):
        df, warns = _frame(req)
        try:
            z = STATE["bundle"].embed(df)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"embeddings": z.round(6).tolist(), "warnings": warns}

    @app.post("/reconstruct")
    def reconstruct(req: Records):
        df, warns = _frame(req)
        try:
            dec, err = STATE["bundle"].reconstruct(df)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"reconstruction": dec.where(dec.notna(), None).to_dict("records"), "reconstruction_error": [round(float(e), 6) for e in err], "warnings": warns}


def main():  # pragma: no cover
    import uvicorn

    uvicorn.run("aeclf.serving.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":  # pragma: no cover
    main()
