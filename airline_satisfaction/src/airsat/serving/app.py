"""Step 10: real-time inference service (FastAPI).

    AIRSAT_MODEL_DIR=runs/<run>/champion uvicorn airsat.serving.app:app --host 0.0.0.0 --port 8000

POST /predict   {"records": [ {<raw passenger row, same columns as the training file>}, ... ]}
GET  /health    liveness + model loaded
GET  /model-info algorithm, threshold, training metrics, schema, lineage
Input is validated against the schema stored in the model bundle; unseen categories / out-of-range values are
reported as warnings, missing required columns are rejected with HTTP 422.
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
    model_dir = model_dir or os.environ.get("AIRSAT_MODEL_DIR", "champion")
    STATE["bundle"] = ModelBundle.load(model_dir)
    STATE["meta"] = ModelBundle.read_meta(model_dir)
    STATE["loaded_at"] = time.time()
    return STATE["bundle"]


if FastAPI is not None:

    class PredictRequest(BaseModel):
        records: list[dict] = Field(..., min_length=1, max_length=10000, description="raw passenger rows")
        include_probability: bool = True

    @asynccontextmanager
    async def lifespan(app):
        if "bundle" not in STATE:
            load_model()
        yield

    app = FastAPI(title="Airline passenger satisfaction", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok" if "bundle" in STATE else "model_not_loaded"}

    @app.get("/model-info")
    def model_info():
        meta = STATE["meta"]
        return {"algorithm": meta["algorithm"], "threshold": meta["threshold"], "metadata": meta["metadata"],
                "required_columns": STATE["bundle"].schema.required_columns,
                "ordinal_ranges": STATE["bundle"].schema.ordinal_ranges, "categories": STATE["bundle"].schema.categories}

    @app.post("/predict")
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        df = pd.DataFrame(req.records)
        try:
            _, warnings_ = validate_input(df, STATE["bundle"].schema)
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        proba = STATE["bundle"].predict_proba(df)
        thr = STATE["bundle"].threshold
        rows = [{"satisfied": bool(p >= thr), **({"probability": round(float(p), 6)} if req.include_probability else {})} for p in proba]
        return {"predictions": rows, "model": STATE["meta"]["algorithm"], "threshold": thr, "warnings": warnings_,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def main():  # pragma: no cover
    import uvicorn

    uvicorn.run("airsat.serving.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":  # pragma: no cover
    main()
