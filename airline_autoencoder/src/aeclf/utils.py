"""Small shared helpers: seeding, hashing, git info, JSON encoding, timing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if np.isnan(v) or np.isinf(v) else v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return super().default(o)


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def environment_info() -> dict:
    import sklearn

    info = {"python": platform.python_version(), "platform": platform.platform(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("xgboost", "lightgbm", "torch", "optuna", "mlflow"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


@contextmanager
def timer():
    t = {"start": time.perf_counter(), "seconds": None}
    yield t
    t["seconds"] = time.perf_counter() - t["start"]


def safe_name(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
