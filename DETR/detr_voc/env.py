"""Cache control: every cache, temporary and output path is placed under the destination folder."""
import os
import sys
import tempfile
from pathlib import Path


def make_paths(dest) -> dict:
    d = Path(str(dest)[len("dbfs:"):] if str(dest).startswith("dbfs:/") else dest)
    return {"dest": d, "data": d / "data", "runs": d / "runs", "checkpoints": d / "checkpoints", "tmp": d / "tmp",
            "mlflow": d / "mlflow", "mlflow_artifacts": d / "mlflow" / "artifacts",
            "cache": d / "cache", "torch_home": d / "cache" / "torch", "mpl": d / "cache" / "matplotlib",
            "cuda_cache": d / "cache" / "cuda", "xdg": d / "cache" / "xdg", "pycache": d / "cache" / "pycache"}


def setup_environment(dest) -> dict:
    """Create the folder layout and point every library cache at it. Call once, before heavy imports if possible."""
    p = make_paths(dest)
    if str(p["dest"]).startswith("/Volumes/"):
        parts = p["dest"].parts
        if len(parts) < 5:
            raise ValueError("a Volume path must look like /Volumes/<catalog>/<schema>/<volume>/<folder>")
        if not Path(*parts[:5]).exists():
            raise FileNotFoundError(f"{Path(*parts[:5])} does not exist: create the volume first (CREATE VOLUME ...)")
    for k, path in p.items():
        if k != "mlflow_artifacts":
            path.mkdir(parents=True, exist_ok=True)
    p["mlflow_artifacts"].mkdir(parents=True, exist_ok=True)
    env = {"TORCH_HOME": p["torch_home"], "XDG_CACHE_HOME": p["xdg"], "MPLCONFIGDIR": p["mpl"],
           "CUDA_CACHE_PATH": p["cuda_cache"], "TMPDIR": p["tmp"], "TEMP": p["tmp"], "TMP": p["tmp"],
           "TRITON_CACHE_DIR": p["cache"] / "triton", "HF_HOME": p["cache"] / "huggingface"}
    for k, v in env.items():
        os.environ[k] = str(v)
    tempfile.tempdir = str(p["tmp"])
    sys.pycache_prefix = str(p["pycache"])
    try:
        import torch
        torch.hub.set_dir(str(p["torch_home"]))
    except ImportError:
        pass
    return p
