"""
Version identifiers for reproducible tracking (PRD §3.3.10 / G-02):
dataset version id, code version id, search-space version.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent


def dataset_version_id(df: pd.DataFrame) -> str:
    """Stable id of the exact rows a run saw: content hash of (segment, dataset, date, value) plus the
    nested lineage (source_adapter, rule_version, extraction_batch_id) the Data Module attached."""
    cols = [c for c in ("segment_id", "dataset", "series_role", "date", "value") if c in df.columns]
    body = pd.util.hash_pandas_object(df[cols].sort_values(cols[:4]).reset_index(drop=True), index=False).values.tobytes()
    from contract import lineage_field
    has = "lineage" in df
    lineage = json.dumps({
        "adapters": sorted(map(str, lineage_field(df, "source_adapter").unique())) if has else [],
        "rule_version": sorted(map(str, lineage_field(df, "rule_version").unique())) if has else [],
        "batches": sorted(map(str, lineage_field(df, "extraction_batch_id").unique())) if has else [],
    })
    return hashlib.sha256(body + lineage.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def code_version_id() -> str:
    """git commit of the code if the package sits in a git checkout, otherwise a content hash of every
    .py file in the package (so a code change always changes the id)."""
    try:
        out = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return f"git:{out.stdout.strip()}"
    except Exception:
        pass
    h = hashlib.sha256()
    for p in sorted(_ROOT.rglob("*.py")):
        if "vendor" in p.parts or "logs" in p.parts or "registry_store" in p.parts:
            continue
        h.update(str(p.relative_to(_ROOT)).encode())
        h.update(p.read_bytes())
    return f"src:{h.hexdigest()[:12]}"


def search_space_version(space: dict) -> str:
    return hashlib.sha256(json.dumps(space, sort_keys=True, default=str).encode()).hexdigest()[:12]
