"""Reading data and coercing the target. Works for training files (with target) and unseen files (without)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRUE_TOKENS = {"true", "t", "1", "1.0", "yes", "y", "satisfied"}
FALSE_TOKENS = {"false", "f", "0", "0.0", "no", "n", "neutral or dissatisfied", "dissatisfied", "neutral"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".feather"}:
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    # R's write.csv adds an unnamed row-number column; it carries no information.
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    return df.drop(columns=unnamed)


def coerce_target(series: pd.Series, positive_label=True) -> np.ndarray:
    """Map TRUE/FALSE, 1/0, 'satisfied'/'neutral or dissatisfied' ... to int {0,1}. Raises on unknown tokens."""
    if series.dtype == bool:
        out = series.astype(int).to_numpy()
    else:
        tokens = series.astype(str).str.strip().str.lower()
        mapped = np.where(tokens.isin(TRUE_TOKENS), 1, np.where(tokens.isin(FALSE_TOKENS), 0, -1))
        if (mapped == -1).any():
            bad = sorted(set(tokens[mapped == -1]))[:5]
            raise ValueError(f"Unrecognised target values {bad}; extend TRUE_TOKENS/FALSE_TOKENS in io_utils.py")
        out = mapped.astype(int)
    if positive_label is False:
        out = 1 - out
    return out


def load_training_data(path: str | Path, cfg: dict):
    """Returns X (features only), y (0/1), ids (or None), info (dict of facts about the load)."""
    df = read_table(path)
    dcfg = cfg["data"]
    target, id_col = dcfg["target"], dcfg.get("id_column")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    info = {"path": str(path), "rows_raw": int(len(df)), "columns_raw": int(df.shape[1])}
    if dcfg.get("sample_rows") and len(df) > dcfg["sample_rows"]:
        df = df.sample(dcfg["sample_rows"], random_state=cfg["project"]["seed"]).reset_index(drop=True)
        info["sampled_rows"] = int(len(df))
    missing_target = df[target].isna()
    if missing_target.any():
        info["rows_dropped_missing_target"] = int(missing_target.sum())
        df = df.loc[~missing_target].reset_index(drop=True)
    y = pd.Series(coerce_target(df[target], dcfg.get("positive_label", True)), name=target)
    ids = df[id_col].copy() if id_col and id_col in df.columns else None
    X = df.drop(columns=[c for c in (target, id_col) if c and c in df.columns]).reset_index(drop=True)
    info["rows"] = int(len(X))
    return X, y, ids, info


def load_unseen_data(path: str | Path, cfg: dict):
    """Unseen data: the target is optional (if present it is returned so metrics can be computed)."""
    df = read_table(path)
    dcfg = cfg["data"]
    target, id_col = dcfg["target"], dcfg.get("id_column")
    y = None
    if target in df.columns:
        y = pd.Series(coerce_target(df[target], dcfg.get("positive_label", True)), name=target)
    ids = df[id_col].copy() if id_col and id_col in df.columns else pd.Series(np.arange(len(df)), name=id_col or "id")
    X = df.drop(columns=[c for c in (target, id_col) if c and c in df.columns]).reset_index(drop=True)
    return X, y, ids
