"""Input validation for inference (batch CLI and HTTP service share this)."""
from __future__ import annotations

import pandas as pd

from .schema import Schema


class InputValidationError(ValueError):
    pass


def validate_input(df: pd.DataFrame, schema: Schema | dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean frame with required columns only, list of non-fatal warnings). Raise on fatal problems."""
    if isinstance(schema, dict):
        schema = Schema.from_dict(schema)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise InputValidationError(f"Missing required columns: {missing}")
    warnings_: list[str] = []
    for c in schema.continuous + schema.ordinal:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        coerced = int(df[c].isna().sum() - before)
        if coerced:
            warnings_.append(f"{coerced} non-numeric values in '{c}' treated as missing")
    for c, (lo, hi) in schema.ordinal_ranges.items():
        bad = int(((df[c] < lo) | (df[c] > hi)).sum())
        if bad:
            warnings_.append(f"{bad} values of '{c}' outside the training range [{lo:g}, {hi:g}]")
    for c in schema.nominal:
        df[c] = df[c].astype("object").where(df[c].notna(), None)
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        unseen = set(map(str, df[c].dropna().unique())) - set(schema.categories.get(c, []))
        if unseen:
            warnings_.append(f"unseen categories in '{c}': {sorted(unseen)[:5]} (encoded as all-zero / -1)")
    n_missing = int(df[schema.required_columns].isna().sum().sum())
    if n_missing:
        warnings_.append(f"{n_missing} missing cells will be imputed with training statistics")
    return df, warnings_
