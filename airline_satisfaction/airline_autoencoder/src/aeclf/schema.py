"""Step 2a: decide which columns are continuous, ordinal, nominal, identifiers or ignored.

Rules (documented because they drive every later step):
  * object / bool / category dtype                         -> nominal
  * numeric, integer-valued, <= `max_ordinal_levels` levels -> ordinal (survey ratings such as 0-5)
  * any other numeric                                      -> continuous
  * user overrides in config.schema_overrides always win.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Schema:
    continuous: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    nominal: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    ordinal_ranges: dict = field(default_factory=dict)
    numeric_ranges: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)

    @property
    def required_columns(self) -> list[str]:
        return self.continuous + self.ordinal + self.nominal

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(**d)


def infer_schema(X: pd.DataFrame, cfg: dict, max_ordinal_levels: int = 10) -> Schema:
    ov = cfg.get("schema_overrides", {}) or {}
    s = Schema()
    forced = {}
    for role in ("continuous", "ordinal", "nominal", "ignore"):
        for c in ov.get(role, []) or []:
            forced[c] = role
    for col in X.columns:
        ser = X[col]
        nun = ser.nunique(dropna=True)
        if col in forced:
            role, why = forced[col], "config override"
        elif pd.api.types.is_bool_dtype(ser) or not pd.api.types.is_numeric_dtype(ser):
            # covers object, category and pandas>=3 'str' dtypes
            role, why = "nominal", f"non-numeric dtype ({ser.dtype}), {nun} levels"
        elif nun <= 1:
            role, why = "ignore", "constant column carries no information"
        else:
            numeric = pd.to_numeric(ser, errors="coerce")
            non_null = numeric.dropna()
            integer_valued = bool(len(non_null)) and bool(np.all(np.isclose(non_null, np.round(non_null))))
            if integer_valued and nun <= max_ordinal_levels:
                role, why = "ordinal", f"integer-valued with {nun} levels"
            else:
                role, why = "continuous", f"numeric with {nun} distinct values"
        s.reasons[col] = f"{role}: {why}"
        if role == "ignore":
            s.ignored.append(col)
        elif role == "nominal":
            s.nominal.append(col)
            s.categories[col] = sorted(map(str, ser.dropna().unique()))
        elif role == "ordinal":
            s.ordinal.append(col)
            num = pd.to_numeric(ser, errors="coerce")
            s.ordinal_ranges[col] = [float(num.min()), float(num.max())]
        else:
            s.continuous.append(col)
            num = pd.to_numeric(ser, errors="coerce")
            s.numeric_ranges[col] = [float(num.min()), float(num.max())]
    return s
