"""
Shared plumbing: canonical contract rows -> a regular daily frame (target, historical exogenous, calendar covariates).
Used by every module that feeds a neural network (NeuralForecast adapter and the custom PyTorch modules), so gap handling
is identical everywhere: missing calendar days are never invented silently (`gap_policy` = "error" | "zero" | "ffill").
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .utils import endogenous_series, exogenous_frame

CALENDAR_COLS = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"]


def calendar_features(dates) -> pd.DataFrame:
    """Future-known covariates derivable from the date alone (day-of-week / day-of-month / month, sin-cos)."""
    d = pd.DatetimeIndex(dates)
    f = pd.DataFrame(index=d)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * d.dayofweek / 7), np.cos(2 * np.pi * d.dayofweek / 7)
    f["dom_sin"], f["dom_cos"] = np.sin(2 * np.pi * (d.day - 1) / 31), np.cos(2 * np.pi * (d.day - 1) / 31)
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * (d.month - 1) / 12), np.cos(2 * np.pi * (d.month - 1) / 12)
    return f


class DailyFrameMixin:
    """Requires `self.hyperparameters`, `supports_hist_exog`."""
    supports_hist_exog = True

    # ---- data mapping ---------------------------------------------------------------------------------------
    def _fill(self, s: pd.Series, idx: pd.DatetimeIndex, what: str) -> pd.Series:
        missing = idx.difference(s.index)
        policy = self.hyperparameters.get("gap_policy", "error")
        if len(missing) and policy == "error":
            raise ValueError(f"{what}: {len(missing)} calendar days missing between {idx.min().date()} and "
                             f"{idx.max().date()}; neural modules need a regular daily index. Set hyperparameter "
                             f"gap_policy to 'zero' or 'ffill' to choose how gaps are filled (not assumed).")
        s = s.reindex(idx)
        if policy == "zero":
            return s.fillna(0.0)
        if policy == "ffill":
            return s.ffill().bfill()
        return s

    def _frame(self, segment_df: pd.DataFrame, series_id: str | None = None, exog_cols: list[str] | None = None) -> pd.DataFrame:
        y = endogenous_series(segment_df)
        idx = pd.date_range(y.index.min(), y.index.max(), freq=self.frequency)
        y = self._fill(y, idx, "endogenous series")
        df = pd.DataFrame({"unique_id": series_id or str(segment_df["segment_id"].iloc[0]), "ds": idx, "y": y.to_numpy(dtype=float)})
        use_exog = bool(self.hyperparameters.get("use_exog", True)) and self.supports_hist_exog
        if use_exog:
            ex = exogenous_frame(segment_df)
            cols = exog_cols if exog_cols is not None else list(ex.columns)
            if cols:
                missing_cols = [c for c in cols if c not in ex.columns]
                if missing_cols:
                    raise ValueError(f"exogenous datasets {missing_cols} were used in training but are absent from the supplied rows")
                for c in cols:
                    df[c] = self._fill(ex[c].astype(float), idx, f"exogenous {c}").to_numpy()
                if df[cols].isna().any().any():
                    raise ValueError("exogenous columns contain gaps after alignment (see gap_policy)")
        return df

