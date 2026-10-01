"""Small shared helpers for pulling endogenous/exogenous series out of
canonical contract rows. Not a shared statistical-test library (PRD §6
is explicit that none is assumed by default) -- just data plumbing every
algorithm module needs regardless of its modeling approach."""

from __future__ import annotations
import pandas as pd


def endogenous_series(segment_df: pd.DataFrame) -> pd.Series:
    """Return the endogenous series indexed by date, summed across any
    endogenous datasets present (e.g. AR_Actual_Cleared + Bank_Inflow_EBS
    both feeding one inflow segment)."""
    endog = segment_df[segment_df["series_role"] == "endogenous"]
    if endog.empty:
        raise ValueError("no endogenous rows in segment_df")
    s = endog.groupby("date")["value"].sum().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def exogenous_frame(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide date x dataset frame of exogenous series, or an
    empty frame (same index as endogenous) if none exist."""
    exog = segment_df[segment_df["series_role"] == "exogenous"]
    if exog.empty:
        return pd.DataFrame()
    wide = exog.pivot_table(index="date", columns="dataset", values="value", aggfunc="sum")
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.sort_index()


def future_dates(last_date: pd.Timestamp, horizon: int, freq: str = "D") -> pd.DatetimeIndex:
    """The next `horizon` periods strictly after `last_date`, at `freq` (default daily -- this request made
    frequency configurable rather than hard-coded; callers pass their module's configured `self.frequency`).
    For an unanchored freq like "D", `last_date` is itself a valid grid point and the generated range starts
    there, so dropping its first element is correct. For an ANCHORED freq (e.g. "W" week-ending-Sunday, "ME"
    month-end), pandas instead starts the range at the first grid point >= last_date, which can already be
    strictly after it -- blindly dropping element 0 would then discard a genuinely future date. Filtering by
    `> last_date` (rather than always dropping index 0) is correct in both cases."""
    last_date = pd.Timestamp(last_date)
    dr = pd.date_range(last_date, periods=horizon + 1, freq=freq)
    return dr[dr > last_date][:horizon]


def next_period_after(cutoff: pd.Timestamp, freq: str = "D") -> pd.Timestamp:
    """The first timestamp strictly after `cutoff` at `freq` -- generalises the `cutoff + Timedelta(days=1)` idiom
    to any frequency (a week or a month after `cutoff` is not `cutoff` plus one day), and to an ANCHORED freq
    where the next grid point after `cutoff` may not be exactly one period-length away."""
    cutoff = pd.Timestamp(cutoff)
    dr = pd.date_range(cutoff, periods=2, freq=freq)
    return dr[dr > cutoff][0]


def future_known_datasets(segment_df: pd.DataFrame, min_future: int = 1) -> list[str]:
    """
    Future-known covariates, inferred from STRUCTURE only (decision on G-28/G-29): an exogenous series that has at least
    `min_future` values dated AFTER the last endogenous observation is known ahead of time. The algorithm does not know or
    care what the series is or where it came from -- only its dates. Returned sorted, so positions are deterministic.
    Point-in-time correctness of such a series (nothing in it that was unknown at the forecast origin) is the Data Module's
    responsibility; an algorithm cannot check it.
    """
    endog = segment_df[segment_df["series_role"] == "endogenous"]
    if endog.empty:
        return []
    last = endog["date"].max()
    ex = segment_df[segment_df["series_role"] == "exogenous"]
    out = []
    for name, g in ex.groupby("dataset"):
        if int((g["date"] > last).sum()) >= min_future:
            out.append(str(name))
    return sorted(out)


def future_length(segment_df: pd.DataFrame, dataset: str) -> int:
    """Number of dated values of `dataset` after the last endogenous observation."""
    last = segment_df.loc[segment_df["series_role"] == "endogenous", "date"].max()
    g = segment_df[(segment_df["series_role"] == "exogenous") & (segment_df["dataset"] == dataset)]
    return int((g["date"] > last).sum())
