"""Step 4: feature engineering (stateless apart from which indicator columns exist, learned on TRAIN only).

Why these features (each is cheap, available at prediction time, and addresses a specific modelling weakness):
  * rating aggregates   - 13 correlated 0-5 ratings; aggregates give linear models/MLPs a low-noise summary
                          and give trees a shortcut to "overall experience" (0 means *not applicable*, so it is
                          masked out of the aggregates and counted separately)
  * weak-link features  - min / n_low capture "one terrible service ruins the flight", which a mean hides
  * delay features      - raw delays are extremely skewed (most are 0): log1p, recovered time, on-time flag
  * distance features   - log scale + bands, delay per 1000 km (the same delay hurts more on a short flight)
  * segment interactions- business-travel x loyalty, digital rating x business travel: linear models cannot
                          discover interactions by themselves
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import Schema

RATING_GROUPS = {
    "digital": ["Inflight wifi service", "Ease of Online booking", "Online boarding"],
    "comfort": ["Seat comfort", "Leg room service", "Food and drink", "Inflight entertainment", "Cleanliness"],
    "service": ["On-board service", "Checkin service", "Baggage handling"],
    "logistics": ["Departure/Arrival time convenient", "Gate location"],
}
CLASS_ORDER = {"eco": 0, "eco plus": 1, "business": 2}
AGE_BINS = [-np.inf, 17, 24, 39, 59, np.inf]
AGE_LABELS = ["<18", "18-24", "25-39", "40-59", "60+"]
DIST_BINS = [-np.inf, 800, 2500, np.inf]


@dataclass
class FeatureSet:
    continuous: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    nominal: list[str] = field(default_factory=list)
    meta: list[str] = field(default_factory=list)       # kept for slicing/plots, never fed to a model

    @property
    def model_columns(self) -> list[str]:
        return self.continuous + self.ordinal + self.binary + self.nominal

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSet":
        return cls(**d)


def _find(cols, *needles):
    for c in cols:
        low = c.lower()
        if all(n in low for n in needles):
            return c
    return None


class FeatureEngineer:
    def __init__(self, schema: Schema, features_cfg: dict | None = None):
        self.schema = schema
        self.cfg = features_cfg or {}
        self.enabled = self.cfg.get("engineer", True)
        self.late_threshold = self.cfg.get("arrival_delay_on_time_threshold", 15)
        self.zero_is_na = self.cfg.get("zero_rating_means_not_applicable", True)
        cols = schema.continuous + schema.ordinal + schema.nominal
        self.dep_delay = _find(cols, "departure", "delay")
        self.arr_delay = _find(cols, "arrival", "delay")
        self.distance = _find(cols, "distance")
        self.age = "Age" if "Age" in cols else None   # exact match: "age" is also inside "Baggage"
        self.cls = _find(cols, "class")
        self.travel = _find(cols, "type of travel")
        self.customer = _find(cols, "customer type")
        self.rating_cols = list(schema.ordinal)
        self.na_flag_cols: list[str] = []
        self.feature_set_: FeatureSet | None = None

    # -- fitting learns only which rating columns ever contain 0 ("not applicable") ----------------------
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        if self.zero_is_na:
            self.na_flag_cols = [c for c in self.rating_cols if (pd.to_numeric(df[c], errors="coerce") == 0).mean() >= 0.001]
        out = self.transform(df.head(1000).copy())  # build the feature set from a real transform
        return self._finalise(out)

    def _finalise(self, out: pd.DataFrame) -> "FeatureEngineer":
        s = self.schema
        fs = FeatureSet()
        engineered_cont = ["rating_mean_nz", "rating_std_nz", "rating_min_nz", "rating_range_nz",
                           "digital_mean_nz", "comfort_mean_nz", "service_mean_nz", "logistics_mean_nz",
                           "total_delay", "delay_recovered", "log1p_dep_delay", "log1p_arr_delay", "log_distance",
                           "delay_per_1000km", "digital_x_business"]
        engineered_ord = ["n_zero_ratings", "n_low_ratings", "n_high_ratings", "class_ordinal",
                          "distance_band_code", "age_band_code"]
        engineered_bin = [f"{c}__na" for c in self.na_flag_cols] + [
            "is_arrival_delayed", "arr_delay_was_missing", "is_business_travel", "is_loyal", "loyal_x_business"]
        engineered_nom = ["travel_class"]
        fs.continuous = [c for c in s.continuous if c in out.columns]
        fs.ordinal = [c for c in s.ordinal if c in out.columns]
        fs.nominal = [c for c in s.nominal if c in out.columns]
        if self.enabled:
            fs.continuous += [c for c in engineered_cont if c in out.columns]
            fs.ordinal += [c for c in engineered_ord if c in out.columns]
            fs.binary += [c for c in engineered_bin if c in out.columns]
            fs.nominal += [c for c in engineered_nom if c in out.columns]
        fs.meta = [c for c in ["age_band"] if c in out.columns]
        self.feature_set_ = fs
        return self

    # -- transform --------------------------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.schema.continuous + self.schema.ordinal:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        if not self.enabled:
            return out

        rating = out[[c for c in self.rating_cols if c in out.columns]]
        if rating.shape[1]:
            nz = rating.where(rating > 0) if self.zero_is_na else rating
            out["rating_mean_nz"] = nz.mean(axis=1)
            out["rating_std_nz"] = nz.std(axis=1).fillna(0.0)
            out["rating_min_nz"] = nz.min(axis=1)
            out["rating_range_nz"] = nz.max(axis=1) - nz.min(axis=1)
            out["n_zero_ratings"] = (rating == 0).sum(axis=1)
            out["n_low_ratings"] = ((rating >= 1) & (rating <= 2)).sum(axis=1)
            out["n_high_ratings"] = (rating >= 4).sum(axis=1)
            for group, members in RATING_GROUPS.items():
                present = [m for m in members if m in nz.columns]
                if present:
                    out[f"{group}_mean_nz"] = nz[present].mean(axis=1)
            for c in self.na_flag_cols:
                if c in out.columns:
                    out[f"{c}__na"] = (out[c] == 0).astype(int)

        if self.dep_delay and self.arr_delay:
            dep = out[self.dep_delay]
            arr_missing = out[self.arr_delay].isna()
            arr = out[self.arr_delay].fillna(dep)         # arrival delay tracks departure delay closely
            out["arr_delay_was_missing"] = arr_missing.astype(int)
            out["total_delay"] = dep.fillna(0) + arr
            out["delay_recovered"] = dep.fillna(0) - arr
            out["log1p_dep_delay"] = np.log1p(dep.clip(lower=0))
            out["log1p_arr_delay"] = np.log1p(arr.clip(lower=0))
            out["is_arrival_delayed"] = (arr > self.late_threshold).astype(int)
            if self.distance:
                out["delay_per_1000km"] = arr.clip(lower=0) / (out[self.distance].clip(lower=1) / 1000.0)
        if self.distance:
            d = out[self.distance]
            out["log_distance"] = np.log1p(d.clip(lower=0))
            out["distance_band_code"] = pd.cut(d, DIST_BINS, labels=False)
        if self.age:
            a = out[self.age]
            out["age_band"] = pd.cut(a, AGE_BINS, labels=AGE_LABELS).astype(object)
            out["age_band_code"] = pd.cut(a, AGE_BINS, labels=False)
        if self.cls:
            out["class_ordinal"] = out[self.cls].astype(str).str.strip().str.lower().map(CLASS_ORDER)
        if self.travel:
            out["is_business_travel"] = out[self.travel].astype(str).str.lower().str.contains("business").astype(int)
        if self.customer:
            low = out[self.customer].astype(str).str.lower()
            out["is_loyal"] = (low.str.contains("loyal") & ~low.str.contains("disloyal")).astype(int)
        if "is_business_travel" in out and "is_loyal" in out:
            out["loyal_x_business"] = out["is_business_travel"] * out["is_loyal"]
        if "digital_mean_nz" in out and "is_business_travel" in out:
            out["digital_x_business"] = out["digital_mean_nz"] * out["is_business_travel"]
        if self.cls and self.travel:
            out["travel_class"] = out[self.travel].astype(str) + " | " + out[self.cls].astype(str)
        return out

    @property
    def feature_set(self) -> FeatureSet:
        if self.feature_set_ is None:
            raise RuntimeError("FeatureEngineer.fit must be called first")
        return self.feature_set_
