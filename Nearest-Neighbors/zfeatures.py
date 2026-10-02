"""
features.py - temporal split + feature engineering for the Zillow Zestimate data (target: logerror).

Split (by transactiondate)
    train : <= 2017-02-28      (2016 .. Feb 2017)
    val   : 2017-03-01 .. 2017-07-31
    test  : >= 2017-08-01

Everything that is "learned" (medians, clip bounds, category levels, scaling ...) is fitted on TRAIN ONLY.

Pipeline
    1. derived features (lat/lon scaling + quadratic terms, age, ratios, tract id from censustractandblock)
    2. 'none means absent' count/area columns -> 0 (zero-fill) ; flag columns -> 0/1
    3. missing-indicator columns for numeric features with > 5 % missing in train
    4. log1p for strongly right-skewed non-negative numerics (train skew > 2)
    5. median imputation, winsorising at train 0.5/99.5 percentiles
    6. categorical columns -> one-hot (rare levels pooled into 'other', NaN is its own level, reference level dropped)
    7. standardise every column with train mean / std
    Target for FITTING = logerror clipped at the train 1st / 99th percentile; evaluation always uses raw logerror.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2017-02-28")
VAL_START, VAL_END = pd.Timestamp("2017-03-01"), pd.Timestamp("2017-07-31")
TEST_START = pd.Timestamp("2017-08-01")

# ---- column roles, taken from the Zillow data dictionary --------------------------------------------------------
CAT_LOW = ["airconditioningtypeid", "architecturalstyletypeid", "buildingclasstypeid", "decktypeid",
           "heatingorsystemtypeid", "propertylandusetypeid", "storytypeid", "typeconstructiontypeid", "fips",
           "regionidcounty"]
CAT_HIGH = ["regionidcity", "regionidzip", "regionidneighborhood", "propertycountylandusecode", "propertyzoningdesc",
            "tract"]                                   # tract is derived from censustractandblock
FLAG_COLS = ["hashottuborspa", "fireplaceflag", "taxdelinquencyflag", "pooltypeid2", "pooltypeid7", "pooltypeid10"]
ZERO_FILL = ["poolcnt", "poolsizesum", "fireplacecnt", "garagecarcnt", "garagetotalsqft", "basementsqft",
             "yardbuildingsqft17", "yardbuildingsqft26", "threequarterbathnbr", "taxdelinquencyyear"]
NUMERIC_BASE = ["bathroomcnt", "bedroomcnt", "buildingqualitytypeid", "calculatedbathnbr", "finishedfloor1squarefeet",
                "calculatedfinishedsquarefeet", "finishedsquarefeet12", "finishedsquarefeet13", "finishedsquarefeet15",
                "finishedsquarefeet50", "finishedsquarefeet6", "fullbathcnt", "latitude", "longitude",
                "lotsizesquarefeet", "numberofstories", "roomcnt", "unitcnt", "yearbuilt",
                "structuretaxvaluedollarcnt", "taxvaluedollarcnt", "assessmentyear", "landtaxvaluedollarcnt", "taxamount"]
TOP_N_HIGH = 25            # levels kept for high-cardinality categoricals
MIN_FREQ_LOW = 0.005       # min train frequency for low-cardinality levels
MISS_THRESHOLD = 0.05


def split_masks(dates: pd.Series):
    return {"train": dates <= TRAIN_END, "val": (dates >= VAL_START) & (dates <= VAL_END), "test": dates >= TEST_START}


def _col(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def _flag(df, c):
    if c not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df[c].astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "y", "yes", "t"]).astype(float)


def _cat(df, c):
    if c == "tract":
        raw = _col(df, "censustractandblock")
        s = raw.map(lambda v: str(int(v))[:10] if pd.notna(v) and v > 0 else "missing")
        return s
    if c not in df.columns:
        return pd.Series("missing", index=df.index)
    s = df[c]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 0.9 * s.notna().sum() and s.notna().any():       # numeric-looking id -> clean integer string
        return num.map(lambda v: str(int(v)) if pd.notna(v) else "missing")
    return s.astype(str).where(s.notna(), "missing").str.strip()


def _div(a, b):
    return pd.Series(np.where(b > 0, a / b.where(b > 0, 1.0), np.nan), index=a.index)


class FeatureBuilder:
    def fit(self, df: pd.DataFrame, y: np.ndarray):
        self.lat0 = float(_col(df, "latitude").mean() / 1e6) if "latitude" in df else 0.0
        self.lon0 = float(_col(df, "longitude").mean() / 1e6) if "longitude" in df else 0.0
        num, flags, cats = self._assemble(df)
        n = len(num)
        nan_frac = num.isna().mean()
        self.num_cols = [c for c in num.columns if nan_frac[c] < 1.0]
        num = num[self.num_cols]
        self.miss_cols = [c for c in self.num_cols if c not in ZERO_FILL and MISS_THRESHOLD < nan_frac[c] < 1.0]
        # log1p for strongly right-skewed non-negative columns
        self.log_cols = []
        for c in self.num_cols:
            v = num[c].dropna()
            if len(v) > 10 and v.nunique() > 2 and v.min() >= 0 and v.skew() > 2:
                self.log_cols.append(c)
        num_t = self._log(num)
        self.medians = num_t.median().fillna(0.0).to_dict()
        num_t = num_t.fillna(self.medians)
        self.lo = num_t.quantile(0.005).to_dict()
        self.hi = num_t.quantile(0.995).to_dict()
        # categorical levels
        self.cat_levels, self.cat_ref, self.cat_keep = {}, {}, {}
        for c in cats.columns:
            vc = cats[c].value_counts()
            if c in CAT_HIGH:
                keep = list(vc.index[:TOP_N_HIGH])
            else:
                keep = list(vc.index[vc / n >= MIN_FREQ_LOW])
            mapped = cats[c].where(cats[c].isin(keep), "other")
            vcm = mapped.value_counts()
            self.cat_ref[c] = vcm.index[0]
            self.cat_levels[c] = [l for l in vcm.index if l != self.cat_ref[c]]
            self.cat_keep[c] = keep
        # build once on train to learn standardisation and drop constant columns
        X, names, groups = self._design(num_t, flags, cats)
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.keep_cols = np.where(self.std > 1e-8)[0]
        self.feature_names = [names[i] for i in self.keep_cols]
        raw_groups = [groups[i] for i in self.keep_cols]
        _, self.group_ids = np.unique(raw_groups, return_inverse=True)
        self.group_ids = self.group_ids.tolist()
        self.clip_lo, self.clip_hi = (float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99)))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num, flags, cats = self._assemble(df)
        num = num.reindex(columns=self.num_cols)
        num = self._log(num).fillna(self.medians)
        X, _, _ = self._design(num, flags, cats)
        X = (X - self.mean) / np.where(self.std > 1e-8, self.std, 1.0)
        return X[:, self.keep_cols].astype(np.float32)

    # -------------------------------------------------------------------------------------------------------------
    def _assemble(self, df):
        num = pd.DataFrame({c: _col(df, c) for c in NUMERIC_BASE + ZERO_FILL}, index=df.index)
        num["latitude"] = num["latitude"] / 1e6
        num["longitude"] = num["longitude"] / 1e6
        for c in ZERO_FILL:
            num[c] = num[c].fillna(0.0)                                  # 'none means absent'
        lat, lon = num["latitude"] - self.lat0, num["longitude"] - self.lon0
        num["lat_sq"], num["lon_sq"], num["lat_x_lon"] = lat ** 2, lon ** 2, lat * lon
        num["age"] = 2017 - num["yearbuilt"]
        num["bath_per_bed"] = num["bathroomcnt"] / (num["bedroomcnt"] + 1)
        num["extra_rooms"] = num["roomcnt"] - num["bedroomcnt"] - num["bathroomcnt"]
        num["tax_per_sqft"] = _div(num["taxvaluedollarcnt"], num["calculatedfinishedsquarefeet"])
        num["structure_ratio"] = _div(num["structuretaxvaluedollarcnt"], num["taxvaluedollarcnt"])
        num["land_ratio"] = _div(num["landtaxvaluedollarcnt"], num["taxvaluedollarcnt"])
        num["tax_rate"] = _div(num["taxamount"], num["taxvaluedollarcnt"])
        num["living_lot_ratio"] = _div(num["calculatedfinishedsquarefeet"], num["lotsizesquarefeet"])
        num = num.replace([np.inf, -np.inf], np.nan)
        flags = pd.DataFrame({c: _flag(df, c) for c in FLAG_COLS}, index=df.index)
        cats = pd.DataFrame({c: _cat(df, c) for c in CAT_LOW + CAT_HIGH}, index=df.index)
        return num, flags, cats

    def _log(self, num):
        num = num.copy()
        for c in self.log_cols:
            if c in num:
                num[c] = np.log1p(num[c].clip(lower=0))
        return num

    def _design(self, num, flags, cats):
        """num is already log-transformed + imputed (train) ; returns matrix, names, group labels."""
        blocks, names, groups = [], [], []
        raw_num = num.copy()
        lo = pd.Series(self.lo); hi = pd.Series(self.hi)
        blocks.append(raw_num.clip(lower=lo.reindex(raw_num.columns), upper=hi.reindex(raw_num.columns), axis=1).values)
        names += list(raw_num.columns); groups += list(raw_num.columns)
        blocks.append(flags.values); names += list(flags.columns); groups += list(flags.columns)
        for c in cats.columns:
            s = cats[c].where(cats[c].isin(self.cat_keep[c]), "other")
            for lvl in self.cat_levels[c]:
                blocks.append((s == lvl).astype(float).values[:, None])
                names.append(f"{c}={lvl}"); groups.append(f"cat:{c}")
        return np.hstack([b if b.ndim == 2 else b[:, None] for b in blocks]).astype(np.float64), names, groups


def build_prepared(csv_path, prepared_dir, cfg, logger=print):
    """Read zillow.csv, split temporally, engineer features on train, write arrays for all methods."""
    out = Path(prepared_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger(f"reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    df = df.dropna(subset=["transactiondate", "logerror"]).sort_values("transactiondate", kind="stable").reset_index(drop=True)
    masks = split_masks(df["transactiondate"])
    parts = {k: df[m].reset_index(drop=True) for k, m in masks.items()}
    counts = {k: int(len(v)) for k, v in parts.items()}
    logger(f"split sizes: {counts}")
    if min(counts.values()) == 0:
        raise ValueError(f"empty split: {counts}. Check transactiondate coverage (needs rows before 2017-03, in Mar-Jul 2017 and from 2017-08).")

    y_tr = parts["train"]["logerror"].values.astype(float)
    fb = FeatureBuilder().fit(parts["train"], y_tr)

    # ---- missing indicators (computed from the raw NaN pattern) ----
    def with_indicators(part):
        num, _, _ = fb._assemble(part)
        ind = num[fb.miss_cols].isna().astype(np.float32).values if fb.miss_cols else np.zeros((len(part), 0), np.float32)
        return ind

    ind_names = [f"{c}__missing" for c in fb.miss_cols]
    ind_tr = with_indicators(parts["train"])
    ind_keep = np.where(ind_tr.std(0) > 1e-8)[0] if ind_tr.shape[1] else np.array([], int)
    ind_mean, ind_std = (ind_tr.mean(0), ind_tr.std(0)) if ind_tr.shape[1] else (None, None)

    def make_X(part):
        X = fb.transform(part)
        if len(ind_keep):
            ind = (with_indicators(part) - ind_mean) / np.where(ind_std > 1e-8, ind_std, 1.0)
            X = np.hstack([X, ind[:, ind_keep].astype(np.float32)])
        return X

    names = list(fb.feature_names) + [ind_names[i] for i in ind_keep]
    n_base = len(fb.feature_names)
    group_ids = list(fb.group_ids)
    nxt = (max(group_ids) + 1) if group_ids else 0
    group_ids += list(range(nxt, nxt + len(ind_keep)))

    lo, hi = fb.clip_lo, fb.clip_hi
    for k, part in parts.items():
        X = make_X(part)
        np.save(out / f"X_{k}.npy", X)
        y = part["logerror"].values.astype(float)
        np.save(out / f"y_raw_{k}.npy", y)
        if k in ("train", "val"):
            np.save(out / f"y_fit_{k}.npy", np.clip(y, lo, hi))
        np.save(out / f"dates_{k}.npy", part["transactiondate"].values.astype("datetime64[D]"))
        logger(f"  {k}: X {X.shape}")

    meta = {"feature_names": names, "group_ids": group_ids, "n_features": len(names), "clip_bounds": [lo, hi],
            "split_counts": counts,
            "date_ranges": {k: [str(v["transactiondate"].min().date()), str(v["transactiondate"].max().date())] for k, v in parts.items()},
            "log1p_columns": fb.log_cols, "missing_indicator_columns": fb.miss_cols,
            "zero_filled_columns": ZERO_FILL, "n_base_features": n_base,
            "categorical_levels": {c: [fb.cat_ref[c]] + fb.cat_levels[c] for c in fb.cat_levels}}
    (out / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    cfg.to_json(out / "run_config.json")
    logger(f"features: {len(names)} columns ({len(ind_keep)} missing indicators); fit-target clip bounds [{lo:.4f}, {hi:.4f}]")
    return meta
