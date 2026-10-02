"""
prep_reg.py - temporal split, target handling and predictor choice for the Zillow kernel-method study.

Split (transactiondate):  train <= 2017-02-28 | validation 2017-03-01..2017-07-31 | test >= 2017-08-01
Target       : logerror.  Fitting target = logerror clipped at the TRAIN 1st/99th percentiles; every metric uses the raw logerror.
Binary target: large_miss = 1{|logerror| > train 90th percentile of |logerror|}  (nonparametric logistic regression)
Predictor    : ONE continuous predictor for the 1-D smoothers.  --predictor <column> or "auto": every candidate is scored on TRAIN only
               (5-fold time-blocked CV of a 20-bin step function; best raw-logerror MSE wins) - the score table is written out.
               Transform: winsorise at train 0.5/99.5 percentiles, log1p if train skew > 2 (non-negative, > 20 unique values),
               median-impute, z-score.  Coordinates (latitude, longitude; scaled by 1e6) are z-scored for the thin-plate spline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2017-02-28")
VAL_START, VAL_END = pd.Timestamp("2017-03-01"), pd.Timestamp("2017-07-31")
TEST_START = pd.Timestamp("2017-08-01")

CAT_LOW = ["airconditioningtypeid", "architecturalstyletypeid", "buildingclasstypeid", "decktypeid", "heatingorsystemtypeid",
           "propertylandusetypeid", "storytypeid", "typeconstructiontypeid", "fips", "regionidcounty"]
CAT_HIGH = ["regionidcity", "regionidzip", "regionidneighborhood", "propertycountylandusecode", "propertyzoningdesc", "tract"]
FLAG_COLS = ["hashottuborspa", "fireplaceflag", "taxdelinquencyflag", "pooltypeid2", "pooltypeid7", "pooltypeid10"]
CANDIDATES = ["yearbuilt", "calculatedfinishedsquarefeet", "taxvaluedollarcnt", "structuretaxvaluedollarcnt", "landtaxvaluedollarcnt",
              "taxamount", "lotsizesquarefeet", "bathroomcnt", "bedroomcnt", "roomcnt", "buildingqualitytypeid", "finishedsquarefeet12",
              "garagetotalsqft", "tax_per_sqft", "structure_ratio", "land_ratio", "living_lot_ratio"]


def split_masks(d):
    return {"train": d <= TRAIN_END, "val": (d >= VAL_START) & (d <= VAL_END), "test": d >= TEST_START}


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def candidate_frame(df):
    out = pd.DataFrame({c: _num(df, c) for c in CANDIDATES if c not in ("tax_per_sqft", "structure_ratio", "land_ratio", "living_lot_ratio")})

    def div(a, b):
        return pd.Series(np.where(b > 0, a / b.where(b > 0, 1.0), np.nan), index=df.index)
    tv, sq = _num(df, "taxvaluedollarcnt"), _num(df, "calculatedfinishedsquarefeet")
    out["tax_per_sqft"] = div(tv, sq)
    out["structure_ratio"] = div(_num(df, "structuretaxvaluedollarcnt"), tv)
    out["land_ratio"] = div(_num(df, "landtaxvaluedollarcnt"), tv)
    out["living_lot_ratio"] = div(sq, _num(df, "lotsizesquarefeet"))
    return out.replace([np.inf, -np.inf], np.nan)


class PredictorTransform:
    def fit(self, v):
        v = pd.Series(v).dropna()
        self.lo, self.hi = float(v.quantile(0.005)), float(v.quantile(0.995))
        vv = v.clip(self.lo, self.hi)
        self.log = bool(len(vv) > 20 and vv.nunique() > 20 and vv.min() >= 0 and vv.skew() > 2)
        w = np.log1p(vv) if self.log else vv
        self.med = float(w.median())
        w = w.fillna(self.med)
        self.mu, self.sd = float(w.mean()), float(w.std()) or 1.0
        return self

    def __call__(self, v):
        w = pd.Series(v).clip(self.lo, self.hi)
        w = np.log1p(w) if self.log else w
        return ((w.fillna(self.med) - self.mu) / self.sd).values.astype(float)


def _score_predictor(x, y_fit, y_raw, folds=5, bins=20):
    n = len(x)
    e = np.linspace(0, n, folds + 1).astype(int)
    se = 0.0
    for k in range(folds):
        te = np.zeros(n, bool); te[e[k]:e[k + 1]] = True
        edges = np.unique(np.quantile(x[~te], np.linspace(0, 1, bins + 1)[1:-1]))
        b_tr, b_te = np.searchsorted(edges, x[~te], "right"), np.searchsorted(edges, x[te], "right")
        sums = np.bincount(b_tr, weights=y_fit[~te], minlength=len(edges) + 1)
        cnt = np.bincount(b_tr, minlength=len(edges) + 1)
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), y_fit[~te].mean())
        se += np.sum((y_raw[te] - mean[b_te]) ** 2)
    return se / n


def build_prepared(csv_path, out_dir, cfg, logger=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger(f"reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    df = df.dropna(subset=["transactiondate", "logerror"]).sort_values("transactiondate", kind="stable").reset_index(drop=True)
    masks = split_masks(df["transactiondate"])
    parts = {k: df[m].reset_index(drop=True) for k, m in masks.items()}
    counts = {k: len(v) for k, v in parts.items()}
    logger(f"split sizes: {counts}")
    if min(counts.values()) == 0:
        raise ValueError(f"empty split {counts}: need rows before 2017-03, in Mar-Jul 2017 and from 2017-08")
    y = {k: v["logerror"].values.astype(float) for k, v in parts.items()}
    lo, hi = np.quantile(y["train"], [0.01, 0.99])
    cand = {k: candidate_frame(v) for k, v in parts.items()}
    base = float(np.mean((y["train"] - np.clip(y["train"], lo, hi).mean()) ** 2))
    rows = []
    for c in CANDIDATES:
        miss = float(cand["train"][c].isna().mean())
        if miss > 0.2 or cand["train"][c].nunique() < 10:
            rows.append({"candidate": c, "pct_missing": 100 * miss, "cv_mse": np.nan, "gain_vs_mean_pct": np.nan, "eligible": False})
            continue
        tf = PredictorTransform().fit(cand["train"][c])
        sc = _score_predictor(tf(cand["train"][c]), np.clip(y["train"], lo, hi), y["train"])
        rows.append({"candidate": c, "pct_missing": 100 * miss, "cv_mse": sc, "gain_vs_mean_pct": 100 * (1 - sc / base), "eligible": True})
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "predictor_selection.csv", index=False)
    ranked = list(sel[sel["eligible"]].sort_values("cv_mse")["candidate"])
    if cfg.predictor != "auto":
        if cfg.predictor not in CANDIDATES:
            raise ValueError(f"--predictor must be one of {CANDIDATES} or 'auto'")
        ranked = [cfg.predictor] + [c for c in ranked if c != cfg.predictor]
    multi = ranked[: max(2, cfg.n_multi)]
    tfs = {c: PredictorTransform().fit(cand["train"][c]) for c in multi}
    for k in parts:
        xm = np.column_stack([tfs[c](cand[k][c]) for c in multi])
        np.save(out / f"x1_{k}.npy", xm[:, 0])
        np.save(out / f"xm_{k}.npy", xm)
        np.save(out / f"y_raw_{k}.npy", y[k])
        np.save(out / f"dates_{k}.npy", parts[k]["transactiondate"].values.astype("datetime64[D]"))
        if k == "train":
            np.save(out / "y_fit_train.npy", np.clip(y[k], lo, hi))
    meta = {"task": "regression", "input_kinds": ["x1", "xm"], "predictor": multi[0], "x1_label": multi[0] + (" (log1p, z-scored)" if tfs[multi[0]].log else " (z-scored)"),
            "multi_predictors": multi, "split_counts": counts, "clip_bounds": [float(lo), float(hi)],
            "date_ranges": {k: [str(v["transactiondate"].min().date()), str(v["transactiondate"].max().date())] for k, v in parts.items()}}
    (out / "prepared_meta.json").write_text(json.dumps(meta, indent=2))
    cfg.to_json(out / "run_config.json")
    logger(f"single predictor: {multi[0]} | structured local regression / RBF use {multi}")
    return meta
