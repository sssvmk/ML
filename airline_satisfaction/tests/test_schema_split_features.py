import numpy as np
import pandas as pd

from airsat.config import load_config
from airsat.features import FeatureEngineer
from airsat.io_utils import coerce_target, load_training_data
from airsat.schema import infer_schema
from airsat.splitting import make_splits, population_report


def _data(csv_path):
    cfg = load_config()
    X, y, ids, info = load_training_data(csv_path, cfg)
    return cfg, X, y, ids


def test_target_coercion_variants():
    assert coerce_target(pd.Series(["TRUE", "FALSE", "true"])).tolist() == [1, 0, 1]
    assert coerce_target(pd.Series([True, False])).tolist() == [1, 0]
    assert coerce_target(pd.Series(["satisfied", "neutral or dissatisfied"])).tolist() == [1, 0]


def test_schema_roles(csv_path):
    cfg, X, y, _ = _data(csv_path)
    s = infer_schema(X, cfg)
    assert set(s.continuous) == {"Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"}
    assert "Online boarding" in s.ordinal and len(s.ordinal) == 13
    assert set(s.nominal) == {"Gender", "Customer Type", "Type of Travel", "Class"}


def test_split_is_disjoint_complete_and_stratified(csv_path):
    cfg, X, y, _ = _data(csv_path)
    sp = make_splits(X, y, cfg)
    all_idx = np.concatenate(list(sp.values()))
    assert len(all_idx) == len(set(all_idx)) == len(X)
    assert abs(len(sp["train"]) / len(X) - 0.70) < 0.03
    for k in ("validation", "test"):
        assert abs(y.iloc[sp[k]].mean() - y.mean()) < 0.03
    rep = population_report(X, y, sp, infer_schema(X, cfg), cfg)
    assert rep["representative"]


def test_duplicates_never_straddle_splits(csv_path):
    cfg, X, y, _ = _data(csv_path)
    Xd = pd.concat([X, X.iloc[:300]], ignore_index=True)
    yd = pd.concat([y, y.iloc[:300]], ignore_index=True)
    sp = make_splits(Xd, yd, cfg)
    h = pd.util.hash_pandas_object(Xd, index=False)
    owner = {}
    for name, idx in sp.items():
        for hv in set(h.iloc[idx]):
            assert owner.setdefault(hv, name) == name, "identical rows ended up in different splits"


def test_feature_engineering_masks_not_applicable_and_is_stateless_on_transform(csv_path):
    cfg, X, y, _ = _data(csv_path)
    s = infer_schema(X, cfg)
    fe = FeatureEngineer(s, cfg["features"]).fit(X)
    row = X.iloc[[0]].copy()
    for c in s.ordinal:
        row[c] = 4
    row["Inflight wifi service"] = 0               # not applicable
    out = fe.transform(row)
    assert out["rating_mean_nz"].iloc[0] == 4.0     # the 0 must not drag the mean down
    assert out["n_zero_ratings"].iloc[0] == 1
    assert set(fe.feature_set.model_columns) <= set(out.columns)
    assert fe.transform(row).equals(out)            # deterministic


def test_feature_engineering_handles_missing_arrival_delay(csv_path):
    cfg, X, y, _ = _data(csv_path)
    s = infer_schema(X, cfg)
    fe = FeatureEngineer(s, cfg["features"]).fit(X)
    row = X.iloc[[0]].copy()
    row["Arrival Delay in Minutes"] = np.nan
    out = fe.transform(row)
    assert out["arr_delay_was_missing"].iloc[0] == 1 and np.isfinite(out["total_delay"].iloc[0])
