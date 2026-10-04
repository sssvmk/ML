"""Unit tests for splits.py: split logic and the data audit."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splits  # noqa: E402


def test_three_way_is_a_partition_with_expected_sizes():
    tr, va, te = splits.split_three_way(1000, val_size=0.2, test_size=0.2, seed=1)
    assert (len(tr), len(va), len(te)) == (600, 200, 200)
    allidx = np.concatenate([tr, va, te])
    assert len(set(allidx)) == 1000 and set(allidx) == set(range(1000))  # disjoint and complete


def test_three_way_deterministic_and_seed_sensitive():
    a = splits.split_three_way(500, seed=3)
    b = splits.split_three_way(500, seed=3)
    c = splits.split_three_way(500, seed=4)
    assert all((x == y).all() for x, y in zip(a, b))
    assert not (a[2] == c[2]).all()


def test_stratified_three_way_preserves_class_balance():
    y = np.array([0] * 900 + [1] * 100)
    tr, va, te = splits.split_three_way(1000, y, val_size=0.2, test_size=0.2, seed=0, stratify=True)
    for idx in (tr, va, te):
        assert y[idx].mean() == pytest.approx(0.1, abs=0.005)


def test_carve_validation_counts_and_stratification():
    y = np.repeat(np.arange(10), 600)  # 6000 rows, balanced
    tr, va = splits.carve_validation(6000, y, val_size=500, seed=0, stratify=True)
    assert (len(tr), len(va)) == (5500, 500)
    assert set(tr).isdisjoint(set(va))
    assert (np.bincount(y[va], minlength=10) == 50).all()


@pytest.mark.parametrize("kwargs", [dict(val_size=0.6, test_size=0.6), dict(val_size=0, test_size=0.2)])
def test_bad_sizes_rejected(kwargs):
    with pytest.raises(ValueError):
        splits.split_three_way(100, **kwargs)


# --------------------------------- audit ---------------------------------- #
def make_cls(n=300, k=3, seed=0, dup=False, nan=False, drop_class=False):
    rng = np.random.default_rng(seed)
    sets = {}
    for name in ("train", "val", "test"):
        x = rng.normal(size=(n, 4)).astype(np.float32)
        y = np.arange(n) % k
        if drop_class and name == "val":
            y = np.where(y == 2, 0, y)
        sets[name] = [x, y]
    if dup:
        sets["test"][0][:5] = sets["train"][0][:5]
    if nan:
        sets["val"][0][0, 0] = np.nan
    return {n_: TensorDataset(torch.tensor(x), torch.tensor(y).long()) for n_, (x, y) in sets.items()}


INFO_CLS = dict(task="classification", out_dim=3, y_mean=0.0, y_std=1.0)


def test_audit_clean_data_passes():
    rep = splits.audit_splits(make_cls(), INFO_CLS)
    assert rep["ok"] and not rep["errors"] and not rep["warnings"]
    assert rep["sizes"] == {"train": 300, "val": 300, "test": 300}
    assert all(v == 0 for v in rep["overlap_identical_rows"].values())


def test_audit_flags_nan_as_error():
    rep = splits.audit_splits(make_cls(nan=True), INFO_CLS)
    assert not rep["ok"] and any("non-finite" in e for e in rep["errors"])


def test_audit_flags_missing_class_as_error():
    rep = splits.audit_splits(make_cls(drop_class=True), INFO_CLS)
    assert not rep["ok"] and any("missing" in e for e in rep["errors"])


def test_audit_counts_identical_rows_across_splits():
    rep = splits.audit_splits(make_cls(dup=True), INFO_CLS)
    assert rep["ok"]  # duplicates are a warning, not a stop
    assert rep["overlap_identical_rows"]["train&test"] == 5
    assert any("identical feature rows" in w for w in rep["warnings"])


def test_audit_regression_target_stats_and_shift_warning():
    rng = np.random.default_rng(0)

    def ds(shift=0.0):
        x = torch.tensor(rng.normal(size=(500, 3)), dtype=torch.float32)
        y = torch.tensor(rng.normal(size=(500, 1)) + shift, dtype=torch.float32)
        return TensorDataset(x, y)

    info = dict(task="regression", out_dim=1, y_mean=2.0, y_std=1.5)
    ok = splits.audit_splits({"train": ds(), "val": ds(), "test": ds()}, info)
    assert ok["ok"] and "target_stats" in ok
    shifted = splits.audit_splits({"train": ds(), "val": ds(), "test": ds(shift=1.0)}, info)
    assert any("target mean" in w for w in shifted["warnings"])
