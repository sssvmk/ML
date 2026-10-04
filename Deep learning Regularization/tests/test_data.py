import torch

from regpipe.data import audit_splits, batch_indices, make_splits, stratified_holdout, stratified_subset


def test_dataset_reports_pre_split_and_carves_validation(data):
    assert data.info["dataset_pre_split"] is True
    assert data.info["validation_carved_from_train"] is True
    assert sum(data.info["sizes"].values()) == len(data.y_train) + len(data.y_val) + len(data.y_test)


def test_stratified_holdout_disjoint_and_balanced():
    y = torch.arange(10).repeat(100)
    keep, hold = stratified_holdout(y, 200, seed=1)
    assert len(set(keep.tolist()) & set(hold.tolist())) == 0
    assert len(keep) + len(hold) == len(y)
    assert torch.bincount(y[hold], minlength=10).tolist() == [20] * 10


def test_audit_detects_injected_duplicate(cfg):
    from regpipe.data import load_raw
    raw = load_raw(cfg["data"], 0)
    raw["x_test"][0] = raw["x_train"][0]            # plant an exact duplicate across train and test
    s = make_splits(raw, 400, 0)
    a = audit_splits(s)
    assert a["duplicates_train_test"] + a["duplicates_val_test"] >= 1
    assert a["problems"]


def test_stratified_subset_is_balanced(data):
    idx = stratified_subset(data.y_train, 100, seed=0)
    assert torch.bincount(data.y_train[idx], minlength=10).tolist() == [10] * 10


def test_batches_cover_every_example_once():
    g = torch.Generator().manual_seed(0)
    seen = torch.cat(list(batch_indices(103, 10, g)))
    assert sorted(seen.tolist()) == list(range(103))
