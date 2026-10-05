"""Exercise the real CIFAR-10 loading path on a tiny fake copy in torchvision's on-disk format (no download)."""
import pickle

import numpy as np
import pytest
from torchvision import datasets

from src.data import DATASETS, build_datasets, build_loaders, get_norm
from src.utils import ROOT, load_config


@pytest.fixture()
def fake_cifar10(tmp_path, monkeypatch):
    base = tmp_path / "cifar-10-batches-py"
    base.mkdir()
    rng = np.random.default_rng(0)

    def dump(name, n):
        labels = (np.arange(n) % 10).tolist()
        data = rng.integers(0, 256, (n, 3072), dtype=np.uint8)
        with open(base / name, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

    for i in range(1, 6):
        dump(f"data_batch_{i}", 100)
    dump("test_batch", 100)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    with open(base / "batches.meta", "wb") as f:
        pickle.dump({"label_names": names}, f)
    import torchvision.datasets.cifar as cifar_module
    monkeypatch.setattr(datasets.CIFAR10, "_check_integrity", lambda self: True)
    monkeypatch.setattr(cifar_module, "check_integrity", lambda *a, **k: True)  # metadata file checksum
    return tmp_path


def test_cifar10_pipeline_loads_and_splits(fake_cifar10):
    cfg = load_config(ROOT / "config" / "train.yaml", [f"data.root={fake_cifar10}", "data.num_workers=0",
                                                       "data.batch_size=16"])
    assert cfg["data"]["dataset"] == "cifar10" and cfg["data"]["num_classes"] == 10
    ds = build_datasets(cfg)
    assert ds["classes"][3] == "cat" and len(ds["classes"]) == 10
    assert ds["data_version"].startswith("cifar10-")
    assert not set(ds["train_idx"].tolist()) & set(ds["val_idx"].tolist())
    assert len(ds["train_idx"]) + len(ds["val_idx"]) == 500 and len(ds["test"]) == 100
    assert ds["norm"] == get_norm(cfg) == (DATASETS["cifar10"]["mean"], DATASETS["cifar10"]["std"])
    import torch
    x, y = next(iter(build_loaders(cfg, ds, torch.device("cpu"))["train"]))
    assert x.shape == (16, 3, 32, 32) and y.max() < 10


def test_num_classes_follows_dataset_not_config():
    cfg = load_config(ROOT / "config" / "train.yaml", ["data.dataset=cifar100", "data.num_classes=10"])
    assert cfg["data"]["num_classes"] == 100
    with pytest.raises(ValueError):
        load_config(ROOT / "config" / "train.yaml", ["data.dataset=imagenet"])


def test_dataset_stats_differ_between_cifar10_and_cifar100():
    assert DATASETS["cifar10"]["mean"] != DATASETS["cifar100"]["mean"]
