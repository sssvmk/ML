import numpy as np
import torch

from src.data import MEAN, STD, build_datasets, eval_transform, pil_to_uint8, preprocess_uint8_batch, stratified_split
from src.utils import ROOT, load_config


def _cfg():
    cfg = load_config(ROOT / "config" / "train.yaml", ["data.dataset=synthetic", "data.synthetic_size=1000"])
    return cfg


def test_split_is_disjoint_stratified_and_complete():
    targets = np.repeat(np.arange(100), 500)
    tr, va = stratified_split(targets, 0.1, seed=0)
    assert not set(tr) & set(va)
    assert len(tr) + len(va) == len(targets)
    assert np.bincount(targets[va], minlength=100).min() == 50  # every class present in val


def test_datasets_no_leakage_and_all_classes():
    ds = build_datasets(_cfg())
    assert not set(ds["train_idx"].tolist()) & set(ds["val_idx"].tolist())
    assert len(ds["classes"]) == _cfg()["data"]["num_classes"] == 10


def test_serving_preprocessing_matches_training_transform():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    from PIL import Image
    tf = eval_transform()
    a = torch.stack([tf(Image.fromarray(x)) for x in arr])
    b = preprocess_uint8_batch(arr)
    assert torch.allclose(a, b, atol=1e-6)
    assert len(MEAN) == len(STD) == 3


def test_preprocess_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        preprocess_uint8_batch(np.zeros((1, 32, 32, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        preprocess_uint8_batch(np.zeros((1, 3, 32, 32), dtype=np.uint8))


def test_pil_to_uint8_shape():
    from PIL import Image
    out = pil_to_uint8(Image.new("RGB", (100, 60)))
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
