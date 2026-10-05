import numpy as np
import pytest
import torch
from smoke_over import SMOKE
from voc_fixture import make_coco, make_voc_year

from detr_voc.config import load_config
from detr_voc.data import (VOC_CLASSES, DetrDataset, SyntheticSource, build_data, build_loaders, coco_records, collate_eval,
                           collate_train, hflip, normalise_boxes, pad_batch, parse_voc_xml, random_size_crop, resize_to, split_ids,
                           train_augment, voc_records)
from detr_voc.env import setup_environment


@pytest.fixture()
def voc_world(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    ids12 = make_voc_year(paths["data"], "2012", {"trainval": 20}, offset=1)
    make_voc_year(paths["data"], "2007", {"trainval": 6, "test": 8}, offset=2)
    return paths, ids12


def voc_cfg(*extra):
    return load_config("voc", ["data.download=false", "data.num_workers=0", "data.batch_size=4", "data.val_fraction=0.25",
                               "data.train_eval_images=4", "aug.scales=[64,80]", "aug.max_size=120", "aug.crop_resize=[60,70]",
                               "aug.crop_min=30", "aug.crop_max=60", "aug.test_size=64", *extra])


def test_voc_records_use_zero_based_boxes_and_keep_difficult_flags(voc_world):
    paths, ids = voc_world
    a = parse_voc_xml(paths["data"] / "VOCdevkit" / "VOC2012" / "Annotations" / f"{ids['trainval'][0]}.xml")
    assert a["objects"][0]["box"] == [10.0, 5.0, 51.0, 46.0] and a["objects"][1]["difficult"] is True
    recs = voc_records(paths["data"], "2012_trainval")
    assert len(recs) == 20 and recs[0]["labels"].tolist() == [1, 4] and recs[0]["difficult"].tolist() == [False, True]


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"{i:06d}" for i in range(100)]
    tr, va = split_ids(ids, 0.1, 42)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == ids and split_ids(ids, 0.1, 42) == (tr, va)


def test_augmentation_primitives():
    from PIL import Image
    img = Image.new("RGB", (100, 50))
    boxes = np.array([[10, 5, 40, 25], [60, 10, 90, 40]], np.float32)
    f, fb = hflip(img, boxes)
    assert fb[0].tolist() == [60.0, 5.0, 90.0, 25.0] and hflip(f, fb)[1].tolist() == boxes.tolist()
    r, rb = resize_to(img, boxes, 100, 1333)
    assert r.size == (200, 100) and rb[0].tolist() == [20.0, 10.0, 80.0, 50.0]
    torch.manual_seed(0)
    c, cb, cl = random_size_crop(img, boxes, np.array([1, 2]), 20, 30)
    assert 20 <= c.width <= 30 and 20 <= c.height <= 30 and len(cb) == len(cl)
    assert (cb[:, 2] > cb[:, 0]).all() and cb.min() >= 0 and (cb[:, [0, 2]] <= c.width).all() and (cb[:, [1, 3]] <= c.height).all()
    n = normalise_boxes(np.array([[10, 5, 40, 25]], np.float32), 100, 50)
    assert torch.allclose(n[0], torch.tensor([0.25, 0.3, 0.3, 0.4]))


def test_train_augment_keeps_labels_aligned_boxes_valid_and_follows_the_recipe():
    from PIL import Image
    aug = voc_cfg()["aug"]
    img, boxes, labels = Image.new("RGB", (96, 72)), np.array([[10, 5, 51, 46], [41, 21, 91, 66]], np.float32), np.array([1, 4])
    sizes, n_crop_like = set(), 0
    for seed in range(40):
        torch.manual_seed(seed)
        im, b, lb = train_augment(img, boxes, labels, aug)
        assert len(b) == len(lb) and min(im.size) in (64, 80) or max(im.size) <= 120
        if len(b):
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all() and b.min() >= -1e-4
            assert (b[:, [0, 2]] <= im.width + 1e-3).all() and (b[:, [1, 3]] <= im.height + 1e-3).all()
        sizes.add(im.size)
        n_crop_like += len(b) < 2
    assert len(sizes) > 5                                    # scale augmentation really varies the size
    torch.manual_seed(3)
    a, b = train_augment(img, boxes, labels, aug), None
    torch.manual_seed(3)
    b = train_augment(img, boxes, labels, aug)
    assert a[0].size == b[0].size and np.array_equal(a[1], b[1])             # reproducible with the torch seed


def test_training_and_eval_samples(voc_world):
    paths, _ = voc_world
    data = build_data(voc_cfg(), paths)
    torch.manual_seed(0)
    x, t = data["train"][0]
    assert x.dtype == torch.float32 and x.shape[0] == 3 and t["boxes"].shape[1] == 4 and len(t["boxes"]) == len(t["labels"])
    assert t["boxes"].min() >= 0 and t["boxes"].max() <= 1 and (t["labels"] >= 0).all() and (t["labels"] < 20).all()   # 0-based classes
    xv, tv, meta = data["val"][0]
    rec = data["val"].source.records[0]                       # the val split is random: derive the expectation from the record
    assert meta["size"] == (96, 72) and meta["boxes"][0].tolist() == [10.0, 5.0, 51.0, 46.0] and meta["labels"].tolist() == rec["labels"].tolist()
    assert xv.shape[1:] == (64, 85) and torch.allclose(tv["boxes"], normalise_boxes(rec["boxes"], 96, 72), atol=1e-6)
    assert torch.allclose(tv["boxes"][0], torch.tensor([30.5 / 96, 25.5 / 72, 41 / 96, 41 / 72]), atol=1e-5)
    assert meta["difficult"].tolist() == [False, True] and tv["labels"].tolist() == [l - 1 for l in rec["labels"].tolist()]  # noqa: E741


def test_batches_are_padded_with_a_mask_that_marks_the_padding():
    a, b = torch.ones(3, 20, 30), torch.ones(3, 25, 18)
    batch, mask = pad_batch([a, b])
    assert batch.shape == (2, 3, 25, 30) and mask.shape == (2, 25, 30)
    assert not mask[0, :20, :30].any() and mask[0, 20:, :].all() and not mask[1, :25, :18].any() and mask[1, :, 18:].all()
    assert float(batch[0, :, 20:, :].abs().sum()) == 0
    imgs, m, tg = collate_train([(a, {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}), (b, {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})])
    assert imgs.shape == (2, 3, 25, 30) and len(tg) == 2
    assert len(collate_eval([(a, {}, {"x": 1}), (b, {}, {"x": 2})])) == 4


def test_build_data_voc_uses_voc2007_test_and_no_leakage(voc_world):
    paths, _ = voc_world
    data = build_data(voc_cfg(), paths)
    assert len(data["train"]) + len(data["val"]) == 20 and len(data["val"]) == 5 and len(data["test"]) == 8
    assert data["classes"] == list(VOC_CLASSES) and data["data_version"].startswith("voc2012-tr15-va5-te8-")
    ids = lambda ds: {ds.source.records[i]["id"] for i in range(len(ds))}  # noqa: E731
    assert not ids(data["train"]) & ids(data["val"]) and not (ids(data["train"]) | ids(data["val"])) & ids(data["test"])
    ld = build_loaders(voc_cfg(), data, torch.device("cpu"))
    images, mask, targets = next(iter(ld["train"]))
    assert images.shape[0] == 4 and mask.dtype == torch.bool and len(targets) == 4
    xv, mv, tv, metas = next(iter(ld["val"]))
    assert len(metas) == len(tv) == len(xv) and ld["test"] is not None and ld["train_eval"] is not None
    with pytest.raises(FileNotFoundError, match="data.download is false"):
        build_data(voc_cfg(), setup_environment(paths["dest"].parent / "empty"))


def test_coco_records_and_dataset_have_no_test_set(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    make_coco(paths["data"], "val", 12)
    make_coco(paths["data"], "train", 20)
    recs, names = coco_records(paths["data"], "val")
    assert names == ["cat1", "cat3", "cat7", "cat90"] and len(recs) == 9 and max(int(r["labels"].max()) for r in recs) == 4
    cfg = load_config("coco", ["data.download=false", "data.num_workers=0", "data.batch_size=4", "data.coco_val_images=5",
                               "data.train_eval_images=4", "aug.scales=[64]", "aug.max_size=100", "aug.crop_resize=[60]", "aug.crop_min=30",
                               "aug.crop_max=50", "aug.test_size=64"])
    data = build_data(cfg, paths)
    assert data["test"] is None and len(data["val"]) == 5 and len(data["classes"]) == 4 and cfg["data"]["num_foreground"] == 80


def test_synthetic_source_and_images_without_objects(tmp_path):
    a, b = SyntheticSource(5, 3, 64, 7).get(2), SyntheticSource(5, 3, 64, 7).get(2)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(a[1], b[1])
    src = SyntheticSource(3, 3, 64, 1)
    ds = DetrDataset(src, load_config("voc", SMOKE)["aug"], True, include_difficult=False)
    x, t = ds[0]
    assert x.shape[0] == 3 and len(t["boxes"]) == len(t["labels"])
