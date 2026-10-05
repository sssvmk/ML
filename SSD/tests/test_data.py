import pickle

import numpy as np
import pytest
import torch
from smoke_over import SMOKE
from voc_fixture import make_coco, make_voc_year

from ssd_voc.config import load_config
from ssd_voc.data import (
    VOC_CLASSES,
    DetectionDataset,
    SyntheticSource,
    build_data,
    build_loaders,
    coco_records,
    parse_voc_xml,
    split_ids,
    voc_records,
)
from ssd_voc.env import setup_environment


@pytest.fixture()
def voc_world(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    ids12 = make_voc_year(paths["data"], "2012", {"trainval": 20}, offset=1)
    ids07 = make_voc_year(paths["data"], "2007", {"trainval": 6, "test": 8}, offset=2)
    return paths, ids12, ids07


def voc_cfg(*extra):
    return load_config("voc", ["data.download=false", "data.num_workers=0", "data.batch_size=4", "model.input_size=96",
                               "data.val_fraction=0.25", "data.train_eval_images=4", *extra])


def test_voc_xml_is_converted_from_one_based_pixel_boxes(voc_world):
    paths, ids12, _ = voc_world
    a = parse_voc_xml(paths["data"] / "VOCdevkit" / "VOC2012" / "Annotations" / f"{ids12['trainval'][0]}.xml")
    assert (a["width"], a["height"]) == (96, 72) and a["objects"][0]["box"] == [10.0, 5.0, 51.0, 46.0]
    assert a["objects"][0]["difficult"] is False and a["objects"][1]["difficult"] is True


def test_voc_records_labels_difficult_and_index_cache(voc_world):
    paths, ids12, _ = voc_world
    recs = voc_records(paths["data"], "2012_trainval")
    assert len(recs) == 20 and recs[0]["labels"].tolist() == [1, 4] and recs[0]["difficult"].tolist() == [False, True]
    assert recs[0]["boxes"].dtype == np.float32 and recs[1]["id"].startswith("2012:")
    caches = list(paths["data"].glob("index_voc2012_*.pkl"))
    assert len(caches) == 1 and pickle.loads(caches[0].read_bytes())[3]["id"] == recs[3]["id"]


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"{i:06d}" for i in range(100)]
    tr, va = split_ids(ids, 0.1, 42)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == ids
    assert split_ids(ids, 0.1, 42) == (tr, va) and split_ids(ids, 0.1, 7)[1] != va and split_ids(ids, 0.0, 1) == (ids, [])


def test_build_data_voc2012_uses_voc2007_test_as_the_untouched_test_set(voc_world):
    paths, ids12, ids07 = voc_world
    data = build_data(voc_cfg(), paths)
    assert len(data["train"]) + len(data["val"]) == 20 and len(data["val"]) == 5 and len(data["test"]) == 8
    assert data["classes"] == list(VOC_CLASSES) and data["data_version"].startswith("voc2012-tr15-va5-te8-")
    train_ids = {data["train"].source.records[i]["id"] for i in range(len(data["train"]))}
    val_ids = {data["val"].source.records[i]["id"] for i in range(len(data["val"]))}
    test_ids = {data["test"].source.records[i]["id"] for i in range(len(data["test"]))}
    assert not train_ids & val_ids and not (train_ids | val_ids) & test_ids      # no leakage into the test set
    plus = build_data(voc_cfg('data.voc_train_sets=["2012_trainval","2007_trainval"]'), paths)   # paper's 07+12 style
    assert len(plus["train"]) + len(plus["val"]) == 26 and len(plus["test"]) == 8
    with pytest.raises(FileNotFoundError, match="data.download is false"):
        build_data(voc_cfg("data.test_set=null", 'data.voc_train_sets=["2012_trainval"]'), setup_environment(paths["dest"].parent / "empty") | {"data": paths["dest"].parent / "nothing"})


@pytest.mark.parametrize("aug", [
    {"photometric": False, "crop": False, "flip": False},
    {"photometric": True, "crop": True, "flip": True, "expansion": False},
    {"photometric": True, "crop": True, "flip": True, "expansion": True, "expansion_max_side": 4.0},
])
def test_training_samples_have_valid_normalised_boxes_and_labels(voc_world, aug):
    paths, *_ = voc_world
    cfg = voc_cfg()
    cfg["aug"].update(aug)
    ds = build_data(cfg, paths)["train"]
    torch.manual_seed(0)
    for _ in range(3):
        for i in range(len(ds)):
            x, t = ds[i]
            b = t["boxes"]
            assert x.shape == (3, 96, 96) and x.dtype == torch.float32 and len(b) == len(t["labels"])
            if len(b):
                assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all() and b.min() >= 0 and b.max() <= 1
                assert t["labels"].min() >= 1 and t["labels"].max() <= 20


def test_expansion_shrinks_objects_on_average(voc_world):
    paths, *_ = voc_world
    areas = {}
    for expand in (False, True):
        cfg = voc_cfg()
        cfg["aug"].update({"photometric": False, "crop": False, "flip": False, "expansion": expand})
        ds = build_data(cfg, paths)["train"]
        torch.manual_seed(3)
        a = [float(((t["boxes"][:, 2] - t["boxes"][:, 0]) * (t["boxes"][:, 3] - t["boxes"][:, 1])).mean())
             for _ in range(8) for x, t in (ds[i] for i in range(len(ds))) if len(t["boxes"])]
        areas[expand] = float(np.mean(a))
    assert areas[True] < areas[False]               # zoom-out creates small objects (paper sec. 3.6)


def test_eval_samples_keep_original_pixel_boxes_for_the_metric(voc_world):
    paths, *_ = voc_world
    x, t, meta = build_data(voc_cfg(), paths)["val"][0]
    assert x.shape == (3, 96, 96) and meta["size"] == (96, 72) and meta["difficult"].tolist() == [False, True]
    assert meta["boxes"][0].tolist() == [10.0, 5.0, 51.0, 46.0] and float(t["boxes"].max()) <= 1.0
    assert torch.allclose(t["boxes"][0], torch.tensor([10 / 96, 5 / 72, 51 / 96, 46 / 72]), atol=1e-6)


def test_images_without_usable_objects_are_handled(voc_world):
    paths, *_ = voc_world
    cfg = voc_cfg()
    data = build_data(cfg, paths)
    src = data["train"].source
    ds = DetectionDataset(src, 96, True, cfg["aug"], include_difficult=False)
    for rec in src.records:
        rec["difficult"][:] = True                   # every object is 'difficult' -> nothing left to train on
    x, t = ds[0]
    assert x.shape == (3, 96, 96) and t["boxes"].shape == (0, 4) and t["labels"].shape == (0,)


def test_coco_records_map_labels_flag_crowd_and_drop_empty_images(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    make_coco(paths["data"], "val", 12)
    recs, names = coco_records(paths["data"], "val")
    assert names == ["cat1", "cat3", "cat7", "cat90"] and len(recs) == 9              # 3 images without annotations dropped
    r = next(r for r in recs if r["id"] == "coco:3")
    assert r["labels"].tolist()[0] in (1, 2, 3, 4) and r["difficult"].tolist() == [False, True]   # crowd -> ignored
    assert all((r["boxes"][:, 2] > r["boxes"][:, 0]).all() for r in recs)             # the 0.5 px wide box was dropped
    assert max(int(l.max()) for l in (r["labels"] for r in recs)) == 4              # contiguous 1..4, not 90
    assert (paths["data"] / "index_coco_val2017.pkl").exists()


def test_build_data_coco_has_no_test_set(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    make_coco(paths["data"], "val", 12)
    make_coco(paths["data"], "train", 20)
    cfg = load_config("coco", ["data.download=false", "data.num_workers=0", "data.batch_size=4", "model.input_size=96",
                               "data.coco_val_images=5", "data.train_eval_images=4"])
    data = build_data(cfg, paths)
    assert data["test"] is None and len(data["val"]) == 5 and len(data["train"]) == 15 and len(data["classes"]) == 4
    assert load_config("coco")["data"]["num_foreground"] == 80
    x, t = data["train"][0]
    assert t["labels"].max() <= 4


def test_synthetic_source_is_deterministic_and_loaders_collate(tmp_path):
    a, b = SyntheticSource(5, 3, 64, 7).get(2), SyntheticSource(5, 3, 64, 7).get(2)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(a[1], b[1])
    cfg = load_config("voc", SMOKE)
    paths = setup_environment(tmp_path / "d")
    data = build_data(cfg, paths)
    ld = build_loaders(cfg, data, torch.device("cpu"))
    x, tg = next(iter(ld["train"]))
    assert x.shape == (4, 3, 128, 128) and len(tg) == 4
    xv, tv, mv = next(iter(ld["val"]))
    assert len(mv) == len(tv) == len(xv) and ld["test"] is not None and ld["train_eval"] is not None
