import pytest
import torch
from PIL import Image

from src.data import (
    VOC_CLASSES,
    VOC2007Detection,
    build_datasets,
    build_loaders,
    build_transforms,
    collate,
    ensure_voc2007,
    parse_voc_xml,
    read_ids,
    split_ids,
    voc_dir,
)
from src.utils import ROOT, load_config
from tests.voc_fixture import make_fake_voc


@pytest.fixture()
def voc(tmp_path):
    ids = make_fake_voc(tmp_path)
    return tmp_path, ids


def test_parse_voc_xml_converts_to_zero_based_boxes(voc):
    root, ids = voc
    a = parse_voc_xml(voc_dir(root) / "Annotations" / f"{ids['trainval'][0]}.xml")
    assert a["width"] == 80 and a["height"] == 60 and len(a["objects"]) == 2
    assert a["objects"][0]["box"] == [10.0, 5.0, 51.0, 46.0] and a["objects"][1]["difficult"] is True


def test_dataset_item_labels_and_difficult_filter(voc):
    root, ids = voc
    ds = VOC2007Detection(root, ids["trainval"], build_transforms(False), include_difficult=True, return_difficult=True)
    img, t = ds[1]
    assert img.shape == (3, 60, 80) and img.dtype == torch.float32 and 0 <= img.min() and img.max() <= 1
    assert t["labels"].tolist() == [VOC_CLASSES.index(VOC_CLASSES[1]) + 1, VOC_CLASSES.index(VOC_CLASSES[4]) + 1]
    assert t["difficult"].tolist() == [False, True]
    no_diff = VOC2007Detection(root, ids["trainval"], build_transforms(False), include_difficult=False)
    assert len(no_diff[1][1]["labels"]) == 1


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"{i:06d}" for i in range(100)]
    tr, va = split_ids(ids, 0.1, 42)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == ids
    assert split_ids(ids, 0.1, 42) == (tr, va) and split_ids(ids, 0.1, 7)[1] != va
    assert split_ids(ids, 0.0, 42) == (ids, [])


def test_build_datasets_from_fake_voc_and_loaders(voc):
    root, ids = voc
    cfg = load_config(ROOT / "config" / "train.yaml", [f"data.root={root}", "data.download=false",
                                                       "data.num_workers=0", "data.val_fraction=0.25"])
    assert cfg["data"]["num_classes"] == 20
    ds = build_datasets(cfg)
    assert len(ds["train"]) + len(ds["val"]) == 20 and len(ds["test"]) == 8
    assert ds["classes"] == list(VOC_CLASSES) and ds["data_version"].startswith("voc2007-tr15-va5-te8-")
    ld = build_loaders(cfg, ds, torch.device("cpu"))
    images, targets = next(iter(ld["train"]))
    assert len(images) == 4 and not hasattr(targets[0]["boxes"], "canvas_size")  # plain tensors after collate
    assert ld["val"] is not None and ld["train_eval"] is not None
    cfg0 = load_config(ROOT / "config" / "train.yaml", [f"data.root={root}", "data.download=false",
                                                        "data.num_workers=0", "data.val_fraction=0"])
    assert build_loaders(cfg0, build_datasets(cfg0), torch.device("cpu"))["val"] is None


def test_missing_dataset_error_names_the_download_urls(tmp_path):
    with pytest.raises(FileNotFoundError, match="VOCtrainval_06-Nov-2007.tar"):
        ensure_voc2007(tmp_path, download=False)


@pytest.mark.parametrize("aug", ["none", "flip", "standard", "strong"])
def test_augmentation_keeps_boxes_valid_and_labels_aligned(voc, aug):
    root, ids = voc
    ds = VOC2007Detection(root, ids["trainval"], build_transforms(True, aug))
    torch.manual_seed(0)
    for i in range(len(ds)):
        img, t = ds[i]
        b, h, w = t["boxes"].as_subclass(torch.Tensor), img.shape[-2], img.shape[-1]
        assert len(b) == len(t["labels"])
        if len(b):
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            assert b.min() >= -1e-3 and b[:, [0, 2]].max() <= w + 1e-3 and b[:, [1, 3]].max() <= h + 1e-3


def test_unknown_aug_rejected_and_read_ids(voc):
    with pytest.raises(ValueError):
        build_transforms(True, "cutmix")
    root, ids = voc
    assert read_ids(root, "test") == ids["test"]
    img = Image.open(voc_dir(root) / "JPEGImages" / f"{ids['test'][0]}.jpg")
    assert collate([(img, {"labels": torch.tensor([1])})])[1][0]["labels"].tolist() == [1]
