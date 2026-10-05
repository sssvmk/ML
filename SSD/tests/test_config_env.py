import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from ssd_voc.config import STAGES, load_config, parse_overrides, registered_name
from ssd_voc.env import make_paths, setup_environment


def test_every_stage_loads_with_the_paper_recipe():
    for stage in STAGES:
        cfg = load_config(stage, dest="/tmp/x")
        assert cfg["loss"]["neg_pos_ratio"] == 3 and cfg["loss"]["iou_threshold"] == 0.5 and cfg["loss"]["alpha"] == 1.0
        assert cfg["schedule"]["momentum"] == 0.9 and cfg["schedule"]["weight_decay"] == 0.0005
        assert cfg["data"]["batch_size"] == 32 and cfg["model"]["dropout"] == 0.0        # the paper removes dropout
        assert cfg["postprocess"]["score_thresh"] == 0.01 and cfg["postprocess"]["nms_iou"] == 0.45
        assert cfg["postprocess"]["max_detections"] == 200 and cfg["stage"] == stage


def test_stage_overrides_match_the_paper_schedules():
    voc, coco = load_config("voc"), load_config("coco")
    assert [(p["lr"], p["iters"]) for p in voc["schedule"]["phases"]] == [(0.001, 60000), (0.0001, 20000)]
    assert [(p["lr"], p["iters"]) for p in coco["schedule"]["phases"]] == [(0.001, 160000), (0.0001, 40000), (0.00001, 40000)]
    assert coco["data"]["dataset"] == "coco2017" and coco["data"]["num_foreground"] == 80 and coco["data"]["test_set"] is None
    assert coco["model"]["priors"] == {"first_scale": 0.07, "min_scale": 0.15, "max_scale": 0.9}   # paper sec. 3.4
    assert voc["model"]["priors"]["min_scale"] == 0.2 and voc["data"]["num_foreground"] == 20
    for long_stage in ("voc_long", "coco_long", "voc_from_coco_long"):
        c = load_config(long_stage)
        assert c["aug"]["expansion"] is True and c["schedule"]["scale"] == 2.0
    assert load_config("voc")["aug"]["expansion"] is False
    assert load_config("voc_from_coco")["init_from"] == "auto:coco"
    assert load_config("voc_from_coco_long")["init_from"] == "auto:coco_long"


def test_user_overrides_win_and_are_validated():
    cfg = load_config("voc", ["schedule.eval_every=250", 'schedule.phases=[{"lr":0.5,"iters":3}]', "model.backbone=resnet18"])
    assert cfg["schedule"]["eval_every"] == 250 and cfg["schedule"]["phases"] == [{"lr": 0.5, "iters": 3}]
    assert parse_overrides(["a.b=1", "c=[1,2]"]) == {"a.b": 1, "c": [1, 2]}
    with pytest.raises(ValueError, match="unknown stage"):
        load_config("nope")
    with pytest.raises(ValueError, match="a.b=value"):
        load_config("voc", ["broken"])
    with pytest.raises(ValueError, match="boxes_per_loc"):
        load_config("voc", ["model.boxes_per_loc=[4,6]"])
    with pytest.raises(ValueError, match="unknown dataset"):
        load_config("voc", ["data.dataset=imagenet"])


def test_registered_name():
    assert registered_name(load_config("voc")) == "ssd-resnet50-voc2012"
    assert registered_name(load_config("voc", ["mlflow.registered_model_name=main.default.ssd"])) == "main.default.ssd"


def test_every_cache_and_temp_location_is_inside_dest(tmp_path):
    dest = tmp_path / "run"
    paths = setup_environment(dest)
    for key in ("TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "TMPDIR", "TEMP", "TMP", "TRITON_CACHE_DIR",
                "HF_HOME"):
        assert Path(os.environ[key]).resolve().is_relative_to(dest.resolve()), key
    assert Path(tempfile.gettempdir()).resolve() == (dest / "tmp").resolve()
    assert Path(torch.hub.get_dir()).resolve().is_relative_to(dest.resolve())
    assert Path(sys.pycache_prefix).resolve().is_relative_to(dest.resolve())
    for k in ("data", "runs", "checkpoints", "tmp", "mlflow", "mlflow_artifacts", "torch_home"):
        assert paths[k].is_dir() and paths[k].resolve().is_relative_to(dest.resolve()), k
    with tempfile.NamedTemporaryFile() as f:  # anything created through tempfile lands in <dest>/tmp
        assert Path(f.name).resolve().is_relative_to((dest / "tmp").resolve())


def test_volume_destination_must_exist():
    with pytest.raises(ValueError, match="Volume path"):
        setup_environment("/Volumes/main/default")
    with pytest.raises(FileNotFoundError, match="CREATE VOLUME"):
        setup_environment("/Volumes/no_such_catalog/s/v/ssd")
    assert make_paths("dbfs:/FileStore/x")["dest"] == Path("/FileStore/x")
