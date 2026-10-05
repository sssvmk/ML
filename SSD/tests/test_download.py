import io
import json
import tarfile
import zipfile
from collections import namedtuple

import pytest
from PIL import Image
from server import FakeHost

from ssd_voc import data_download as dd
from ssd_voc.env import setup_environment

Usage = namedtuple("Usage", "total used free")


def jpeg(seed=0) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (seed * 40 % 255, 90, 160)).save(buf, "JPEG")
    return buf.getvalue()


def make_tar(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for k, v in entries.items():
            info = tarfile.TarInfo(k)
            info.size = len(v)
            t.addfile(info, io.BytesIO(v))
    return buf.getvalue()


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in entries.items():
            z.writestr(k, v)
    return buf.getvalue()


VOC_TAR = make_tar({"VOCdevkit/VOC2012/ImageSets/Main/trainval.txt": b"a\nb\n", "VOCdevkit/VOC2012/Annotations/a.xml": b"<x/>",
                    "VOCdevkit/VOC2012/JPEGImages/a.jpg": jpeg(1), "VOCdevkit/VOC2012/JPEGImages/b.jpg": jpeg(2)})


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda p: Usage(10**12, 0, 10**12))


def test_voc_download_extracts_into_dest_and_cleans_staging(tmp_path, monkeypatch):
    paths = setup_environment(tmp_path / "dest")
    with FakeHost({"/voc2012.tar": VOC_TAR}) as h:
        monkeypatch.setenv("DATASET_URL_VOC2012_TRAINVAL", f"{h.base}/voc2012.tar")
        assert dd.ensure_voc(paths["data"], paths["tmp"], ["2012_trainval"]) == {"2012_trainval": "downloaded"}
        gets = h.stats["GET"]
        assert dd.ensure_voc(paths["data"], paths["tmp"], ["2012_trainval"]) == {"2012_trainval": "present"}
        assert h.stats["GET"] == gets                                          # nothing downloaded twice
    assert dd.voc_available(paths["data"], "2012_trainval")
    assert not list(paths["tmp"].glob("*.tar")) and not list(paths["tmp"].glob("*.part"))   # archive removed from staging
    assert (paths["data"] / "VOCdevkit" / "VOC2012" / "JPEGImages" / "a.jpg").exists()


def test_keep_archives_and_download_switch(tmp_path, monkeypatch):
    paths = setup_environment(tmp_path / "dest")
    with pytest.raises(FileNotFoundError, match="data.download is false"):
        dd.ensure_voc(paths["data"], paths["tmp"], ["2012_trainval"], allow_download=False)
    with pytest.raises(ValueError, match="unknown VOC set"):
        dd.ensure_voc(paths["data"], paths["tmp"], ["2010_test"])
    with FakeHost({"/v.tar": VOC_TAR}) as h:
        monkeypatch.setenv("DATASET_URL_VOC2012_TRAINVAL", f"{h.base}/v.tar")
        dd.ensure_voc(paths["data"], paths["tmp"], ["2012_trainval"], keep_archives=True)
    assert list(paths["tmp"].glob("*.tar"))          # kept, but still inside <dest>/tmp


def test_download_resumes_after_a_dropped_connection(tmp_path):
    payload = bytes(range(256)) * 4000
    with FakeHost({"/big.bin": payload}, flaky_once=("/big.bin",)) as h:
        out = dd.download(f"{h.base}/big.bin", tmp_path / "big.bin", chunk=4096)
        assert out.read_bytes() == payload and not (tmp_path / "big.bin.part").exists() and h.stats["RANGE"] >= 1


def test_download_failure_message_is_actionable(tmp_path):
    with pytest.raises(RuntimeError, match="DATASET_URL_"):
        dd.download(f"{FakeHost({}).base}/nothing", tmp_path / "x", retries=2, timeout=1)


def test_unsafe_archives_are_rejected(tmp_path):
    evil_zip = tmp_path / "e.zip"
    evil_zip.write_bytes(make_zip({"../escape.txt": b"x"}))
    with pytest.raises(ValueError, match="unsafe"):
        dd.extract_zip(evil_zip, tmp_path / "o")
    with pytest.raises(KeyError):
        dd.extract_zip(evil_zip, tmp_path / "o2", ["missing.txt"])


def test_coco_val_downloads_only_the_instances_file_and_images(tmp_path, monkeypatch):
    paths = setup_environment(tmp_path / "dest")
    files = {"/val.zip": make_zip({f"val2017/{i:012d}.jpg": jpeg(i) for i in range(1, 4)}),
             "/ann.zip": make_zip({"annotations/instances_val2017.json": json.dumps({"images": []}).encode(),
                                   "annotations/captions_val2017.json": b"{}"})}
    with FakeHost(files) as h:
        monkeypatch.setenv("DATASET_URL_COCO_VAL_IMAGES", f"{h.base}/val.zip")
        monkeypatch.setenv("DATASET_URL_COCO_ANNOTATIONS", f"{h.base}/ann.zip")
        assert dd.ensure_coco(paths["data"], paths["tmp"], ("val",))["val"].startswith("downloaded (3 images")
        gets = h.stats["GET"]
        assert dd.ensure_coco(paths["data"], paths["tmp"], ("val",)) == {"val": "present"} and h.stats["GET"] == gets
    root = paths["data"] / "coco2017"
    assert (root / "annotations" / "instances_val2017.json").exists() and not (root / "annotations" / "captions_val2017.json").exists()
    assert not list(paths["tmp"].glob("*.zip"))
    with pytest.raises(FileNotFoundError):
        dd.ensure_coco(paths["data"], paths["tmp"], ("train",), allow_download=False)


def test_disk_space_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda p: Usage(10**10, 0, 3 * 10**9))
    with pytest.raises(OSError, match="free"):
        dd.ensure_space(tmp_path, 22.0)
    dd.ensure_space(tmp_path, 1.0)
