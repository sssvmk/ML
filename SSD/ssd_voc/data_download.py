"""Robust, resumable downloads of Pascal VOC and COCO into <dest>/data. Staging happens under <dest>/tmp only."""
import json
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

log = logging.getLogger("ssd_voc.download")

COCO_BASE = "http://images.cocodataset.org"
URLS = {
    "voc2012_trainval": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "voc2007_trainval": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "voc2007_test": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    "coco_val_images": f"{COCO_BASE}/zips/val2017.zip",
    "coco_train_images": f"{COCO_BASE}/zips/train2017.zip",
    "coco_annotations": f"{COCO_BASE}/annotations/annotations_trainval2017.zip",
}
VOC_SETS = {"2012_trainval": ("voc2012_trainval", "VOC2012", "trainval"),
            "2007_trainval": ("voc2007_trainval", "VOC2007", "trainval"),
            "2007_test": ("voc2007_test", "VOC2007", "test")}


def get_url(key: str) -> str:
    """Override with DATASET_URL_<KEY> (e.g. an internal mirror)."""
    return os.environ.get(f"DATASET_URL_{key.upper()}", URLS[key])


def _human(n: float) -> str:
    return f"{n / 1e6:,.0f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def ensure_space(path: Path, needed_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    if free < needed_gb:
        raise OSError(f"only {free:.1f} GB free under {path}, need about {needed_gb:.1f} GB")


def download(url: str, out, *, retries: int = 6, chunk: int = 1 << 20, timeout: int = 60,
             session: requests.Session | None = None, log_every: float = 15.0) -> Path:
    """Resumable download with retries; a finished file of the expected size is not downloaded again."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    total = None
    try:
        h = s.head(url, allow_redirects=True, timeout=timeout)
        if h.ok and "Content-Length" in h.headers:
            total = int(h.headers["Content-Length"])
    except requests.RequestException:
        pass
    if out.exists() and out.stat().st_size > 0 and (total is None or out.stat().st_size == total):
        return out
    part = out.with_name(out.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with s.get(url, stream=True, timeout=timeout, allow_redirects=True,
                       headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:
                    if total is not None and have == total:
                        break
                    part.unlink(missing_ok=True)
                    continue
                r.raise_for_status()
                resumed = bool(have) and r.status_code == 206
                if total is None:
                    cr = r.headers.get("Content-Range")
                    total = int(cr.rsplit("/", 1)[1]) if cr and "/" in cr else (
                        int(r.headers["Content-Length"]) + (have if resumed else 0) if "Content-Length" in r.headers
                        else None)
                done = have if resumed else 0
                last = time.time()
                with open(part, "ab" if resumed else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        done += len(block)
                        if time.time() - last >= log_every:
                            last = time.time()
                            log.info("  %s: %s%s", out.name, _human(done), f" ({100 * done / total:.0f}%)" if total else "")
            if total is not None and part.stat().st_size != total:
                raise OSError(f"incomplete download: {part.stat().st_size} of {total} bytes")
            break
        except (requests.RequestException, OSError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("download attempt %d/%d failed (%s); retrying in %ds", attempt, retries, e, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not download {url} after {retries} attempts: {last_err}. If the host is blocked from "
                           "this workspace, allow it, or set DATASET_URL_<KEY> to an internal mirror.")
    part.replace(out)
    log.info("downloaded %s (%s)", out.name, _human(out.stat().st_size))
    return out


def _inside(base: Path, target: Path) -> bool:
    return str(target.resolve()).startswith(str(base.resolve()) + os.sep)


def extract_tar(tpath, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tpath) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(out_dir, filter="data")
        else:
            for m in t.getmembers():
                if not _inside(out_dir, out_dir / m.name):
                    raise ValueError(f"unsafe path in archive: {m.name}")
            t.extractall(out_dir)


def extract_zip(zpath, out_dir, members: list[str] | None = None) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = members if members is not None else z.namelist()
        have = set(z.namelist())
        missing = [m for m in names if m not in have]
        if missing:
            raise KeyError(f"{missing[0]} not found in {Path(zpath).name}")
        for n in names:
            if not _inside(out_dir, out_dir / n):
                raise ValueError(f"unsafe path in archive: {n}")
        z.extractall(out_dir, names)
    return names


def voc_root(data_dir, year: str) -> Path:
    return Path(data_dir) / "VOCdevkit" / f"VOC{year}"


def voc_available(data_dir, key: str) -> bool:
    _, folder, split = VOC_SETS[key]
    root = Path(data_dir) / "VOCdevkit" / folder
    return (root / "Annotations").is_dir() and (root / "JPEGImages").is_dir() and \
        (root / "ImageSets" / "Main" / f"{split}.txt").exists()


def ensure_voc(data_dir, tmp_dir, keys, keep_archives=False, allow_download=True) -> dict:
    """Make sure the requested VOC sets exist under data_dir (download + extract if needed)."""
    done = {}
    for key in keys:
        if key not in VOC_SETS:
            raise ValueError(f"unknown VOC set {key!r}; choose from {sorted(VOC_SETS)}")
        if voc_available(data_dir, key):
            done[key] = "present"
            continue
        if not allow_download:
            raise FileNotFoundError(f"VOC set {key} not found under {data_dir} and data.download is false")
        url_key = VOC_SETS[key][0]
        ensure_space(Path(tmp_dir), 4.0)
        arc = download(get_url(url_key), Path(tmp_dir) / Path(get_url(url_key)).name)
        extract_tar(arc, data_dir)
        if not keep_archives:
            arc.unlink(missing_ok=True)
        if not voc_available(data_dir, key):
            raise RuntimeError(f"{key} was extracted but is incomplete under {data_dir}")
        done[key] = "downloaded"
    return done


def coco_root(data_dir) -> Path:
    return Path(data_dir) / "coco2017"


def ensure_coco(data_dir, tmp_dir, splits=("val",), keep_archives=False, allow_download=True) -> dict:
    """COCO 2017: annotations (instances only) plus val2017 and/or train2017 images (train is ~18 GB)."""
    root = coco_root(data_dir)
    done = {}
    for split in splits:
        ann = root / "annotations" / f"instances_{split}2017.json"
        imgs = root / f"{split}2017"
        marker = root / f".complete_{split}2017"
        if ann.exists() and marker.exists():
            done[split] = "present"
            continue
        if not allow_download:
            raise FileNotFoundError(f"COCO {split}2017 not found under {root} and data.download is false")
        ensure_space(Path(tmp_dir), 22.0 if split == "train" else 3.0)
        if not ann.exists():
            arc = download(get_url("coco_annotations"), Path(tmp_dir) / "annotations_trainval2017.zip")
            extract_zip(arc, root, [f"annotations/instances_{split}2017.json"])
            if not keep_archives:
                arc.unlink(missing_ok=True)
        arc = download(get_url(f"coco_{split}_images"), Path(tmp_dir) / f"{split}2017.zip", timeout=120)
        extract_zip(arc, root)
        if not keep_archives:
            arc.unlink(missing_ok=True)
        n = sum(1 for _ in imgs.glob("*.jpg"))
        marker.write_text(json.dumps({"images": n, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        done[split] = f"downloaded ({n} images)"
    return done
