"""Tiny fake Pascal VOC trees (2012 trainval, 2007 trainval + test) and a fake COCO val set in the real on-disk formats."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from detr_voc.data import VOC_CLASSES

XML = """<annotation><filename>{id}.jpg</filename><size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objs}</annotation>"""
OBJ = """<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>{d}</difficult>
<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"""


def make_voc_year(root, year, splits: dict[str, int], w=96, h=72, offset=0):
    base = Path(root) / "VOCdevkit" / f"VOC{year}"
    for sub in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(offset)
    ids, k = {}, 0
    for split, n in splits.items():
        lst = [f"{year}_{k + i:06d}" for i in range(n)]
        k += n
        ids[split] = lst
        (base / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(lst) + "\n")
        for j, iid in enumerate(lst):
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(base / "JPEGImages" / f"{iid}.jpg")
            objs = OBJ.format(name=VOC_CLASSES[j % 20], d=0, x1=11, y1=6, x2=51, y2=46)
            objs += OBJ.format(name=VOC_CLASSES[(j + 3) % 20], d=1, x1=41, y1=21, x2=91, y2=66)
            (base / "Annotations" / f"{iid}.xml").write_text(XML.format(id=iid, w=w, h=h, objs=objs))
    return ids


def make_coco(root, split="val", n=12, w=80, h=60):
    base = Path(root) / "coco2017"
    (base / "annotations").mkdir(parents=True, exist_ok=True)
    (base / f"{split}2017").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    cats = [{"id": i, "name": f"cat{i}"} for i in (1, 3, 7, 90)]       # sparse ids, like COCO
    images, anns, k = [], [], 1
    for i in range(1, n + 1):
        fn = f"{i:012d}.jpg"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(base / f"{split}2017" / fn)
        images.append({"id": i, "file_name": fn, "width": w, "height": h})
        if i % 4 == 0:
            continue                                                     # every 4th image has no annotations
        anns.append({"id": k, "image_id": i, "category_id": cats[i % 4]["id"], "bbox": [10, 5, 30, 40], "iscrowd": 0}); k += 1
        anns.append({"id": k, "image_id": i, "category_id": 3, "bbox": [40, 20, 0.5, 30], "iscrowd": 0}); k += 1   # degenerate
        if i % 3 == 0:
            anns.append({"id": k, "image_id": i, "category_id": 1, "bbox": [0, 0, 60, 50], "iscrowd": 1}); k += 1
    (base / "annotations" / f"instances_{split}2017.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    (base / f".complete_{split}2017").write_text("{}")
    return images, anns
