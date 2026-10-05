"""Tiny fake Pascal VOC 2007 tree in the real on-disk format."""
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import VOC_CLASSES

XML = """<annotation><filename>{id}.jpg</filename><size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objs}</annotation>"""
OBJ = """<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>{d}</difficult>
<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"""


def make_fake_voc(root, n_trainval=20, n_test=8, w=80, h=60):
    base = Path(root) / "VOCdevkit" / "VOC2007"
    for sub in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = {"trainval": [f"{i:06d}" for i in range(n_trainval)],
           "test": [f"{n_trainval + i:06d}" for i in range(n_test)]}
    for split, lst in ids.items():
        (base / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(lst) + "\n")
        for k, iid in enumerate(lst):
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(base / "JPEGImages" / f"{iid}.jpg")
            objs = OBJ.format(name=VOC_CLASSES[k % 20], d=0, x1=11, y1=6, x2=51, y2=46)           # 1-based box
            objs += OBJ.format(name=VOC_CLASSES[(k + 3) % 20], d=1, x1=41, y1=21, x2=71, y2=56)   # difficult object
            (base / "Annotations" / f"{iid}.xml").write_text(XML.format(id=iid, w=w, h=h, objs=objs))
    return ids
