"""Command line interface (the notebook calls the same functions)."""
import argparse
import json
import logging
import sys

from ssd_voc.config import STAGES, load_config
from ssd_voc.env import setup_environment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SSD (ResNet) on VOC2012 / COCO. Everything lives under --dest.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-stages")
    for name in ("prepare-data", "benchmark", "train", "test", "register"):
        s = sub.add_parser(name)
        s.add_argument("--dest", required=True, help="destination folder: data, caches, runs, checkpoints, MLflow")
        s.add_argument("--stage", default="voc", choices=sorted(STAGES))
        s.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override config values")
        if name in ("test", "register"):
            s.add_argument("--run-id", required=True)
        if name == "test":
            s.add_argument("--force", action="store_true")
        if name == "register":
            s.add_argument("--model-name", default=None)
    s = sub.add_parser("predict")
    s.add_argument("--dest", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", nargs="+", required=True)
    s.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if a.cmd == "list-stages":
        for k, (desc, _) in STAGES.items():
            print(f"{k:20s} {desc}")
        return 0
    paths = setup_environment(a.dest)
    if a.cmd == "predict":
        from ssd_voc.serve import Predictor
        print(json.dumps(Predictor(a.checkpoint).predict(a.image, a.threshold), indent=2))
        return 0
    cfg = load_config(a.stage, a.set, dest=a.dest)
    if a.cmd == "prepare-data":
        from ssd_voc.data import build_data
        data = build_data(cfg, paths)
        print(json.dumps({"data_version": data["data_version"], "classes": len(data["classes"]),
                          "train": len(data["train"]), "val": len(data["val"] or []), "test": len(data["test"] or [])}))
    elif a.cmd == "benchmark":
        from ssd_voc.train import benchmark
        print(json.dumps(benchmark(cfg, paths), indent=2))
    elif a.cmd == "train":
        from ssd_voc.train import run_training
        run_training(cfg, paths)
    elif a.cmd == "test":
        from ssd_voc.evaluate import run_test_evaluation
        run_test_evaluation(cfg, paths, a.run_id, a.force)
    elif a.cmd == "register":
        from ssd_voc.data import build_data
        from ssd_voc.serve import register_candidate
        data = build_data(cfg, paths)
        ds = data["val"] or data["test"]
        register_candidate(cfg, paths, a.run_id, ds.source.get(0)[0], a.model_name)
    return 0
