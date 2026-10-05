"""Command line interface for single steps (the pipeline script runs them all in order)."""
import argparse
import json
import logging
import sys

from detr_voc.config import STAGES, load_config
from detr_voc.env import setup_environment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DETR on VOC2012 / COCO. Everything lives under --dest.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-stages")
    for name in ("prepare-data", "benchmark", "train", "test", "package"):
        s = sub.add_parser(name)
        s.add_argument("--dest", required=True)
        s.add_argument("--stage", default="voc", choices=sorted(STAGES))
        s.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
        if name in ("test", "package"):
            s.add_argument("--run-id", required=True)
        if name == "test":
            s.add_argument("--force", action="store_true")
    s = sub.add_parser("predict")
    s.add_argument("--dest", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", nargs="+", required=True)
    s.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if a.cmd == "list-stages":
        for k, (desc, _) in STAGES.items():
            print(f"{k:8s} {desc}")
        return 0
    paths = setup_environment(a.dest)
    if a.cmd == "predict":
        from detr_voc.infer import Predictor
        print(json.dumps(Predictor(a.checkpoint).predict(a.image, a.threshold), indent=2))
        return 0
    cfg = load_config(a.stage, a.set, dest=a.dest)
    if a.cmd == "prepare-data":
        from detr_voc.data import build_data
        d = build_data(cfg, paths)
        print(json.dumps({"data_version": d["data_version"], "train": len(d["train"]), "val": len(d["val"] or []), "test": len(d["test"] or [])}))
    elif a.cmd == "benchmark":
        from detr_voc.train import benchmark
        print(json.dumps(benchmark(cfg, paths), indent=2))
    elif a.cmd == "train":
        from detr_voc.train import run_training
        run_training(cfg, paths)
    elif a.cmd == "test":
        from detr_voc.evaluate import run_test_evaluation
        run_test_evaluation(cfg, paths, a.run_id, a.force)
    elif a.cmd == "package":
        from detr_voc.data import build_data
        from detr_voc.serve import package_model
        d = build_data(cfg, paths)
        print(json.dumps(package_model(cfg, paths, a.run_id, (d["val"] or d["test"]).source.get(0)[0], paths["dest"] / "models" / "packaged")))
    return 0
