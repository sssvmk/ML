#!/usr/bin/env python3
"""Command line entry point.

  python run.py gpu-check                         # is the GPU visible and correct? which settings are fastest?
  python run.py benchmark                         # time a few steps with the configured settings, project full-run cost
  python run.py train   [--config C] [--set a.b=v ...]
  python run.py test    --run-id RUN              # one-time test-set evaluation
  python run.py register --run-id RUN             # package, register, verify reload, alias `candidate`
  python run.py promote --version N --approver NAME [--accept-gap "reason"]
  python run.py predict --model-uri models:/vgg16-cifar100@champion --image a.png b.png
"""
import argparse
import json

from src.utils import load_config, load_env


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "benchmark", "gpu-check", "test", "register", "promote"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="config/train.yaml")
        s.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override config values")
        if name in ("test", "register"):
            s.add_argument("--run-id", required=True)
        if name == "test":
            s.add_argument("--force", action="store_true")
        if name in ("benchmark", "gpu-check"):
            s.add_argument("--steps", type=int, default=6)
        if name == "promote":
            s.add_argument("--version", required=True)
            s.add_argument("--approver", required=True)
            s.add_argument("--accept-gap", default=None, help="reason, if the target is not met")
    s = sub.add_parser("predict")
    s.add_argument("--model-uri", required=True)
    s.add_argument("--image", nargs="+", required=True)
    s.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    env = load_env()

    if args.cmd == "predict":
        from src.serve import Predictor
        print(json.dumps(Predictor(args.model_uri, env["tracking_uri"]).predict(args.image, args.topk), indent=2))
        return
    cfg = load_config(args.config, args.set)
    if args.cmd == "benchmark":
        from src.train import benchmark
        print(json.dumps(benchmark(cfg, args.steps), indent=2))
    elif args.cmd == "gpu-check":
        from src.hwcheck import gpu_check
        gpu_check(cfg, args.steps)
    elif args.cmd == "train":
        from src.train import run_training
        print(json.dumps(run_training(cfg, env, args.config), indent=2))
    elif args.cmd == "test":
        from src.evaluate import run_test_evaluation
        run_test_evaluation(cfg, env, args.run_id, args.force)
    elif args.cmd == "register":
        from src.register import register_candidate
        register_candidate(cfg, env, args.run_id)
    elif args.cmd == "promote":
        from src.promote import run_gates
        run_gates(cfg, env, args.version, args.approver, args.accept_gap)


if __name__ == "__main__":
    main()
