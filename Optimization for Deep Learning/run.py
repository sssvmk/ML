#!/usr/bin/env python3
"""Run the whole chapter-8 study with ONE command.

  python run.py                                    # all 40 methods on real MNIST (torchvision downloads it)
  python run.py --methods adam,lbfgs,init_glorot   # a subset (the baseline is always included)
  python run.py --fast                             # 2 epochs, 2 trials, 5 full-batch iterations per method
  python run.py --smoke                            # synthetic MNIST-shaped data, tiny: pipeline check only
  python run.py --set data.source=npz:/path/mnist.npz --set train.epochs=30
  python run.py --promote --approver "Your Name"   # move the `champion` alias if every gate passes

Results: <out>/comparison.md|csv|png, all_loss_curves.png, MODEL_CARD.md, methods/<name>/..., mlflow.db
Browse runs:  mlflow ui --backend-store-uri sqlite:///<out>/mlflow.db
"""
import argparse
import logging
import sys

from optpipe.config import load_config
from optpipe.methods import REGISTRY
from optpipe.orchestrator import run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--methods", default="all", help="comma-separated names or 'all': " + ",".join(REGISTRY))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--approver", default="")
    a = ap.parse_args(argv)

    over = list(a.set)
    if a.fast:
        over += ["train.epochs=2", "tune.epochs=1", "tune.n_trials=2", "full_batch.iterations=5", "tune.fullbatch_iters=3"]
    if a.smoke:
        over += ["data.source=synthetic", "data.val_size=500", "data.synthetic_train=3000", "data.synthetic_test=800",
                 "train.epochs=2", "tune.epochs=1", "tune.n_trials=2", "full_batch.iterations=4", "tune.fullbatch_iters=3",
                 "model.hidden=[32,16]", "method_settings.newton.curvature_batch=500"]
    cfg = load_config(a.config, over)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    names = "all" if a.methods == "all" else [m.strip() for m in a.methods.split(",") if m.strip()]
    result = run(cfg, names, a.out, promote=a.promote, approver=a.approver)
    print(f"\nbest full-network method: {result['best']}   results in {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
