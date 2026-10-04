#!/usr/bin/env python3
"""Run the whole regularization study with ONE command.

  python run.py                                   # all 19 methods on real MNIST (downloads via torchvision)
  python run.py --methods l2_weight_decay,dropout # a subset (baseline is always included)
  python run.py --fast                            # quick pass: 2 epochs, 2 trials per method
  python run.py --smoke                           # synthetic MNIST-shaped data, tiny: pipeline check only
  python run.py --set data.source=npz:/path/mnist.npz --set train.epochs=30
  python run.py --promote --approver "Your Name"  # move the `champion` alias if every gate passes

Results: <out>/comparison.md|csv|png, all_loss_curves.png, MODEL_CARD.md, methods/<name>/..., mlflow.db
Browse runs:  mlflow ui --backend-store-uri sqlite:///<out>/mlflow.db
"""
import argparse
import logging
import sys

from regpipe.config import load_config
from regpipe.methods import REGISTRY
from regpipe.orchestrator import run


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
        over += ["train.epochs=2", "tune.epochs=1", "tune.n_trials=2"]
    if a.smoke:
        over += ["data.source=synthetic", "data.val_size=500", "data.synthetic_train=3000", "data.synthetic_test=800",
                 "train.epochs=2", "tune.epochs=1", "tune.n_trials=2", "model.hidden=[64,32]",
                 "method_settings.under_constrained.n_train=200", "method_settings.semi_supervised.n_labeled=200",
                 "method_settings.bagging.n_members=2", "method_settings.bagging.tune_members=2",
                 "method_settings.tangent.n_prototypes=300"]
    cfg = load_config(a.config, over)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    names = "all" if a.methods == "all" else [m.strip() for m in a.methods.split(",") if m.strip()]
    result = run(cfg, names, a.out, promote=a.promote, approver=a.approver)
    print(f"\nbest full-data method: {result['best']}   results in {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
