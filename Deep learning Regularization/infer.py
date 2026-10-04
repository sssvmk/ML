#!/usr/bin/env python3
"""Inference system.  Classify digits with a saved bundle or a registered MLflow model.

  python infer.py --model results/methods/dropout/bundle --image digit.png
  python infer.py --model results/methods/bagging/bundle --npy batch.npy          # float [n,784] in [0,1]
  python infer.py --model results/methods/dropout/bundle --test-index 0 1 2       # needs the MNIST data
  python infer.py --model "models:/MNIST-Reg@candidate" --tracking-uri sqlite:///results/mlflow.db --image d.png

Images: any size (resized to 28x28); white digit on black is the MNIST convention.  --invert auto
flips dark-on-light images automatically.  Output is JSON: label, confidence, top-3, probabilities.
"""
import argparse
import json
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="bundle directory or models:/NAME@alias URI")
    ap.add_argument("--tracking-uri", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", nargs="+")
    g.add_argument("--npy")
    g.add_argument("--test-index", nargs="+", type=int)
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--data-source", default="mnist")
    ap.add_argument("--data-root", default="./data")
    a = ap.parse_args(argv)

    from regpipe.bundle import read_image

    truth = None
    if a.image:
        x = np.stack([read_image(p, a.invert) for p in a.image])
    elif a.npy:
        x = np.load(a.npy).astype(np.float32).reshape(-1, 784)
    else:
        from regpipe.data import load_raw, to_float
        raw = load_raw({"source": a.data_source, "root": a.data_root})
        x = to_float(raw["x_test"][a.test_index]).numpy()
        truth = raw["y_test"][a.test_index].tolist()

    if a.model.startswith(("models:/", "runs:/")):
        import mlflow
        if a.tracking_uri:
            mlflow.set_tracking_uri(a.tracking_uri)
        df = mlflow.pyfunc.load_model(a.model).predict(x)
        preds = [{"label": int(r.label), "confidence": float(r.confidence)} for r in df.itertuples()]
    else:
        from regpipe.bundle import Predictor
        preds = Predictor(a.model).predict(x)
    if truth is not None:
        for p, t in zip(preds, truth):
            p["true_label"] = t
    json.dump(preds, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
