#!/usr/bin/env python3
"""
MLP inference. Loads the self-contained model.pt bundle (architecture + weights +
preprocessing) from a local path or from the MLflow Model Registry.

MNIST (digit classification)
    python infer.py --model runs/mnist/model.pt --image digit1.png digit2.png
    python infer.py --registry models:/MLP-mnist@champion --image digit.png
    python infer.py --model runs/mnist/model.pt --test-index 0 1 2 3     # MNIST test samples

    Images: any common format, ideally one digit on a plain background. By default the digit
    is cropped, scaled to 20px and centered in a 28x28 canvas (how MNIST was built) and
    colors are auto-inverted when the background is light. --raw skips the recentering.

California Housing (price regression)
    python infer.py --model runs/housing/model.pt --csv houses.csv
    python infer.py --model runs/housing/model.pt --values=8.3,41,6.9,1.0,322,2.5,37.88,-122.23
    python infer.py --registry models:/MLP-housing@champion --demo

    Use --values=... with '=' so a leading negative number isn't read as a flag.
    Feature order: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
    Longitude. A CSV needs those column names (case-insensitive; extra columns ignored).

Your own CSV model (trained with run.py --csv-path ...)
    python infer.py --model runs/<name>/model.pt --csv new_rows.csv [--out-csv predictions.csv]
    Rows need the ORIGINAL feature columns (any extra columns, including the target, are ignored).
    Missing values are fine: they are imputed exactly as in training. Output: classification ->
    label, confidence, prob_<class>; regression -> prediction. `out_of_range_warning` is true
    when a numeric value is more than 6 standard deviations from the training mean.

--registry needs the tracking URI: --tracking-uri or the MLFLOW_TRACKING_URI environment
variable (default sqlite:///mlflow.db). Add --json for machine-readable output.

In Python:
    from infer import Predictor
    p = Predictor("runs/mnist/model.pt")       # or Predictor.from_registry("models:/MLP-mnist@champion")
    p.predict_mnist(arrays)                     # arrays: [n,28,28] floats in 0..1
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

from mlp_core import build_mlp


class Predictor:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        b = torch.load(model_path, map_location=self.device, weights_only=True)
        self.task, self.dataset = b["task"], b["dataset"]
        self.pre, self.class_names = b["preproc"], b.get("class_names")
        self.test_metrics = b.get("test_metrics", {})
        self.model = build_mlp(b["in_dim"], b["hidden"], b["out_dim"], b.get("dropout", 0.0))
        self.model.load_state_dict(b["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def from_registry(cls, uri, device="cpu"):
        """uri like models:/MLP-mnist@champion or models:/MLP-mnist/3"""
        import mlflow
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        hits = glob.glob(os.path.join(local, "**", "model.pt"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"no model.pt bundle found inside {uri}")
        return cls(hits[0], device)

    # ------------------------------ classification ------------------------------ #
    @torch.no_grad()
    def mnist_probs(self, arrays):
        """arrays: floats [n,28,28] or [n,784] in 0..1 (white digit on black) -> probs [n,10]."""
        if self.task != "classification":
            raise ValueError("this model is not a classifier")
        x = np.asarray(arrays, dtype=np.float32)
        x = x.reshape(len(x), -1)
        if x.shape[1] != 784:
            raise ValueError(f"expected 784 pixels per image, got {x.shape[1]}")
        if not np.isfinite(x).all():
            raise ValueError("inputs contain NaN or infinite values")
        if x.max() > 1.5 or x.min() < -0.5:
            raise ValueError("pixel values must be scaled to 0..1 (not 0..255)")
        x = (x - self.pre["mean"]) / self.pre["std"]
        logits = self.model(torch.from_numpy(x).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_mnist(self, arrays, topk=3):
        out = []
        for p in self.mnist_probs(arrays):
            idx = np.argsort(-p)[:topk]
            out.append({
                "label": self.class_names[int(idx[0])], "confidence": float(p[idx[0]]),
                "top": [{"label": self.class_names[int(i)], "prob": float(p[i])} for i in idx],
            })
        return out

    # ------------------------- generic CSV (tabular) models ------------------------ #
    @torch.no_grad()
    def tabular_frame(self, df):
        """Raw rows (DataFrame with the original feature columns) -> predictions DataFrame."""
        import pandas as pd
        from tabular import TabularPreprocessor

        if self.pre.get("type") != "tabular":
            raise ValueError("this model was not trained from a CSV (use predict_mnist / predict_housing)")
        if not hasattr(self, "_prep"):
            self._prep = TabularPreprocessor.from_dict(self.pre)
        X = self._prep.transform(df)
        n_num = self._prep.n_numeric
        far = (np.abs(X[:, :n_num]) > 6).any(axis=1) if n_num else np.zeros(len(X), dtype=bool)
        out = self.model(torch.from_numpy(X).to(self.device)).cpu()
        if self.task == "classification":
            probs = torch.softmax(out, dim=1).numpy()
            frame = pd.DataFrame({
                "label": [self.class_names[i] for i in probs.argmax(axis=1)],
                "confidence": probs.max(axis=1).astype("float64")})
            for i, cname in enumerate(self.class_names):
                frame[f"prob_{cname}"] = probs[:, i].astype("float64")
        else:
            frame = pd.DataFrame({"prediction": (out.numpy().ravel() * self.pre["y_std"]
                                                 + self.pre["y_mean"]).astype("float64")})
        frame["out_of_range_warning"] = far
        return frame

    # -------------------------------- regression -------------------------------- #
    @torch.no_grad()
    def predict_housing(self, rows):
        """rows: array [n,8] of raw (unscaled) features in the documented order."""
        if self.task != "regression":
            raise ValueError("this model is not a regressor")
        X = np.atleast_2d(np.asarray(rows, dtype=np.float32))
        names = self.pre["feature_names"]
        if X.shape[1] != len(names):
            raise ValueError(f"expected {len(names)} features {names}, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("inputs contain NaN or infinite values")
        z = (X - np.array(self.pre["x_mean"], dtype=np.float32)) / np.array(self.pre["x_scale"], dtype=np.float32)
        out = self.model(torch.from_numpy(z).to(self.device)).cpu().numpy().ravel()
        pred = out * self.pre["y_std"] + self.pre["y_mean"]
        far = (np.abs(z) > 6).any(axis=1)  # far outside the training distribution
        return [{"prediction_100k": float(p), "prediction_usd": float(p * 1e5),
                 "out_of_range_warning": bool(f)} for p, f in zip(pred, far)]


# ------------------------------ image helpers ------------------------------ #
def load_digit_image(path, invert="auto", recenter=True):
    """Image file -> float32 [28,28], white digit on black background, values 0..1."""
    from PIL import Image

    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.float32) / 255.0
    if invert == "yes" or (invert == "auto" and a.mean() > 0.5):
        a = 1.0 - a
    if not recenter:
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((28, 28), Image.BILINEAR),
                          dtype=np.float32) / 255.0
    mask = a > 0.2
    if not mask.any():  # blank image: nothing to crop
        return np.zeros((28, 28), dtype=np.float32)
    ys, xs = np.where(mask)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - small.width) // 2, (28 - small.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def read_housing_csv(path, feature_names):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        missing = [n for n in feature_names if n.lower() not in cols]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        return [[float(r[cols[n.lower()]]) for n in feature_names] for r in reader]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="path to a local model.pt bundle")
    src.add_argument("--registry", help="MLflow model URI, e.g. models:/MLP-mnist@champion")
    ap.add_argument("--tracking-uri", help="MLflow tracking URI (for --registry)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--topk", type=int, default=3)
    g = ap.add_argument_group("MNIST inputs")
    g.add_argument("--image", nargs="+", help="image file(s) of handwritten digits")
    g.add_argument("--test-index", type=int, nargs="+", help="indices into the MNIST test set")
    g.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    g.add_argument("--raw", action="store_true", help="skip crop/recenter; just resize to 28x28")
    h = ap.add_argument_group("Housing inputs")
    h.add_argument("--csv", help="CSV with the feature columns (housing: the 8 columns; "
                                 "your own CSV model: the original feature columns)")
    h.add_argument("--out-csv", help="write predictions to this CSV (models trained from your own CSV)")
    h.add_argument("--values", help="comma-separated feature values for one house")
    h.add_argument("--demo", action="store_true", help="predict the first 3 rows of the dataset")
    args = ap.parse_args()

    if args.registry:
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        p = Predictor.from_registry(args.registry)
    else:
        p = Predictor(args.model)
    results = []

    if p.pre.get("type") == "tabular":
        import pandas as pd
        if not args.csv:
            sys.exit("This model was trained from a CSV: pass --csv new_rows.csv (original feature columns)")
        raw = pd.read_csv(args.csv, low_memory=False)
        frame = p.tabular_frame(raw)
        if args.out_csv:
            pd.concat([raw.reset_index(drop=True), frame], axis=1).to_csv(args.out_csv, index=False)
            print(f"wrote {len(frame):,} predictions to {args.out_csv}")
        results = frame.to_dict("records")
        if not args.json:
            print(frame.head(20).to_string(index=False))
            if len(frame) > 20:
                print(f"... {len(frame) - 20:,} more rows")
            if frame["out_of_range_warning"].any():
                print(f"[warning] {int(frame['out_of_range_warning'].sum())} row(s) have a numeric value more "
                      f"than 6 standard deviations from the training mean")
    elif p.task == "classification":
        if args.image:
            arrays = [load_digit_image(f, args.invert, not args.raw) for f in args.image]
            names, truth = args.image, [None] * len(arrays)
        elif args.test_index:
            from torchvision import datasets
            ds = datasets.MNIST("./data", train=False, download=True)
            arrays = [ds.data[i].numpy().astype(np.float32) / 255.0 for i in args.test_index]
            names = [f"test[{i}]" for i in args.test_index]
            truth = [str(int(ds.targets[i])) for i in args.test_index]
        else:
            sys.exit("MNIST model: pass --image FILE... or --test-index N...")
        for name, t, r in zip(names, truth, p.predict_mnist(arrays, args.topk)):
            r["input"] = name
            if t is not None:
                r["true_label"] = t
            results.append(r)
        if not args.json:
            for r in results:
                top = ", ".join(f"{x['label']} ({x['prob']:.1%})" for x in r["top"])
                extra = f"   true: {r['true_label']}" if "true_label" in r else ""
                warn = "   [low confidence]" if r["confidence"] < 0.6 else ""
                print(f"{r['input']}: predicted {r['label']}  |  top: {top}{extra}{warn}")
    else:
        names, truth = p.pre["feature_names"], None
        if args.csv:
            rows = read_housing_csv(args.csv, names)
        elif args.values:
            rows = [[float(v) for v in args.values.split(",")]]
        elif args.demo:
            from sklearn.datasets import fetch_california_housing
            d = fetch_california_housing()
            rows, truth = d.data[:3].tolist(), d.target[:3].tolist()
            print("(demo rows may have been in the training set; smoke test, not an accuracy measure)")
        else:
            sys.exit("Housing model: pass --csv FILE, --values=a,b,..., or --demo")
        for i, r in enumerate(p.predict_housing(rows)):
            r["input"] = dict(zip(names, rows[i]))
            if truth:
                r["actual_100k"] = truth[i]
            results.append(r)
        if not args.json:
            for i, r in enumerate(results):
                extra = f"   actual ${r['actual_100k'] * 1e5:,.0f}" if "actual_100k" in r else ""
                warn = "   [warning: input far outside training range]" if r["out_of_range_warning"] else ""
                print(f"row {i}: predicted ${r['prediction_usd']:,.0f}{extra}{warn}")
            print("note: the dataset caps house values at $500k, so very high prices are unreliable")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
