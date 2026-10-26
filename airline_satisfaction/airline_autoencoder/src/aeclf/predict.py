"""Step 11: batch inference on an UNSEEN dataset with the winning model.

    python -m aeclf.predict --input unseen.csv --model runs/<run>/champion --output predictions.csv [--drift-report]
    python -m aeclf.predict ... --extras      # also writes <output>.embeddings.csv and <output>.reconstruction_error.csv

Output format comes from config.output (default: columns `id,satisfaction`, TRUE/FALSE, like the training file).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .config import load_config
from .io_utils import load_unseen_data
from .metrics import compute_metrics
from .monitoring import drift_report
from .utils import dump_json
from .validation import validate_input


def format_predictions(ids: pd.Series, proba: np.ndarray, label: np.ndarray, out_cfg: dict) -> pd.DataFrame:
    style = out_cfg["label_style"]
    if style == "logical":
        col = pd.Series(label.astype(bool)).map({True: "TRUE", False: "FALSE"})
    elif style == "binary":
        col = pd.Series(label.astype(int))
    else:
        col = pd.Series(label.astype(bool)).map({True: "satisfied", False: "neutral or dissatisfied"})
    out = pd.DataFrame({out_cfg["id_column"]: ids.to_numpy(), out_cfg["prediction_column"]: col.to_numpy()})
    if out_cfg.get("include_probability"):
        out[out_cfg["probability_column"]] = np.round(proba, 6)
    return out


def predict_file(input_path, model_dir, output_path, cfg, drift=False, extras=False) -> pd.DataFrame:
    b = ModelBundle.load(model_dir)
    X, y, ids = load_unseen_data(input_path, cfg)
    clean, warns = validate_input(X, b.schema)
    for w in warns:
        print(f"[validation] {w}")
    proba = b.predict_proba(X)
    label = (proba >= b.threshold).astype(int)
    out = format_predictions(ids, proba, label, cfg["output"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {len(out)} predictions to {output_path} (model: {b.metadata.get('candidate')}, threshold {b.threshold:.3f}, predicted positive rate {label.mean():.1%})")
    if y is not None:
        print("metrics on the labelled unseen file:", json.dumps({k: round(v, 4) for k, v in compute_metrics(y.to_numpy(), proba, b.threshold).items()}))
    if drift:
        rep = drift_report(clean, b.reference_stats)
        dump_json(rep, Path(output_path).with_suffix(".drift.json"))
        print(f"drift: {rep['n_alert']} features in alert, {rep['n_warn']} in warning")
    if extras and b.ae_state is not None:
        emb = pd.DataFrame(b.embed(X), columns=[f"z{i}" for i in range(b.ae_arch["latent"])])
        emb.insert(0, cfg["output"]["id_column"], ids.to_numpy())
        emb.to_csv(Path(output_path).with_suffix(".embeddings.csv"), index=False)
        _, err = b.reconstruct(X)
        pd.DataFrame({cfg["output"]["id_column"]: ids.to_numpy(), "reconstruction_error": err}).to_csv(Path(output_path).with_suffix(".reconstruction_error.csv"), index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--drift-report", action="store_true")
    ap.add_argument("--extras", action="store_true")
    a = ap.parse_args(argv)
    predict_file(a.input, a.model, a.output, load_config(a.config, a.set), a.drift_report, a.extras)


if __name__ == "__main__":
    main()
