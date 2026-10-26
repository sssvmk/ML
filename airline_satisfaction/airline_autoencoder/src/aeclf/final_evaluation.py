"""One-shot evaluation of the winner on the untouched TEST set (classification + reconstruction fidelity)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .bundle import ModelBundle
from .metrics import compute_metrics, slice_metrics
from .plots import plot_calibration, plot_confusion, plot_roc_pr
from .training import reconstruction_report
from .utils import dump_json


class HoldoutAlreadyUsed(RuntimeError):
    pass


def evaluate_on_test(bundle_dir: Path, X_test_raw, y_test, cfg: dict, run_dir: Path, force: bool = False) -> dict:
    run_dir = Path(run_dir)
    lock = run_dir / "test_set_used.lock"
    if lock.exists() and not force:
        raise HoldoutAlreadyUsed(f"{lock} exists: the test set was already evaluated for this run. Re-using it would bias the estimate.")
    b = ModelBundle.load(bundle_dir)
    t0 = time.perf_counter()
    p = b.predict_proba(X_test_raw)
    secs = time.perf_counter() - t0
    y = np.asarray(y_test)
    m = compute_metrics(y, p, b.threshold)
    sl = slice_metrics(X_test_raw.reset_index(drop=True), y, p, b.threshold, cfg["diagnostics"]["slice_columns"])
    rd = run_dir / "reports" / "test"
    metric = cfg["metric"]["primary"]
    primary = -m["log_loss"] if metric == "neg_log_loss" else m[metric]
    target = cfg["metric"].get("target_value")
    recon = None
    if b.ae_state is not None:
        rep, ex = reconstruction_report(b.autoencoder, b.encoder, X_test_raw, "cpu", cfg["reconstruction"]["examples"])
        ex.to_csv(rd.parent / "test_reconstruction_examples.csv", index=False) if rd.parent.exists() else None
        recon = {"summary": rep["summary"], "continuous": rep["continuous"], "categorical": rep["categorical"]}
    out = {"candidate": b.metadata.get("candidate"), "kind": b.kind, "mode": b.mode, "n_test_rows": int(len(y)), "primary_metric": metric, "primary_value": primary,
           "metrics": m, "target_value": target, "meets_target": None if target is None else bool(primary >= target), "slice_metrics": sl,
           "max_slice_accuracy_gap": max((s["accuracy_gap"] for s in sl.values()), default=0.0), "inference_seconds": secs,
           "test_set_evaluations": 1, "pretrained_autoencoder_reconstruction_on_test": recon,
           "artifacts": {"roc_pr": plot_roc_pr(y, p, rd / "roc_pr.png", "(TEST)"), "calibration": plot_calibration(y, p, rd / "calibration.png", "(TEST)"),
                         "confusion": plot_confusion(y, (p >= b.threshold).astype(int), rd / "confusion.png", "(TEST)")}}
    lock.write_text(f"test set evaluated once for {out['candidate']}\n", encoding="utf-8")
    dump_json(out, run_dir / "test_evaluation.json")
    return out
