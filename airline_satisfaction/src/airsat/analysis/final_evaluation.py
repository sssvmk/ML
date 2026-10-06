"""One-shot evaluation of the declared winner on the untouched TEST set (+ slices, calibration, target check)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..bundle import ModelBundle
from ..metrics import compute_metrics, slice_metrics
from ..plots import plot_calibration, plot_confusion, plot_roc_pr
from ..utils import dump_json


class HoldoutAlreadyUsed(RuntimeError):
    pass


def evaluate_on_test(bundle_dir: Path, X_test_raw, y_test, cfg: dict, run_dir: Path, force: bool = False) -> dict:
    run_dir = Path(run_dir)
    lock = run_dir / "test_set_used.lock"
    if lock.exists() and not force:
        raise HoldoutAlreadyUsed(f"{lock} exists: the test set was already evaluated for this run. Re-using it for selection "
                                 "would bias the estimate. Pass force=True only to re-report, never to re-tune.")
    bundle = ModelBundle.load(bundle_dir)
    t0 = time.perf_counter()
    p = bundle.predict_proba(X_test_raw)
    seconds = time.perf_counter() - t0
    y = np.asarray(y_test)
    m = compute_metrics(y, p, bundle.threshold)
    clean_fe = bundle.feature_engineer.transform(X_test_raw)
    slices = slice_metrics(clean_fe, y, p, bundle.threshold, cfg["diagnostics"]["slice_columns"])
    plots_dir = run_dir / "reports" / "test"
    artifacts = {"roc_pr": plot_roc_pr(y, p, plots_dir / "roc_pr.png", "(TEST)"),
                 "calibration": plot_calibration(y, p, plots_dir / "calibration.png", "(TEST)"),
                 "confusion": plot_confusion(y, (p >= bundle.threshold).astype(int), plots_dir / "confusion.png", "(TEST)")}
    metric = cfg["metric"]["primary"]
    primary = -m["log_loss"] if metric == "neg_log_loss" else m[metric]
    target = cfg["metric"].get("target_value")
    max_gap = max((s["accuracy_gap"] for s in slices.values()), default=0.0)
    out = {"algorithm": bundle.algorithm, "n_test_rows": int(len(y)), "primary_metric": metric, "primary_value": primary,
           "metrics": m, "target_value": target, "meets_target": None if target is None else bool(primary >= target),
           "slice_metrics": slices, "max_slice_accuracy_gap": max_gap, "inference_seconds": seconds,
           "test_set_evaluations": 1, "artifacts": artifacts}
    lock.write_text(f"test set evaluated once for {bundle.algorithm}\n", encoding="utf-8")
    dump_json(out, run_dir / "test_evaluation.json")
    return out
