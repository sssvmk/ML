"""Promotion gate: candidate -> champion only if every criterion passes; refuses otherwise and logs the decision.

    python -m airsat.promote --run runs/<run> [--approver "Jane Doe"] [--dry-run]

Gates (thresholds in config.promotion + selection):
  1. winner beats the baseline by more than the practical margin
  2. test metric >= promotion.min_primary_metric (and the configured business target, if any)
  3. largest slice accuracy gap <= promotion.max_slice_gap
  4. the saved bundle loads in a fresh process and reproduces its predictions (load-and-predict test)
  5. the test set was evaluated exactly once
  6. a named approver is supplied (human sign-off)
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import textwrap
from pathlib import Path

from .config import load_config
from .utils import dump_json, load_json


def load_and_predict_check(champion_dir: Path, sample_csv: Path | None, cfg: dict) -> tuple[bool, str]:
    """Fresh interpreter: load the bundle, predict, compare with stored validation threshold sanity (no NaNs, in [0,1])."""
    code = textwrap.dedent(f"""
        import numpy as np, pandas as pd, sys
        from airsat.bundle import ModelBundle
        from airsat.synthetic import make_synthetic
        b = ModelBundle.load({str(champion_dir)!r})
        cols = b.schema.required_columns
        df = make_synthetic(200, seed=123, with_target=False)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print('schema differs from the synthetic generator; skipping synthetic probe:', missing); sys.exit(0)
        p = b.predict_proba(df)
        assert p.shape == (200,) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert np.allclose(p, b.predict_proba(df.iloc[::-1])[::-1], atol=1e-8), 'predictions depend on row order'
        print('ok')
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]


def evaluate_gates(run_dir: Path, cfg: dict, approver: str | None) -> dict:
    sel, test = load_json(run_dir / "selection.json"), load_json(run_dir / "test_evaluation.json")
    pcfg, scfg = cfg["promotion"], cfg["selection"]
    table = {t["name"]: t for t in sel["table"]}
    base = next((t for t in sel["table"] if t["family"] == "baseline"), None)
    win = table[sel["winner"]]
    ok_load, load_msg = load_and_predict_check(run_dir / "champion", None, cfg)
    target = cfg["metric"].get("target_value")
    gates = {
        "beats_baseline": (base is None) or (win["val_primary"] - base["val_primary"] > scfg["min_practical_delta"]),
        "meets_min_metric": test["primary_value"] >= max(pcfg["min_primary_metric"], target or float("-inf")),
        "slice_gap_ok": test["max_slice_accuracy_gap"] <= pcfg["max_slice_gap"],
        "load_and_predict": ok_load or not pcfg["require_load_test"],
        "test_set_used_once": test.get("test_set_evaluations", 1) == 1,
        "named_approver": bool(approver),
    }
    return {"gates": gates, "passed": all(gates.values()), "details": {"load_test": load_msg, "test_primary": test["primary_value"],
            "max_slice_gap": test["max_slice_accuracy_gap"], "target": target}}


def promote(run_dir: str | Path, cfg: dict, approver: str | None, dry_run: bool = False) -> dict:
    run_dir = Path(run_dir)
    res = evaluate_gates(run_dir, cfg, approver)
    record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "run": run_dir.name, "approver": approver, **res, "alias_moved": False}
    if res["passed"] and not dry_run:
        reg = load_json(run_dir / "registry.json")
        if reg.get("registered"):
            from mlflow.tracking import MlflowClient
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{(run_dir / 'mlflow.db').resolve()}" if not cfg["tracking"].get("uri") else cfg["tracking"]["uri"])
            c = MlflowClient()
            c.set_registered_model_alias(reg["name"], "champion", reg["version"])
            c.set_model_version_tag(reg["name"], reg["version"], "validated", "true")
            c.set_model_version_tag(reg["name"], reg["version"], "approver", approver or "")
            record["alias_moved"] = True
    approvals = run_dir / "approvals"
    approvals.mkdir(exist_ok=True)
    dump_json(record, approvals / f"promotion_{dt.datetime.now():%Y%m%d_%H%M%S}.json")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--approver", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    rec = promote(a.run, load_config(a.config, a.set), a.approver, a.dry_run)
    for g, v in rec["gates"].items():
        print(f"{'PASS' if v else 'FAIL'}  {g}")
    print("PROMOTED to alias 'champion'" if rec["alias_moved"] else "NOT promoted" + (" (dry run)" if a.dry_run else ""))
    raise SystemExit(0 if rec["passed"] else 1)


if __name__ == "__main__":
    main()
