"""Promotion gate: move alias `champion` to a registered version only if every criterion passes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.utils import ROOT, registered_name


def _val_metric(client, run) -> float:
    """Validation metric of a run; a refit run (no validation split) uses the run it was refitted from."""
    src = run.data.tags.get("refit_of")
    r = client.get_run(src) if src else run
    return r.data.metrics.get("best_val_metric", float("-inf"))


def run_gates(cfg, env, version: str, approver: str | None, accept_gap_reason: str | None = None) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    client = MlflowClient()
    name = registered_name(cfg, env)
    mv = client.get_model_version(name, version)
    run = client.get_run(mv.run_id)
    m, tags = run.data.metrics, run.data.tags
    target = cfg["metric"]["target_value"]
    test = m.get(f"test_{cfg['metric']['name']}")
    meets = test is not None and test >= target
    gates = {
        "test_set_used_exactly_once": tags.get("test_evaluated") == "true" and tags.get("test_evaluation_count") == "1",
        "target_met_or_gap_accepted": bool(meets or accept_gap_reason or not cfg["promotion"]["require_target"]),
        "load_verified": mv.tags.get("load_verified") == "true",
        "sanity_checks_passed": tags.get("sanity_passed") == "true" or bool(tags.get("refit_of")),
    }
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_val = _val_metric(client, client.get_run(champ.run_id))
        gates["beats_champion_on_validation"] = _val_metric(client, run) >= champ_val + cfg["promotion"]["min_improvement"]
    except MlflowException:
        gates["beats_champion_on_validation"] = True  # no champion yet
    gates["model_card_exists"] = (ROOT / "reports" / "MODEL_CARD.md").exists()
    gates["approver_named"] = bool(approver)
    passed = all(gates.values())
    record = {"model": name, "version": version, "run_id": mv.run_id, "approver": approver,
              "time_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "passed": passed,
              "test_metric": test, "target": target, "gap_accepted_reason": accept_gap_reason,
              "note": "single-run comparison; repeat across seeds before relying on small margins"}
    (ROOT / "approvals").mkdir(exist_ok=True)
    (ROOT / "approvals" / f"{name}_v{version}.json").write_text(json.dumps(record, indent=2))
    with open(ROOT / "approvals" / "promotion_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    if passed:
        client.set_registered_model_alias(name, "champion", version)
        client.set_model_version_tag(name, version, "validated", "true")
        client.set_model_version_tag(name, version, "approver", approver)
    print(json.dumps(record, indent=2))
    return record
