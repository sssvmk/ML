"""Promotion gate: move alias `champion` to a registered version only if every criterion passes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.utils import ROOT, registered_name


def run_gates(cfg, env, version: str, approver: str | None, accept_gap_reason: str | None = None) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    client = MlflowClient()
    name = registered_name(cfg, env)
    mv = client.get_model_version(name, version)
    run = client.get_run(mv.run_id)
    m, tags = run.data.metrics, run.data.tags
    target = cfg["metric"]["target_value"]
    gates = {}

    gates["test_set_used_exactly_once"] = tags.get("test_evaluated") == "true" and tags.get("test_evaluation_count") == "1"
    test = m.get("test_top1")
    meets = test is not None and test >= target
    gates["target_met_or_gap_accepted"] = bool(meets or accept_gap_reason or not cfg["promotion"]["require_target"])
    gates["load_verified"] = mv.tags.get("load_verified") == "true"
    gates["sanity_checks_passed"] = tags.get("sanity_passed") == "true"

    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_val = client.get_run(champ.run_id).data.metrics.get("best_val_top1", float("-inf"))
        gates["beats_champion_on_validation"] = m.get("best_val_top1", float("-inf")) >= champ_val + cfg["promotion"]["min_improvement"]
    except MlflowException:
        gates["beats_champion_on_validation"] = True  # no champion yet
    card = ROOT / "reports" / "MODEL_CARD.md"
    gates["model_card_exists"] = card.exists()
    gates["approver_named"] = bool(approver)

    passed = all(gates.values())
    record = {
        "model": name, "version": version, "run_id": mv.run_id, "approver": approver,
        "time_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "passed": passed,
        "test_top1": test, "target": target, "gap_accepted_reason": accept_gap_reason,
        "note": "single-run comparison; repeat across seeds before relying on small margins",
    }
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
