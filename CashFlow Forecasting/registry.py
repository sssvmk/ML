"""
Model registry and lifecycle (PRD §4.3, §4.4; gap G-01).

Every training / revalidation that wins registers a NEW VERSION for the
segment (one registered model per segment). A version moves
None -> Staging -> Production and the previously-Production version is
Archived. `get_active()` returns the Production version's entry, so the
three cadences (full training, weekly revalidation, daily inference) keep
sharing state exactly as before.

This class is the local system of record and works with no MLflow at all
(JSON index, so the rollback logic is always testable). When an
MLflow backend is supplied (mlflow_logging.MLflowLogger) every register /
transition is mirrored onto MLflow's Model Registry, whose stage is then
the operational truth for anyone browsing MLflow.

Rollback (§4.4, last paragraph) is HUMAN-DECIDED (decision on G-01): when the
production monitor finds that the prior version's live six-metric rank-sum
beats the current Production version's, it only records a PROPOSAL
(`propose_rollback`). A named reviewer then calls `decide_rollback(...)`:
approving executes `rollback()` (archive the failing Production version,
restore the prior one, flag the segment for the next revalidation cycle);
rejecting leaves Production untouched. Every proposal and decision is kept
in the registry index as an audit trail.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import threading

STAGES = ("None", "Staging", "Production", "Archived")


@dataclass
class RegistryEntry:
    segment_id: str
    algorithm_name: str  # or "fallback" when using the rolling combine
    is_fallback: bool
    hyperparameters: dict
    artifact_path: str
    rule_version: str  # Data Module rule_version this model was fit on (§4.2)
    metrics: dict  # backtest metrics used for selection (six-metric, §3.4)
    fallback_combine: str | None = None  # e.g. "rolling_mean_median" when is_fallback
    window: int | None = None  # the per-algorithm searchable rolling-window length actually used (§3.2)
    holdout_metrics: dict | None = None  # final performance on the untouched held-out period (§3.2)
    elimination_log: dict | None = None  # eliminated_mase/eliminated_bias/ranked, for human review (§3.6)
    version: int | None = None  # registry version this entry is (filled by the registry)
    mlflow_run_id: str | None = None
    pooled_group: str | None = None  # G-03: id of the pooled training run whose SHARED artifact this entry points to
    pooled_segments: list | None = None  # G-03: every segment covered by that shared artifact


class ModelRegistry:
    def __init__(self, root: str | Path, mlflow_backend=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry_index.json"
        self._lock = threading.RLock()
        self.backend = mlflow_backend  # optional MLflowLogger
        self._index: dict[str, dict] = {}
        if self._index_path.exists():
            self._index = self._migrate(json.loads(self._index_path.read_text()))

    # ---- persistence -------------------------------------------------------------------
    @staticmethod
    def _migrate(raw: dict) -> dict:
        """Older registries stored one flat entry per segment; wrap those as version 1 / Production."""
        out = {}
        for seg, val in raw.items():
            if "versions" in val:
                out[seg] = val
            else:
                val = {**val, "version": 1}
                out[seg] = {"active_version": 1, "needs_revalidation": False,
                            "versions": [{"version": 1, "stage": "Production", "entry": val,
                                          "created_at": None, "mlflow_version": None, "history": []}]}
        return out

    def _flush(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2, default=str))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- paths -------------------------------------------------------------------------
    def next_version(self, segment_id: str) -> int:
        with self._lock:
            versions = self._index.get(segment_id, {}).get("versions", [])
            return max((v["version"] for v in versions), default=0) + 1

    def artifact_dir(self, segment_id: str, algorithm_name: str, version: int | None = None) -> Path:
        d = self.root / segment_id / algorithm_name / (f"v{version}" if version is not None else "")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pooled_artifact_dir(self, group_id: str, algorithm_name: str) -> Path:
        """One directory per (pooled run, algorithm), shared by every segment that model serves (G-03)."""
        d = self.root / "_pooled" / group_id / algorithm_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_referenced(self, artifact_path: str | Path) -> bool:
        """True if any registered version (any stage, any segment) points at this artifact."""
        target = str(artifact_path)
        with self._lock:
            return any(v["entry"]["artifact_path"] == target
                       for seg in self._index.values() for v in seg["versions"])

    def segments_sharing(self, artifact_path: str | Path) -> list[str]:
        target = str(artifact_path)
        with self._lock:
            return sorted(sid for sid, seg in self._index.items()
                          if any(v["entry"]["artifact_path"] == target for v in seg["versions"]))

    # ---- lifecycle -----------------------------------------------------------------------
    def _seg(self, segment_id: str) -> dict:
        return self._index.setdefault(segment_id, {"active_version": None, "needs_revalidation": False, "versions": []})

    def _set_stage(self, segment_id: str, v: dict, stage: str, reason: str = "") -> None:
        assert stage in STAGES, stage
        prev = v["stage"]
        v["stage"] = stage
        v.setdefault("history", []).append({"at": self._now(), "from": prev, "to": stage, "reason": reason})
        if self.backend is not None and v.get("mlflow_version") is not None:
            self.backend.transition_stage(segment_id, v["mlflow_version"], stage)

    def register_version(self, entry: RegistryEntry, run_id: str | None = None) -> int:
        """Creates the next version (stage None), registers it in MLflow when enabled."""
        with self._lock:
            seg = self._seg(entry.segment_id)
            version = self.next_version(entry.segment_id)
            entry.version = version
            entry.mlflow_run_id = run_id
            mlflow_version = None
            if self.backend is not None and run_id:
                mlflow_version = self.backend.register_version(
                    entry.segment_id, run_id, Path(entry.artifact_path).parent,
                    tags={"algorithm": entry.algorithm_name, "rule_version": entry.rule_version, "registry_version": version},
                )
            seg["versions"].append({"version": version, "stage": "None", "entry": asdict(entry), "created_at": self._now(),
                                    "mlflow_version": mlflow_version, "history": []})
            self._flush()
            return version

    def promote(self, segment_id: str, version: int, reason: str = "new winner") -> None:
        """None -> Staging -> Production; the previous Production version is Archived."""
        with self._lock:
            seg = self._seg(segment_id)
            target = next(v for v in seg["versions"] if v["version"] == version)
            self._set_stage(segment_id, target, "Staging", reason)
            for v in seg["versions"]:
                if v["stage"] == "Production" and v["version"] != version:
                    self._set_stage(segment_id, v, "Archived", f"superseded by v{version}")
            self._set_stage(segment_id, target, "Production", reason)
            seg["active_version"] = version
            seg["needs_revalidation"] = False
            for prop in seg.get("proposals", []):
                if prop["status"] == "pending":
                    prop["status"] = "superseded"
                    prop["decision"] = {"by": "system", "at": self._now(), "note": f"Production moved to v{version}"}
            self._flush()

    def record_winner(self, entry: RegistryEntry, run_id: str | None = None) -> int:
        """Register + promote in one step (what full_train does for a new winner)."""
        v = self.register_version(entry, run_id)
        self.promote(entry.segment_id, v)
        return v

    def rollback(self, segment_id: str, reason: str) -> RegistryEntry | None:
        """
        Archive the failing Production version, restore the most recent previously-
        Production version, flag the segment for the next revalidation. Returns the
        restored entry, or None when there is no earlier version to fall back to
        (the segment is still flagged).
        """
        with self._lock:
            seg = self._seg(segment_id)
            current = next((v for v in seg["versions"] if v["stage"] == "Production"), None)
            seg["needs_revalidation"] = True
            if current is None:
                self._flush()
                return None
            prior = self._prior_of(seg, current)
            self._set_stage(segment_id, current, "Archived", f"rolled back: {reason}")
            current["rolled_back"] = True
            if prior is None:
                seg["active_version"] = None
                self._flush()
                return None
            self._set_stage(segment_id, prior, "Production", f"restored after rollback of v{current['version']}: {reason}")
            seg["active_version"] = prior["version"]
            self._flush()
            return RegistryEntry(**prior["entry"])

    @staticmethod
    def _prior_of(seg: dict, current: dict) -> dict | None:
        """Most recent earlier version that was once Production and was not itself rolled back."""
        for v in sorted(seg["versions"], key=lambda x: x["version"], reverse=True):
            if (v["version"] < current["version"] and not v.get("rolled_back")
                    and any(h["to"] == "Production" for h in v.get("history", []))):
                return v
        return None

    def prior_entry(self, segment_id: str) -> RegistryEntry | None:
        """The version a rollback of the current Production version would restore (None if there is none)."""
        with self._lock:
            seg = self._index.get(segment_id)
            if not seg:
                return None
            current = next((v for v in seg["versions"] if v["stage"] == "Production"), None)
            prior = None if current is None else self._prior_of(seg, current)
            return None if prior is None else RegistryEntry(**prior["entry"])

    # ---- human-reviewed rollback (G-01 decision) ------------------------------------------------
    def propose_rollback(self, segment_id: str, evidence: dict, reason: str) -> dict:
        """
        Records a PENDING proposal to roll back Production -> prior version. Idempotent: an already-pending
        proposal for the same (production, prior) pair is returned (evidence refreshed), not duplicated.
        Changes NO stage and does not flag the segment: only a reviewer's approval does.
        """
        with self._lock:
            seg = self._seg(segment_id)
            current = next((v for v in seg["versions"] if v["stage"] == "Production"), None)
            prior = None if current is None else self._prior_of(seg, current)
            if current is None or prior is None:
                raise RuntimeError(f"segment {segment_id}: no Production version with a prior version to propose a rollback to")
            props = seg.setdefault("proposals", [])
            for p in props:
                if p["status"] == "pending" and p["production_version"] == current["version"] and p["prior_version"] == prior["version"]:
                    p["evidence"], p["last_seen_at"] = evidence, self._now()
                    self._flush()
                    return p
            prop = {"id": f"{segment_id}#rb{len(props) + 1}", "segment_id": segment_id, "status": "pending",
                    "production_version": current["version"], "prior_version": prior["version"], "reason": reason,
                    "evidence": evidence, "created_at": self._now(), "last_seen_at": self._now(), "decision": None}
            props.append(prop)
            self._flush()
            return prop

    def proposals(self, segment_id: str) -> list[dict]:
        """Every rollback proposal ever recorded for the segment (pending, approved, rejected, superseded)."""
        with self._lock:
            return list(self._index.get(segment_id, {}).get("proposals", []))

    def pending_proposals(self, segment_id: str | None = None) -> list[dict]:
        with self._lock:
            segs = [segment_id] if segment_id else list(self._index)
            return [p for s in segs for p in self._index.get(s, {}).get("proposals", []) if p["status"] == "pending"]

    def decide_rollback(self, segment_id: str, proposal_id: str, approve: bool, reviewer: str, note: str = "") -> RegistryEntry | None:
        """
        Human decision on a pending proposal. `reviewer` is mandatory (audit trail). Approve executes
        rollback(); reject leaves Production as is. A proposal whose Production version is no longer the
        current one is stale and cannot be decided (it was superseded when Production moved).
        """
        if not reviewer or not reviewer.strip():
            raise ValueError("a named reviewer is required to decide a rollback proposal")
        with self._lock:
            seg = self._seg(segment_id)
            prop = next((p for p in seg.get("proposals", []) if p["id"] == proposal_id), None)
            if prop is None:
                raise KeyError(f"no proposal {proposal_id!r} for segment {segment_id}")
            if prop["status"] != "pending":
                raise ValueError(f"proposal {proposal_id} is already {prop['status']}")
            current = next((v for v in seg["versions"] if v["stage"] == "Production"), None)
            if current is None or current["version"] != prop["production_version"]:
                prop["status"] = "superseded"
                self._flush()
                raise ValueError(f"proposal {proposal_id} is stale: Production is no longer v{prop['production_version']}")
            prop["decision"] = {"by": reviewer.strip(), "at": self._now(), "note": note, "approved": bool(approve)}
            if not approve:
                prop["status"] = "rejected"
                self._flush()
                return None
            prop["status"] = "approved"
            return self.rollback(segment_id, f"approved by {reviewer.strip()} ({proposal_id}): {prop['reason']}")

    def flag_for_revalidation(self, segment_id: str, flag: bool = True) -> None:
        with self._lock:
            self._seg(segment_id)["needs_revalidation"] = flag
            self._flush()

    # ---- queries ---------------------------------------------------------------------------
    def get_active(self, segment_id: str) -> RegistryEntry | None:
        seg = self._index.get(segment_id)
        if not seg:
            return None
        v = next((x for x in seg["versions"] if x["stage"] == "Production"), None)
        return None if v is None else RegistryEntry(**v["entry"])

    def needs_revalidation(self, segment_id: str) -> bool:
        return bool(self._index.get(segment_id, {}).get("needs_revalidation", False))

    def versions(self, segment_id: str) -> list[dict]:
        return [{"version": v["version"], "stage": v["stage"], "algorithm": v["entry"]["algorithm_name"],
                 "created_at": v["created_at"], "mlflow_version": v.get("mlflow_version"), "history": v.get("history", [])}
                for v in self._index.get(segment_id, {}).get("versions", [])]

    def all_segments(self) -> list[str]:
        return list(self._index.keys())
