"""G-01: registry lifecycle (None->Staging->Production->Archived), MLflow mirror, and the human-reviewed rollback:
live six-metric rank-sum of Production vs the prior version -> PROPOSAL -> named reviewer approves/rejects."""
import numpy as np
import pytest
from adapters.synthetic import SyntheticAdapter
from registry import ModelRegistry
from orchestrator import Orchestrator
from config import load_config, build_candidates

KEEP = ["ridge", "lightgbm"]


def _orch(tmp_path, mlflow=True):
    cfg = load_config("config.json")
    cfg["algorithms"] = {k: v for k, v in cfg["algorithms"].items() if k in KEEP}
    cfg["algorithms"]["ridge"]["hyperparameters"]["n_lags"] = 7  # weekly seasonality needs 7 lags to beat seasonal-naive
    ml = {"enabled": mlflow, "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "experiment_name": "lc"}
    reg = ModelRegistry(tmp_path / "reg")
    orch = Orchestrator(reg, tmp_path / "logs", build_candidates(cfg), max_parallel_workers=3,
                        holdout_periods=20, window_search_enabled=False, backtest_step=60,
                        hyperparameter_search={"enabled": False}, mlflow_config=ml,
                        monitoring={"min_actuals": 2})
    return orch, reg


def _poisoned(df):
    """Same segment, but the target is replaced by zero-mean noise: a model trained on it forecasts ~0."""
    bad = df.copy()
    m = bad["series_role"] == "endogenous"
    bad.loc[m, "value"] = np.random.default_rng(0).normal(0, 2000, int(m.sum()))
    return bad


def _live(df, n_hist=890, horizon=4):
    dates = sorted(df["date"].unique())
    history = df[df["date"] <= dates[n_hist - 1]]
    act = df[(df["series_role"] == "endogenous") & (df["date"].isin(dates[n_hist:n_hist + horizon]))]
    return history, act.groupby("date")["value"].sum().reset_index()


def _two_versions(orch, reg):
    df = SyntheticAdapter().extract(n_days=900)
    seg = df["segment_id"].iloc[0]
    e1 = orch.full_train(seg, df, horizon=4, rule_version="r1")
    assert not e1.is_fallback, "test needs a real winner"
    orch.full_train(seg, _poisoned(df), horizon=4, rule_version="r2")          # v2 = clearly worse on live data
    assert {v["version"]: v["stage"] for v in reg.versions(seg)} == {1: "Archived", 2: "Production"}
    return df, seg


def test_promote_and_archive_mirrored_in_mlflow(tmp_path):
    orch, reg = _orch(tmp_path)
    df, seg = _two_versions(orch, reg)
    if orch.mlflow.active:
        assert orch.mlflow.stage_of(seg, 1) == "Archived" and orch.mlflow.stage_of(seg, 2) == "Production"


def test_worse_production_only_proposes_then_human_approves(tmp_path):
    orch, reg = _orch(tmp_path)
    df, seg = _two_versions(orch, reg)
    history, actuals = _live(df)

    res = orch.monitor_production(seg, actuals, history)
    assert res.action == "rollback_proposed" and res.prior_version == 1 and res.production_version == 2, res
    assert res.combined_rank["prior"] < res.combined_rank["production"]
    # NOTHING changed yet: a proposal is not a rollback
    assert {v["version"]: v["stage"] for v in reg.versions(seg)} == {1: "Archived", 2: "Production"}
    assert reg.get_active(seg).version == 2 and reg.needs_revalidation(seg) is False
    pend = orch.pending_rollback_proposals(seg)
    assert len(pend) == 1 and pend[0]["id"] == res.proposal_id
    assert {"production_metrics", "prior_metrics", "per_metric_rank", "combined_rank"} <= set(pend[0]["evidence"])

    # re-checking does not duplicate the pending proposal
    again = orch.monitor_production(seg, actuals, history)
    assert again.action == "proposal_pending" and again.proposal_id == res.proposal_id and len(orch.pending_rollback_proposals(seg)) == 1

    # a named reviewer is mandatory
    with pytest.raises(ValueError):
        orch.decide_rollback(seg, res.proposal_id, True, reviewer="  ")

    restored = orch.decide_rollback(seg, res.proposal_id, True, reviewer="muni", note="live accuracy clearly worse")
    assert restored.version == 1 and reg.get_active(seg).version == 1
    assert {v["version"]: v["stage"] for v in reg.versions(seg)} == {1: "Production", 2: "Archived"}
    assert reg.needs_revalidation(seg) is True
    p = reg.proposals(seg)[0]
    assert p["status"] == "approved" and p["decision"]["by"] == "muni"
    if orch.mlflow.active:
        assert orch.mlflow.stage_of(seg, 1) == "Production" and orch.mlflow.stage_of(seg, 2) == "Archived"
    with pytest.raises(ValueError):                                              # cannot decide twice
        orch.decide_rollback(seg, res.proposal_id, False, reviewer="muni")

    # flagged segment: weekly revalidation registers a new version even if the same algorithm wins
    e3 = orch.weekly_revalidate(seg, df, horizon=4, rule_version="r2")
    assert e3.version == 3 and reg.needs_revalidation(seg) is False


def test_rejected_proposal_leaves_production_and_is_not_repeated(tmp_path):
    orch, reg = _orch(tmp_path, mlflow=False)
    df, seg = _two_versions(orch, reg)
    history, actuals = _live(df)
    res = orch.monitor_production(seg, actuals, history)
    assert orch.decide_rollback(seg, res.proposal_id, False, reviewer="muni", note="known one-off") is None
    assert reg.get_active(seg).version == 2 and reg.needs_revalidation(seg) is False
    assert reg.proposals(seg)[0]["status"] == "rejected"
    later = orch.monitor_production(seg, actuals, history)
    assert later.action == "proposal_rejected" and orch.pending_rollback_proposals(seg) == []


def test_stale_proposal_is_superseded_when_production_moves(tmp_path):
    orch, reg = _orch(tmp_path, mlflow=False)
    df, seg = _two_versions(orch, reg)
    history, actuals = _live(df)
    res = orch.monitor_production(seg, actuals, history)
    orch.full_train(seg, df, horizon=4, rule_version="r3")                       # Production moves to v3
    assert reg.proposals(seg)[0]["status"] == "superseded" and orch.pending_rollback_proposals(seg) == []
    with pytest.raises(ValueError):
        orch.decide_rollback(seg, res.proposal_id, True, reviewer="muni")


def test_equal_or_better_production_proposes_nothing(tmp_path):
    orch, reg = _orch(tmp_path, mlflow=False)
    df = SyntheticAdapter().extract(n_days=900)
    seg = df["segment_id"].iloc[0]
    orch.full_train(seg, df, horizon=4, rule_version="r1")
    history, actuals = _live(df)
    assert orch.monitor_production(seg, actuals, history).action == "no_prior_version"   # single version
    orch.full_train(seg, df, horizon=4, rule_version="r2")                                # identical model -> tie
    res = orch.monitor_production(seg, actuals, history)
    assert res.action == "ok" and reg.pending_proposals(seg) == [] and reg.get_active(seg).version == 2


def test_weekly_same_winner_keeps_production(tmp_path):
    orch, reg = _orch(tmp_path, mlflow=False)
    df = SyntheticAdapter().extract(n_days=900)
    seg = df["segment_id"].iloc[0]
    orch.full_train(seg, df, horizon=4, rule_version="r1")
    e2 = orch.weekly_revalidate(seg, df, horizon=4, rule_version="r1")
    assert e2.version == 1 and len(reg.versions(seg)) == 1
