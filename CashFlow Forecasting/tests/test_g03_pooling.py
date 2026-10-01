"""G-03 acceptance: one model instance trained across >= 2 segments in one call; registry maps each segment
to the shared artifact; eligibility derived from the series actually supplied."""
import numpy as np
import pytest
from adapters.synthetic import SyntheticAdapter
from algorithms.deepar import DeepARModule
from algorithms.deepvar import DeepVARModule
from algorithms.tft import TFTModule
from config import load_config, build_candidates
from orchestrator import Orchestrator
from pooling import assemble_pool
from registry import ModelRegistry


def _segments(n_days=600):
    """4 segments: 2 entities x currencies x both directions; the JPY-like one is ~100x larger in raw amounts."""
    spec = [("1000", "INR", "AR", 1, 1.0), ("1000", "INR", "AP", 2, 1.0), ("2000", "EUR", "AR", 3, 0.011), ("2000", "JPY", "AP", 4, 100.0)]
    out = {}
    for cc, ccy, proc, seed, mult in spec:
        d = SyntheticAdapter().extract(company_code=cc, currency=ccy, process=proc, n_days=n_days, seed=seed)
        d["value"] = d["value"] * mult
        out[d["segment_id"].iloc[0]] = d
    return out


def test_batch_attributes_scales_and_derived_eligibility():
    segs = _segments()
    b = assemble_pool(segs)
    assert b.n_series == 4 and b.total_obs == 4 * 600
    # attributes are the three generic opaque contract fields — system has no concept of direction or inflow/outflow
    attrs = {s: (a["company_code"], a["currency"], a["dataset"]) for s, a in b.attributes.items()}
    assert all(k in ("company_code", "currency", "dataset") for k in list(b.attributes.values())[0])
    assert set(a["company_code"] for a in b.attributes.values()) == {"1000", "2000"}
    assert set(a["currency"] for a in b.attributes.values()) == {"INR", "EUR", "JPY"}
    cats = b.categories()
    assert set(cats.keys()) == {"company_code", "currency", "dataset"}   # no direction, no entity
    assert max(b.scales.values()) / min(b.scales.values()) > 1000        # scales really differ per series

    for cls in (DeepARModule, DeepVARModule):
        assert cls({}).check_pooled_eligibility(b).eligible
        one = assemble_pool({k: segs[k] for k in list(segs)[:1]})
        assert not cls({}).check_pooled_eligibility(one).eligible          # 1 series is not a pool
    small = assemble_pool({k: v[v["date"] < v["date"].min() + np.timedelta64(100, "D")] for k, v in list(segs.items())[:2]})
    e = DeepARModule({}).check_pooled_eligibility(small)
    assert not e.eligible and "1000" in e.reason                           # pooled N floor from the actual data
    # single-segment path: a segment alone is not a pool
    assert not DeepARModule({}).check_eligibility(next(iter(segs.values()))).eligible
    assert not hasattr(DeepARModule({}), "pooled_series_count")           # manual flag is gone


def test_one_call_trains_one_model_on_all_series_and_forecasts_each_at_its_own_scale():
    segs = _segments()
    m = DeepARModule({"n_lags": 14})
    m.train_pooled(assemble_pool(segs))                                    # ONE call, 4 series
    levels = {}
    for sid, df in segs.items():
        fc = m.infer(df, 4)
        assert len(fc) == 4 and fc["forecast"].notna().all()
        y = df[df["series_role"] == "endogenous"]["value"]
        levels[sid] = (fc["forecast"].mean(), y.tail(28).mean())
    for sid, (f, actual_level) in levels.items():                          # per-series scaling: forecast lives at its own scale
        assert 0.5 < f / actual_level < 1.5, (sid, f, actual_level)


def _orch(tmp_path, keep, step=25, window_search=False, hp_override=None):
    cfg = load_config("config.json")
    cfg["algorithms"] = {k: v for k, v in cfg["algorithms"].items() if k in keep}
    for k, hp in (hp_override or {}).items():
        cfg["algorithms"][k]["hyperparameters"].update(hp)
    reg = ModelRegistry(tmp_path / "reg")
    return Orchestrator(reg, tmp_path / "logs", build_candidates(cfg), max_parallel_workers=3, holdout_periods=20,
                        window_search_enabled=window_search, backtest_step=step, hyperparameter_search={"enabled": False},
                        mlflow_config=None), reg


def test_full_train_pooled_registry_maps_every_segment_to_the_shared_artifact(tmp_path):
    # 900 days per series and the window search ON with a coarse step: several rolling folds per candidate window, so the
    # MASE >= 1 elimination is decided by many folds instead of one 4-day fold (with the window search off the window equals
    # the whole history, which leaves exactly one fold however many days there are).
    orch, reg = _orch(tmp_path, ["deepar"], step=100, window_search=True, hp_override={"deepar": {"max_steps": 60}})
    segs = _segments(n_days=900)
    entries = orch.full_train_pooled(segs, horizon=4, rule_version="r1")
    assert set(entries) == set(segs)
    won = {s: e for s, e in entries.items() if e.algorithm_name == "deepar"}
    assert len(won) >= 2, {s: e.algorithm_name for s, e in entries.items()}     # deepar really competed and won
    paths = {e.artifact_path for e in won.values()}
    assert len(paths) == 1                                                   # ONE shared artifact
    path = paths.pop()
    assert reg.segments_sharing(path) == sorted(won)
    for e in won.values():
        assert e.pooled_group and sorted(e.pooled_segments) == sorted(segs)
        assert e.window and e.metrics.get("mase") is not None                # backtest metrics are per-segment
        assert reg.get_active(e.segment_id).artifact_path == path
    for sid, e in won.items():                                               # production inference uses the shared model
        fc = orch.daily_infer(sid, segs[sid], 4)
        assert len(fc) == 4
    # weekly revalidation over the same data keeps Production per segment (same algorithm, same rules)
    again = orch.weekly_revalidate_pooled(segs, horizon=4, rule_version="r1")
    assert {s: e.version for s, e in again.items() if s in won} == {s: e.version for s, e in won.items()}
    # every shared artifact still on disk is referenced by the registry (unreferenced fits are discarded)
    pooled_root = tmp_path / "reg" / "_pooled"
    for f in pooled_root.rglob("model.pkl"):
        assert reg.is_referenced(f)


def test_pooled_run_needs_at_least_two_segments(tmp_path):
    orch, _ = _orch(tmp_path, ["deepar"])
    one = dict(list(_segments().items())[:1])
    with pytest.raises(ValueError):
        orch.full_train_pooled(one)
