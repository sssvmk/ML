"""G-27 DeepVAR acceptance, the joint-inference interface (decision: pool-level infer_pool) and G-42 Layers 1 and 3."""
import json
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
sys.path.insert(0, "tests")
import admission
from test_deepstate_g26_g42 import _pool, _noisy_pool
from algorithms.deepvar import DeepVARModule, DeepVARNet, lowrank_gaussian_logpdf
from pooling import assemble_pool

HP = {"horizon": 4, "context_length": 40, "max_epochs": 20, "steps_per_epoch": 20, "batch_size": 16, "hidden_size": 16, "rank": 2,
      "learning_rate": 1e-2}


def _dependent_pool(n_days=500):
    """Series 1 = 1.2 x series 0 + small noise (dependent); series 2 and 3 independent of everything."""
    pool = _pool(n_days)
    ids = list(pool)
    a = pool[ids[0]]; ea = a[a["series_role"] == "endogenous"].sort_values("date")
    b = pool[ids[1]].copy(); nb = b[b["series_role"] == "endogenous"].sort_values("date").index
    b.loc[nb, "value"] = 1.2 * ea["value"].to_numpy() + np.random.default_rng(5).normal(0, 300, len(nb))
    pool[ids[1]] = b
    return pool


@pytest.fixture(scope="module")
def pool():
    return _dependent_pool()


@pytest.fixture(scope="module")
def m(pool):
    x = DeepVARModule(dict(HP)); x.train_pooled(assemble_pool(pool)); return x


# ------------------------------------------------------------------------------------------ Layer 1
def test_layer1_density_equals_scipy_and_torch_distributions():
    r = admission.layer1_deepvar()
    assert r["passed"] and r["dtype"] == "float64" and r["trained"] is False and r["cases"] == 27
    assert all(v <= admission.REL_TOL for v in r["worst_relative_difference"].values())


def test_layer1_check_can_fail_a_broken_density():
    bad = admission.layer1_deepvar(seeds=(0,), dims=(3,), ranks=(2,), _sabotage=1e-3)
    assert not bad["passed"] and max(bad["worst_relative_difference"].values()) > admission.REL_TOL


def test_density_special_cases():
    x = torch.zeros(1, 2, dtype=torch.float64)
    d = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    V = torch.zeros(1, 2, 1, dtype=torch.float64)
    assert torch.allclose(lowrank_gaussian_logpdf(x, x, V, d), torch.tensor([-np.log(2 * np.pi)], dtype=torch.float64))  # standard 2-d normal at its mean
    V1 = torch.tensor([[[1.0], [1.0]]], dtype=torch.float64)                                                              # rank-1 coupling of the two dims
    c = lowrank_gaussian_logpdf(torch.tensor([[1.0, 1.0]], dtype=torch.float64), x, V1, d)
    u = lowrank_gaussian_logpdf(torch.tensor([[1.0, -1.0]], dtype=torch.float64), x, V1, d)
    assert c > u                                                                                                          # moving together is likelier than apart


# ------------------------------------------------------------------------------------------ model and interface
def test_is_the_native_joint_model_and_not_the_old_stand_in(m):
    from algorithms._neural_template import NeuralLagModule
    assert not isinstance(m, NeuralLagModule) and DeepVARModule.requires_admission and DeepVARModule.joint_inference
    net = m.network()
    assert isinstance(net, DeepVARNet) and any(isinstance(x, nn.LSTM) for x in net.modules())
    assert net.head_mu.out_features == 4 and net.head_d.out_features == 4 and net.head_v.out_features == 4 * 2


def test_eligibility_is_derived_from_the_pool(pool):
    b = assemble_pool({k: _pool(1100)[k] for k in pool})
    assert DeepVARModule({}).check_pooled_eligibility(b).eligible
    assert not DeepVARModule({}).check_pooled_eligibility(assemble_pool({k: pool[k] for k in list(pool)[:1]})).eligible
    small = assemble_pool({k: v[v["date"] < v["date"].min() + np.timedelta64(100, "D")] for k, v in list(pool.items())[:2]})
    assert "1000" in DeepVARModule({}).check_pooled_eligibility(small).reason
    assert not DeepVARModule({}).check_eligibility(next(iter(pool.values()))).eligible


def test_single_segment_paths_refuse_and_the_pool_call_needs_every_series(m, pool):
    ids = list(pool)
    with pytest.raises(RuntimeError, match="infer_pool"):
        m.infer(pool[ids[0]], 4)
    with pytest.raises(RuntimeError, match="pooled joint model"):
        DeepVARModule({}).train(pool[ids[0]])
    with pytest.raises(ValueError, match="missing"):
        m.infer_pool({k: pool[k] for k in ids[:3]}, 4)
    shifted = dict(pool)
    df = shifted[ids[0]]
    shifted[ids[0]] = df[df["date"] < df["date"].max()]                          # this series ends one day earlier
    with pytest.raises(ValueError, match="same date"):
        m.infer_pool(shifted, 4)


def test_learns_cross_series_dependence_and_scales_windows_so_validation_tracks_training(m):
    pool = _dependent_pool()
    C = m.cross_series_correlation(pool, 4, n=1500)
    assert C.iloc[0, 1] > 0.5 and C.iloc[0, 1] - C.iloc[2, 3] > 0.4              # the truly dependent pair is coupled, an independent pair is not
    h = m._fitted_model["early_stopping"]["history"][-1]
    assert h["val_nll"] < h["train_nll"] + 3.0                                    # no large generalisation gap (per-window scaling)


def test_forecast_of_one_series_conditions_on_the_others(m, pool):
    ids = list(pool)
    base = m.infer_pool(pool, 4)[ids[0]]["forecast"].to_numpy()
    edited = dict(pool)
    df = edited[ids[1]].copy()
    tail = sorted(df["date"].unique())[-10:]
    mk = (df["series_role"] == "endogenous") & df["date"].isin(tail)
    df.loc[mk, "value"] = df.loc[mk, "value"].to_numpy()[::-1]                     # reshuffle the LAST 10 days of a DIFFERENT series
    edited[ids[1]] = df
    assert not np.allclose(base, m.infer_pool(edited, 4)[ids[0]]["forecast"].to_numpy())
    old = dict(pool)
    d0 = old[ids[1]].copy()
    first = sorted(d0["date"].unique())[:50]
    d0.loc[(d0["series_role"] == "endogenous") & d0["date"].isin(first), "value"] += 1e6   # far older than the 40-period context
    old[ids[1]] = d0
    assert np.allclose(base, m.infer_pool(old, 4)[ids[0]]["forecast"].to_numpy())


def test_sample_paths_are_joint_seeded_and_the_median_is_the_forecast(m, pool):
    p1 = m.sample_paths(pool, 6, n=200, seed=3)
    assert p1.shape == (200, 6, 4)
    assert np.allclose(p1, m.sample_paths(pool, 6, n=200, seed=3)) and not np.allclose(p1, m.sample_paths(pool, 6, n=200, seed=4))
    fc = m.infer_pool(pool, 6)
    ids = m._meta["series"]
    med = np.median(m.sample_paths(pool, 6), axis=0)
    for i, sid in enumerate(ids):
        assert np.allclose(fc[sid]["forecast"], med[:, i])
    level = {sid: df[df["series_role"] == "endogenous"]["value"].tail(28).mean() for sid, df in pool.items()}
    assert all(0.75 < fc[sid]["forecast"].mean() / level[sid] < 1.25 for sid in fc)     # each series at its own scale, no FX


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons_exact_dated_rows(m, pool, h):
    out = m.infer_pool(pool, h)
    for sid, df in pool.items():
        fc = out[sid]
        assert len(fc) == h and np.isfinite(fc["forecast"]).all()
        assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df[df["series_role"] == "endogenous"]["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_save_load_identical(m, pool, tmp_path):
    m.save(tmp_path / "p" / "model.pkl")
    l = DeepVARModule({}); l.load(tmp_path / "p" / "model.pkl")
    a, b = m.infer_pool(pool, 4), l.infer_pool(pool, 4)
    assert all(np.allclose(a[s]["forecast"], b[s]["forecast"], rtol=1e-6) for s in a)


# ------------------------------------------------------------------------------------------ Layer 3 and the pipeline
def test_layer3_passes_for_a_joint_model():
    make = lambda **o: DeepVARModule({**HP, "max_epochs": 6, "steps_per_epoch": 10, **o})
    r = admission.layer3(make, lambda: _pool(400), lambda: _noisy_pool(400), None)
    assert r["passed"], [c for c in r["checks"] if not c["passed"]]
    na = next(c for c in r["checks"] if c["check"].startswith("changing future"))
    assert na["applicable"] is False                                              # DeepVAR has no future-known inputs


def _admitted_cfg(tmp_path, algos):
    from config import load_config
    cfg = load_config("config.json")
    cfg["admission"]["reports_dir"] = str(tmp_path / "adm")
    (tmp_path / "adm").mkdir(exist_ok=True)
    (tmp_path / "adm" / "deepvar.json").write_text(json.dumps({"module_sha256": admission.module_source_hash(DeepVARModule), "admitted": True}))
    cfg["algorithms"] = algos
    return cfg


def _orch(tmp_path, cfg):
    from config import build_candidates
    from orchestrator import Orchestrator
    from registry import ModelRegistry
    return Orchestrator(ModelRegistry(tmp_path / "reg"), tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1, holdout_periods=20,
                        window_search_enabled=False, backtest_step=100, hyperparameter_search={"enabled": False}, mlflow_config=None)


def test_pooled_pipeline_uses_one_pool_wide_call_in_backtest_and_holdout(tmp_path):
    cfg = _admitted_cfg(tmp_path, {"deepvar": {"enabled": True, "hyperparameters": {**HP, "max_epochs": 5, "steps_per_epoch": 6}}})
    orch = _orch(tmp_path, cfg)
    calls = {"pool": 0, "single": 0}
    orig_pool, orig_single = DeepVARModule.infer_pool, DeepVARModule.infer
    def spy_pool(self, segments, horizon):
        calls["pool"] += 1
        return orig_pool(self, segments, horizon)
    def spy_single(self, df, h):
        calls["single"] += 1
        return orig_single(self, df, h)
    DeepVARModule.infer_pool, DeepVARModule.infer = spy_pool, spy_single
    try:
        segs = _pool(1100)
        entries = orch.full_train_pooled(segs, horizon=4, rule_version="r1")
    finally:
        DeepVARModule.infer_pool, DeepVARModule.infer = orig_pool, orig_single
    assert set(entries) == set(segs) and calls["single"] == 0 and calls["pool"] >= 2     # backtest fold(s) + holdout, never per segment
    for e in entries.values():
        assert not e.elimination_log["eliminated_error"], e.elimination_log


def test_daily_inference_is_pool_aware_for_a_joint_production_model(tmp_path, m, pool):
    from registry import RegistryEntry
    cfg = _admitted_cfg(tmp_path, {"deepvar": {"enabled": True, "hyperparameters": dict(HP)}})
    orch = _orch(tmp_path, cfg)
    ids = list(pool)
    art = tmp_path / "shared" / "model.pkl"
    m.save(art)
    for sid in ids:                                                               # register the shared artifact as every segment's Production model
        v = orch.registry.register_version(RegistryEntry(segment_id=sid, algorithm_name="deepvar", is_fallback=False, hyperparameters=dict(HP),
                                                        artifact_path=str(art), rule_version="r1", metrics={}, pooled_group="g", pooled_segments=ids))
        orch.registry.promote(sid, v)
    with pytest.raises(RuntimeError, match="daily_infer_pool"):
        orch.daily_infer(ids[0], pool[ids[0]], 4)                                  # one segment cannot serve a joint model
    out = orch.daily_infer_pool(pool, 4)
    assert set(out) == set(ids) and all(len(out[s]) == 4 for s in ids)
    direct = m.infer_pool(pool, 4)
    assert all(np.allclose(out[s]["forecast"], direct[s]["forecast"]) for s in ids)
    with pytest.raises(ValueError, match="missing rows"):
        orch.daily_infer_pool({ids[0]: pool[ids[0]]}, 4)                           # the other pooled series are required
