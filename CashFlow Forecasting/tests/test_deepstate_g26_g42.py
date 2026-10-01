"""G-26 DeepState acceptance + G-42 admission gating (Layers 1 and 3; Layer 2 needs an approved calibration)."""
import json
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import admission
from adapters.synthetic import SyntheticAdapter
from algorithms.deepstate import DeepStateModule, DeepStateNet, build_structure, kalman_loglik
from pooling import assemble_pool

HP = {"horizon": 4, "context_length": 40, "max_epochs": 6, "steps_per_epoch": 10, "batch_size": 16, "hidden_size": 16}


def _pool(n_days=400, future_days=0, seeds=(1, 2, 3, 4)):
    spec = [("1000", "INR", "AR", seeds[0], 1.0), ("1000", "INR", "AP", seeds[1], 1.0), ("2000", "EUR", "AR", seeds[2], 0.011), ("2000", "JPY", "AP", seeds[3], 100.0)]
    out = {}
    for cc, ccy, proc, seed, mult in spec:
        d = SyntheticAdapter().extract(company_code=cc, currency=ccy, process=proc, n_days=n_days, seed=seed, future_days=future_days)
        d["value"] = d["value"] * mult
        out[d["segment_id"].iloc[0]] = d
    return out


def _noisy_pool(n_days=400):
    out = _pool(n_days)
    rng = np.random.default_rng(0)
    for df in out.values():
        m = df["series_role"] == "endogenous"
        df.loc[m, "value"] = 1000.0 + rng.normal(0, 100, int(m.sum()))
    return out


@pytest.fixture(scope="module")
def pool():
    return _pool()


@pytest.fixture(scope="module")
def m(pool):
    x = DeepStateModule(dict(HP)); x.train_pooled(assemble_pool(pool)); return x


# ------------------------------------------------------------------------------------------ Layer 1
def test_layer1_kalman_equals_the_independent_implementation():
    r = admission.layer1_deepstate()
    assert r["passed"] and r["dtype"] == "float64" and r["trained"] is False
    assert all(v <= admission.REL_TOL for v in r["worst_relative_difference"].values()), r["worst_relative_difference"]
    assert {"loglik", "state", "cov", "forecast_mean", "forecast_var"} == set(r["worst_relative_difference"])


def test_layer1_check_can_fail_a_broken_filter():
    bad = admission.layer1_deepstate(seeds=(0,), periods_options=((7,),), _sabotage=1e-3)
    assert not bad["passed"] and max(bad["worst_relative_difference"].values()) > admission.REL_TOL


# ------------------------------------------------------------------------------------------ model structure / filter
def test_structure_level_trend_and_periodic_seasonality():
    F, a, names = build_structure([7], torch.float64)
    assert F.shape == (8, 8) and names[:2] == ["level", "trend"] and len(names) == 8
    s = torch.zeros(8, dtype=torch.float64); s[2:] = torch.tensor([0.5, -0.2, 0.1, 0.3, -0.4, 0.2], dtype=torch.float64)
    obs = []
    for _ in range(21):
        obs.append(float(a @ s)); s = F @ s
    assert np.allclose(obs[:7], obs[7:14]) and np.allclose(obs[:7], obs[14:])          # period 7 (level = trend = 0)
    F2, _, n2 = build_structure([7, 30], torch.float64)
    assert F2.shape[0] == 2 + 6 + 29


def test_kalman_is_differentiable_and_the_mask_is_a_pure_skip():
    F, a, _ = build_structure([7], torch.float64)
    B, T, n = 2, 30, F.shape[0]
    g = torch.full((B, T, n), 0.05, dtype=torch.float64, requires_grad=True)
    sg = torch.full((B, T), 0.3, dtype=torch.float64); b = torch.zeros(B, T, dtype=torch.float64)
    z = torch.randn(B, T, dtype=torch.float64); m0 = torch.zeros(B, n, dtype=torch.float64); P0 = torch.eye(n, dtype=torch.float64)[None].repeat(B, 1, 1)
    ll = kalman_loglik(z, F, a, g, sg, b, m0, P0)
    ll.sum().backward()
    assert torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    full = torch.ones(B, T, dtype=torch.bool)
    assert torch.allclose(kalman_loglik(z, F, a, g.detach(), sg, b, m0, P0, mask=full), ll.detach())
    part = full.clone(); part[:, 10] = False
    assert kalman_loglik(z, F, a, g.detach(), sg, b, m0, P0, mask=part).sum() != ll.sum()   # a missing observation contributes nothing


def test_is_the_native_model_not_deepar_or_a_stand_in(m):
    from algorithms.deepar import DeepARModule
    from algorithms.nf_adapter import NeuralForecastModule
    assert not isinstance(m, (DeepARModule, NeuralForecastModule)) and DeepStateModule.requires_admission
    net = m.network()
    assert isinstance(net, DeepStateNet) and any(isinstance(x, nn.LSTM) for x in net.modules())
    assert "sklearn" not in type(net).__module__


def test_one_shared_network_serves_any_number_of_series(pool):
    two = DeepStateModule(dict(HP)); two.train_pooled(assemble_pool({k: pool[k] for k in list(pool)[:2]}))
    four = DeepStateModule(dict(HP)); four.train_pooled(assemble_pool(pool))
    assert four._fitted_model["pooled"]["n_series"] == 4 and abs(four.n_parameters() - two.n_parameters()) < 500   # only the static embedding grows
    assert four.n_parameters() < 3 * two.n_parameters()


def test_parameters_are_positive_and_forecasts_live_at_each_series_own_scale(m, pool):
    for sid, df in pool.items():
        fc = m.infer(df, 4)
        level = df[df["series_role"] == "endogenous"]["value"].tail(28).mean()
        assert len(fc) == 4 and 0.7 < fc["forecast"].mean() / level < 1.3, sid
    sid, df = next(iter(pool.items()))
    _, _, _, _, (m1, P1, g, sigma, b), _ = m._filter_and_predict(df, 6)
    assert (g > 0).all() and (sigma > 0).all()


def test_intervals_widen_with_the_horizon_and_sample_paths_agree_with_the_analytic_mean(m, pool):
    df = next(iter(pool.values()))
    q = m.infer_quantiles(df, 12)
    assert (q["lo-80"] < q["mean"]).all() and (q["mean"] < q["hi-80"]).all() and (q["lo-90"] < q["lo-80"]).all()
    assert q["std"].iloc[-1] > q["std"].iloc[0]                                    # state uncertainty grows with the horizon
    paths = m.sample_paths(df, 6, n=4000, seed=3)
    assert paths.shape == (4000, 6)
    se = paths.std(axis=0) / np.sqrt(4000)
    assert np.all(np.abs(paths.mean(axis=0) - q["mean"].to_numpy()[:6]) < 5 * se + 1e-6)
    assert np.allclose(paths.std(axis=0), q["std"].to_numpy()[:6], rtol=0.06)       # sampled spread matches the analytic std


def test_filtered_state_tensors_are_exposed(m, pool):
    st = m.filtered_state(next(iter(pool.values())))
    assert st["names"][:2] == ["level", "trend"] and len(st["mean"]) == len(st["variance"]) == 8 and (st["variance"] > 0).all()


def test_training_records_a_like_for_like_loss_and_restores_the_best_weights(m):
    es = m._fitted_model["early_stopping"]
    assert es["used"] and es["restored_best_checkpoint"] and es["history"][0]["train_nll"] > es["history"][-1]["train_nll"]
    assert {"train_nll", "val_nll"} <= set(es["history"][0]) and "negative log-likelihood" in es["loss_units"]


def test_eligibility_is_derived_from_the_pool(pool):
    b = assemble_pool({k: _pool(1100)[k] for k in pool})
    assert DeepStateModule({}).check_pooled_eligibility(b).eligible
    one = assemble_pool({k: pool[k] for k in list(pool)[:1]})
    assert not DeepStateModule({}).check_pooled_eligibility(one).eligible
    small = assemble_pool({k: v[v["date"] < v["date"].min() + np.timedelta64(100, "D")] for k, v in list(pool.items())[:2]})
    assert "1000" in DeepStateModule({}).check_pooled_eligibility(small).reason
    assert not DeepStateModule({}).check_eligibility(next(iter(pool.values()))).eligible
    with pytest.raises(RuntimeError, match="pooled"):
        DeepStateModule({}).train(next(iter(pool.values())))


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons(m, pool, h):
    df = next(iter(pool.values()))
    fc = m.infer(df, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_save_load_unseen_series_and_attribute_checks(m, pool, tmp_path):
    m.save(tmp_path / "p" / "model.pkl")
    l = DeepStateModule({}); l.load(tmp_path / "p" / "model.pkl")
    df = next(iter(pool.values()))
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"], rtol=1e-6)
    new = df.copy(); new["segment_id"] = "9999-INR-inflow"
    assert len(l.infer(new, 4)) == 4
    alien = new.copy(); alien["currency"] = "CHF"
    with pytest.raises(ValueError, match="unseen category"):
        l.infer(alien, 4)
    changed = df.copy(); changed["currency"] = "CHF"
    with pytest.raises(ValueError, match="changed since pooled training"):
        l.infer(changed, 4)


def test_future_known_covariates_are_used_and_causal():
    fk_pool = _pool(400, future_days=10)
    x = DeepStateModule({**HP}); x.train_pooled(assemble_pool(fk_pool))
    assert x._meta["n_fk"] == 1 and x._meta["n_cov"] == 7
    df = next(iter(fk_pool.values()))
    last = df[df["series_role"] == "endogenous"]["date"].max()
    base = x.infer(df, 8)["forecast"].to_numpy()
    edited = df.copy()
    mk = (edited["series_role"] == "exogenous") & (edited["date"] >= last + pd.Timedelta(days=6))
    edited.loc[mk, "value"] = edited.loc[mk, "value"] * 5 + 100
    alt = x.infer(edited, 8)["forecast"].to_numpy()
    assert np.allclose(base[:5], alt[:5], atol=1e-9)                                   # earlier steps unchanged
    assert not np.allclose(base[5:], alt[5:])                                          # later steps do see the covariate
    with pytest.raises(ValueError, match="future-known"):
        x.infer(df, 20)                                                                # only 10 future values exist


# ------------------------------------------------------------------------------------------ Layer 3 + admission gating
def _factories():
    make = lambda **o: DeepStateModule({**HP, **o})
    return make, (lambda: _pool()), (lambda: _noisy_pool()), (lambda: _pool(400, future_days=10))


def test_layer3_passes_all_checks():
    make, pf, nf, fkf = _factories()
    r = admission.layer3(make, pf, nf, fkf)
    assert r["passed"], [c for c in r["checks"] if not c["passed"]]
    names = {c["check"] for c in r["checks"]}
    assert {"same-seed reproducibility", "save -> load -> infer identical", "no NaN / inf in forecasts", "training loss decreases",
            "early stopping triggers on a noisy series", "changing future covariates does not change earlier steps"} <= names
    fkc = next(c for c in r["checks"] if c["check"].startswith("changing future"))
    assert fkc["applicable"] and fkc["later_steps_changed"]


def test_layer2_without_an_approved_calibration_blocks_admission(tmp_path):
    from config import load_config, build_candidates
    make, pf, nf, fkf = _factories()
    l1 = admission.layer1_deepstate(seeds=(0,), periods_options=((7,),))
    l3 = {"passed": True, "checks": []}                                   # what Layer 3 would say if it passed
    rep = admission.run_admission("deepstate", DeepStateModule, l1, l3, calibration=None, reports_dir=tmp_path)
    assert rep["layer2"]["status"] == "not_run" and rep["admitted"] is False
    cfg = load_config("config.json"); cfg["admission"]["reports_dir"] = str(tmp_path)
    assert "deepstate" not in build_candidates(cfg)                       # not admitted -> not a candidate
    assert admission.is_admitted("deepstate", DeepStateModule, cfg)[0] is False


def test_only_a_matching_admitted_report_lets_the_model_compete(tmp_path):
    from config import load_config, build_candidates
    cfg = load_config("config.json"); cfg["admission"]["reports_dir"] = str(tmp_path)
    good = {"algorithm": "deepstate", "module_sha256": admission.module_source_hash(DeepStateModule), "admitted": True,
            "layer1": {"passed": True}, "layer2": {"passed": True}, "layer3": {"passed": True}}
    (tmp_path / "deepstate.json").write_text(json.dumps(good))
    assert "deepstate" in build_candidates(cfg)
    (tmp_path / "deepstate.json").write_text(json.dumps({**good, "module_sha256": "0" * 64}))
    ok, why = admission.is_admitted("deepstate", DeepStateModule, cfg)
    assert not ok and "changed" in why and "deepstate" not in build_candidates(cfg)


def test_admitted_model_runs_through_the_pooled_pipeline(tmp_path):
    from config import load_config, build_candidates
    from orchestrator import Orchestrator
    from registry import ModelRegistry
    cfg = load_config("config.json"); cfg["admission"]["reports_dir"] = str(tmp_path / "adm")
    (tmp_path / "adm").mkdir()
    (tmp_path / "adm" / "deepstate.json").write_text(json.dumps({"module_sha256": admission.module_source_hash(DeepStateModule), "admitted": True}))
    cfg["algorithms"] = {"deepstate": {"enabled": True, "hyperparameters": {**HP, "steps_per_epoch": 6, "max_epochs": 4}}}
    orch = Orchestrator(ModelRegistry(tmp_path / "reg"), tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1, holdout_periods=20,
                        window_search_enabled=False, backtest_step=100, hyperparameter_search={"enabled": False}, mlflow_config=None)
    segs = _pool(1100)
    entries = orch.full_train_pooled(segs, horizon=4, rule_version="r1")
    assert set(entries) == set(segs)
    for e in entries.values():
        assert not e.elimination_log["eliminated_error"], e.elimination_log
