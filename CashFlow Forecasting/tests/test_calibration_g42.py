"""The calibration tooling (G-42 Layer 2 calibration run): scores, oracles and the statistics that turn results into proposals."""
import json
import math
import os
import numpy as np
import pytest
from scipy import stats
from calibration import sim, metrics, analyze

MX = "/tmp/mxenv/bin/python"


def test_crps_of_samples_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(0)
    mu, sd, y = 2.0, 1.5, 3.1
    z = (y - mu) / sd
    exact = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
    est = metrics.crps_samples(rng.normal(mu, sd, 200_000), np.array(y))
    assert abs(float(est) - exact) / exact < 0.01
    assert metrics.crps_samples(np.full(50, 4.0), np.array(4.0)) == 0.0                  # a point mass at the truth scores 0


def test_kupiec_accepts_correct_coverage_and_rejects_wrong_coverage():
    assert metrics.kupiec_pvalue(800, 1000, 0.8) > 0.9
    assert metrics.kupiec_pvalue(700, 1000, 0.8) < 1e-6 and metrics.kupiec_pvalue(900, 1000, 0.8) < 1e-6
    p = metrics.kupiec_pvalue(72, 100, 0.8)
    assert 0.05 < p < 0.2


def test_energy_score_prefers_the_correct_dependence():
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    truth = np.array([1.0, 1.0])                                                         # a draw that moves together
    good = rng.multivariate_normal([0, 0], cov, 4000)
    indep = rng.multivariate_normal([0, 0], np.eye(2), 4000)
    assert metrics.energy_score(good, truth) < metrics.energy_score(indep, truth)


def test_var_system_has_the_requested_spectral_radius_and_correlation():
    for seed in range(4):
        A, S = sim.var_system(seed, radius=0.8, corr=0.6)
        assert abs(max(abs(np.linalg.eigvals(A))) - 0.8) < 1e-12
        assert np.allclose(np.diag(S), 1.0) and np.allclose(S[0, 1], 0.6)
        assert sim.var_system(seed, corr=0.0)[1][0, 1] == 0.0


def test_var_oracle_reproduces_the_exact_conditional_moments():
    Y, tr = sim.sim_var1(3, T=300, corr=0.6)
    s = sim.oracle_var1(Y[:, :-2], 2, tr, n_samples=200_000, seed=1)
    A, Sig, lv, sc = tr["A"], tr["Sigma"], tr["levels"], tr["scales"]
    x0 = (Y[:, -3] - lv) / sc
    m1, m2 = A @ x0, A @ A @ x0
    assert np.allclose(((s[:, 0] - lv) / sc).mean(0), m1, atol=0.02) and np.allclose(((s[:, 1] - lv) / sc).mean(0), m2, atol=0.02)
    assert np.allclose(np.cov(((s[:, 0] - lv) / sc).T), Sig, atol=0.02)
    assert np.allclose(np.cov(((s[:, 1] - lv) / sc).T), A @ Sig @ A.T + Sig, atol=0.03)


def test_deepstate_oracle_is_calibrated_and_its_uncertainty_grows():
    hits = []
    for seed in range(12):
        Y, tr = sim.sim_deepstate(seed, n_series=4, T=250)
        for i in range(4):
            m, v = sim.oracle_deepstate(Y[i, :-7], 7, tr["sigma_level"], tr["sigma_obs"])
            assert np.all(np.diff(v) > 0)                                                # a random-walk level: variance grows with the horizon
            hits.append(np.abs(Y[i, -7:] - m) <= stats.norm.ppf(0.9) * np.sqrt(v))
    cover = np.concatenate(hits).mean()
    assert 0.72 < cover < 0.88, cover                                                    # true 80% intervals of the true model


def test_seed_count_formula_moves_the_right_way():
    base = analyze.n_for_noninferiority(sd_d=1.0, margin=1.0)
    assert analyze.n_for_noninferiority(sd_d=2.0, margin=1.0) > base                      # noisier -> more seeds
    assert analyze.n_for_noninferiority(sd_d=1.0, margin=2.0) < base                      # looser margin -> fewer seeds
    assert analyze.n_for_noninferiority(sd_d=1.0, margin=1.0, true_diff=0.5) > base       # a real shortfall needs more seeds
    assert analyze.n_for_noninferiority(sd_d=1.0, margin=1.0, true_diff=1.0) == -1        # no margin left to detect
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.8)
    assert base >= math.ceil(z ** 2)                                                      # the t correction never goes below the normal approximation


def test_noninferiority_test_separates_equal_from_clearly_worse():
    rng = np.random.default_rng(2)
    ref = rng.normal(1.0, 0.2, 12)
    same = analyze.noninferiority(ref, ref + rng.normal(0, 0.05, 12))
    worse = analyze.noninferiority(ref, ref + 0.8 + rng.normal(0, 0.05, 12))
    assert same["noninferior_on_pilot"] is True and worse["noninferior_on_pilot"] is False
    assert same["margin"] == pytest.approx(same["reference_gap_sd"])                      # margin = 1 x the reference's own seed SD


def _fake(leg, setting, n, shift=0.0, corr=None):
    rng = np.random.default_rng(7)
    rows = []
    for s in range(n):
        base = {"leg": leg, "seed": s, "setting": setting, "n_series": 6, "H": 7, "cfg": {"epochs": 1}, "seconds": {"custom": 1.0, "reference": 2.0}}
        metric = "crps" if leg == "deepstate" else "energy_score"
        for who, add in (("oracle", 0.0), ("reference", 1.0), ("custom", 1.0 + shift)):
            base[who] = {metric: 2.0 + add + rng.normal(0, 0.2), "coverage80_hits": int(rng.binomial(42, 0.8)), "coverage80_n": 42,
                         "step1_predictive_std": 2.1 + rng.normal(0, 0.05), "step1_corr_dim01": (corr or 0) + rng.normal(0, 0.05)}
        if leg == "deepstate":
            base["truth"] = {"step1_predictive_std_oracle": 2.0, "sigma_obs": 2.0}
        else:
            base["truth"] = {"innovation_corr": corr}; base["corr"] = corr
            base["negative_control_diagonal"] = {metric: 2.0 + 3.0 + rng.normal(0, 0.2), "coverage80_hits": 30, "coverage80_n": 42, "step1_corr_dim01": 0.0}
        rows.append(base)
    return rows


def test_analysis_produces_the_proposal_from_results(tmp_path):
    rows = _fake("deepstate", "local_level_weekly", 8) + _fake("deepvar", "var1_corr0.6", 8, corr=0.6) + _fake("deepvar", "var1_corr0.0", 2, corr=0.0)
    f = tmp_path / "r.jsonl"; f.write_text("\n".join(json.dumps(r) for r in rows))
    res = analyze.analyse(f)
    ds, vr, thin = res["legs"]["deepstate/local_level_weekly"], res["legs"]["deepvar/var1_corr0.6"], res["legs"]["deepvar/var1_corr0.0"]
    assert ds["accuracy"]["noninferior_on_pilot"] is True and ds["accuracy"]["seeds_needed_for_80pct_power_if_equal"] > 2
    assert {"reference", "custom"} == set(ds["coverage80"]) and "parameter_recovery_step1_predictive_std_log_ratio" in ds
    assert vr["negative_control_full_vs_diagonal"]["p_full_better"] < 0.01 and vr["parameter_recovery_step1_cross_correlation"]["true_innovation_corr"] == 0.6
    assert "fewer than 3 seeds" in thin["note"]                                           # nothing is estimated from two seeds
    md = analyze.render(res)
    assert "PROPOSED" in md and "Nothing here is approved" in md


def test_approved_calibration_is_explicit_and_traceable_to_a_human():
    from config import load_config
    cal = load_config("config.json")["admission"]["calibration"]
    assert cal is not None and cal["_approved_by"] and cal["alpha"] == 0.05 and cal["n_seeds"] == 12
    assert cal["legs"]["deepvar"] == ["var1_corr0.0", "var1_corr0.6"]                      # both settings are in scope, not just the one that passed


@pytest.mark.skipif(not os.path.exists(MX), reason="the isolated MXNet environment is not present")
def test_reference_worker_runs_in_the_isolated_environment():
    from calibration.run_calibration import run_deepstate, DEFAULTS
    r = run_deepstate(0, {**DEFAULTS, "epochs": 1, "batches": 3, "n_series": 2, "T": 200}, MX)
    assert r["reference"]["crps"] > 0 and r["custom"]["crps"] > 0 and r["oracle"]["crps"] > 0 and r["oracle"]["crps"] < r["reference"]["crps"]
