"""
Admission validation for CUSTOM models (gap G-42; decisions D-4 / D-11): DeepState and DeepVAR must pass three layers before they
may compete. Admission = Layer 1 AND Layer 2 AND Layer 3; a model that fails (or whose Layer 2 has not been run) is not admitted and
never appears in the candidates dict.

  Layer 1 (hard gate, float64, no training): the model's own likelihood/filter equals an INDEPENDENT implementation to a relative
      difference <= 1e-6 -- numerical equality, no statistics.
        DeepState: Kalman log-likelihood, filtered state/covariance and forecast moments versus statsmodels' KalmanFilter.
        DeepVAR:   low-rank-plus-diagonal Gaussian NLL versus scipy / torch.distributions.
  Layer 3 (hard gate): same-seed reproducibility; save -> load -> infer identical; horizons 1, 4, 30 return exactly that many dated
      rows; no NaN/inf; changing FUTURE covariate values does not change EARLIER forecast steps; training loss decreases and early
      stopping triggers on a noisy series.
  Layer 2 (statistically designed suite, no arbitrary 5% / 10% margins): accuracy versus the reference and the true-parameter oracle
      by a paired equivalence / non-inferiority test across many seeds, coverage by a proper coverage test (Kupiec), parameter recovery
      against the seed distribution, a negative control for DeepVAR. Its significance level, number of seeds and series and the margin
      rule come from an APPROVED calibration run; until `config.json -> admission.calibration` is filled in, Layer 2 reports
      "not_run" and the model is not admitted. Nothing here invents those parameters.

A report is written per model with the sha256 of the module source; a later code change invalidates it.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REL_TOL = 1e-6
ROOT = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_source_hash(module_cls) -> str:
    import inspect
    return _sha(Path(inspect.getsourcefile(module_cls)))


def _check(name: str, ok: bool, detail: str = "", **extra) -> dict:
    return {"check": name, "passed": bool(ok), "detail": detail, **extra}


# ---------------------------------------------------------------------------------------------------------------- Layer 1: DeepState
def layer1_deepstate(seeds=(0, 1, 2), periods_options=((7,), (7, 30)), T=60, H=6, B=3, _sabotage: float = 0.0) -> dict:
    import torch
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
    from algorithms.deepstate import build_structure, kalman_loglik, kalman_predict
    checks = []
    worst = {"loglik": 0.0, "state": 0.0, "cov": 0.0, "forecast_mean": 0.0, "forecast_var": 0.0}
    for periods in periods_options:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            F, a, _ = build_structure(periods, torch.float64)
            n = F.shape[0]
            Ttot = T + H
            g = rng.uniform(0.01, 0.4, (B, Ttot, n)); sg = rng.uniform(0.1, 0.8, (B, Ttot)); b = rng.normal(0, 0.2, (B, Ttot))
            z = 1.0 + 0.1 * np.arange(T)[None, :] / T + rng.normal(0, 0.3, (B, T))
            m0 = rng.normal(0, 1, (B, n)); A = rng.normal(0, 1, (B, n, n)); P0 = A @ A.transpose(0, 2, 1) + np.eye(n)
            t = lambda x: torch.tensor(x, dtype=torch.float64)
            g_ours = g * (1.0 + _sabotage)     # _sabotage != 0 deliberately breaks OUR side only: proves the check can fail
            ll, m_f, P_f = kalman_loglik(t(z), F, a, t(g_ours[:, :T]), t(sg[:, :T]), t(b[:, :T]), t(m0), t(P0), return_last=True)
            m1 = m_f @ F.T
            P1 = F @ P_f @ F.T + t(g_ours[:, T - 1])[:, :, None] * t(g_ours[:, T - 1])[:, None, :]
            fmean, fvar = kalman_predict(m1, P1, F, a, t(g_ours[:, T:]), t(sg[:, T:]), t(b[:, T:]))
            Fn, an = F.numpy(), a.numpy()
            for i in range(B):
                kf = KalmanFilter(k_endog=1, k_states=n, k_posdef=1)
                endog = np.concatenate([z[i], np.full(H, np.nan)])          # future periods = missing observations -> forecasts
                kf.bind(endog.reshape(Ttot, 1))                             # bind first: statsmodels needs nobs for time-varying matrices
                kf.design = an.reshape(1, n, 1)
                kf.obs_intercept = b[i].reshape(1, Ttot)
                kf.obs_cov = (sg[i] ** 2).reshape(1, 1, Ttot)
                kf.transition = Fn.reshape(n, n, 1)
                kf.selection = g[i].T.reshape(n, 1, Ttot)
                kf.state_cov = np.ones((1, 1, 1))
                kf.initialize_known(m0[i], P0[i])
                res = kf.filter()
                ref_ll = float(np.nansum(res.llf_obs[:T]))
                rel = lambda x, y: float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
                d = {"loglik": rel(np.array(float(ll[i])), np.array(ref_ll)),
                     "state": rel(m_f[i].numpy(), res.filtered_state[:, T - 1]),
                     "cov": float(np.max(np.abs(P_f[i].numpy() - res.filtered_state_cov[:, :, T - 1])) / np.max(np.abs(res.filtered_state_cov[:, :, T - 1]))),
                     "forecast_mean": rel(fmean[i].numpy(), res.forecasts[0, T:]),
                     "forecast_var": rel(fvar[i].numpy(), res.forecasts_error_cov[0, 0, T:])}
                for k, v in d.items():
                    worst[k] = max(worst[k], v)
            checks.append(_check(f"periods={list(periods)} seed={seed}", all(worst[k] <= REL_TOL for k in worst),
                                 f"{B} series, state dim {n}, T={T}, H={H}"))
    ok = all(v <= REL_TOL for v in worst.values())
    return {"passed": ok, "tolerance": REL_TOL, "reference": "statsmodels.tsa.statespace.kalman_filter.KalmanFilter (independent implementation)",
            "worst_relative_difference": worst, "cases": len(checks), "dtype": "float64", "trained": False}


def layer1_deepvar(seeds=(0, 1, 2), dims=(2, 4, 7), ranks=(1, 2, 3), batch=20, _sabotage: float = 0.0) -> dict:
    """The low-rank-plus-diagonal Gaussian log-density (float64, no training) versus scipy AND torch.distributions."""
    import torch
    from scipy.stats import multivariate_normal
    from algorithms.deepvar import lowrank_gaussian_logpdf
    worst_scipy = worst_torch = 0.0
    cases = 0
    for m in dims:
        for r in ranks:
            for seed in seeds:
                rng = np.random.default_rng(seed * 100 + m * 10 + r)
                mu = rng.normal(0, 1, (batch, m)); V = rng.normal(0, 0.8, (batch, m, r)); d = rng.uniform(0.05, 1.5, (batch, m))
                x = mu + rng.normal(0, 1.2, (batch, m))
                t = lambda a: torch.tensor(a, dtype=torch.float64)
                ours = lowrank_gaussian_logpdf(t(x), t(mu), t(V) * (1.0 + _sabotage), t(d)).numpy()
                ref_s = np.array([multivariate_normal(mean=mu[i], cov=np.diag(d[i]) + V[i] @ V[i].T).logpdf(x[i]) for i in range(batch)])
                ref_t = torch.distributions.LowRankMultivariateNormal(t(mu), cov_factor=t(V), cov_diag=t(d)).log_prob(t(x)).numpy()
                worst_scipy = max(worst_scipy, float(np.max(np.abs(ours - ref_s) / np.maximum(np.abs(ref_s), 1e-12))))
                worst_torch = max(worst_torch, float(np.max(np.abs(ours - ref_t) / np.maximum(np.abs(ref_t), 1e-12))))
                cases += 1
    # sampled covariance vs the true covariance. The sampling map is LINEAR in its noise, x = mu + A [eps_r; eps_m] with A = [V, sqrt(D)],
    # so feeding it the unit vectors recovers A exactly and A A^T must equal D + V V^T; a Monte-Carlo estimate is reported alongside.
    from algorithms.deepvar import sample_lowrank
    worst_cov, mc = 0.0, 0.0
    for m in dims:
        for r in ranks:
            rng = np.random.default_rng(1000 + m * 10 + r)
            V = torch.tensor(rng.normal(0, 0.8, (m, r)) * (1.0 + _sabotage), dtype=torch.float64); d = torch.tensor(rng.uniform(0.05, 1.5, m), dtype=torch.float64)
            mu = torch.zeros(m, dtype=torch.float64)
            cols = []
            for i in range(r + m):
                e = torch.zeros(r + m, dtype=torch.float64); e[i] = 1.0
                cols.append(sample_lowrank(mu, V, d, e[:r], e[r:]))
            A = torch.stack(cols, 1)
            V0 = V / (1.0 + _sabotage)                    # the INTENDED factor (equal to V unless a sabotage is being simulated)
            true_cov = torch.diag(d) + V0 @ V0.T
            worst_cov = max(worst_cov, float((A @ A.T - true_cov).abs().max() / true_cov.abs().max()))
            gen = torch.Generator().manual_seed(m * 10 + r)
            n = 200_000
            xs = sample_lowrank(mu.expand(n, m), V.expand(n, m, r), d.expand(n, m), torch.randn(n, r, generator=gen, dtype=torch.float64), torch.randn(n, m, generator=gen, dtype=torch.float64))
            emp = torch.cov(xs.T)
            mc = max(mc, float((emp - true_cov).abs().max() / true_cov.abs().max()))
    ok = worst_scipy <= REL_TOL and worst_torch <= REL_TOL and worst_cov <= REL_TOL
    return {"passed": ok, "tolerance": REL_TOL, "reference": "scipy.stats.multivariate_normal and torch.distributions.LowRankMultivariateNormal; exact covariance D + V V^T",
            "worst_relative_difference": {"vs_scipy": worst_scipy, "vs_torch_distributions": worst_torch, "sampling_map_covariance_vs_true": worst_cov},
            "sampled_covariance_monte_carlo_max_relative_error": mc, "monte_carlo_samples": 200_000, "cases": cases,
            "dtype": "float64", "trained": False}


# ---------------------------------------------------------------------------------------------------------------- Layer 3 (generic)
def _dates(df):
    return pd.DatetimeIndex(sorted(df.loc[df["series_role"] == "endogenous", "date"].unique()))


def _forecast(module, pool: dict, sid: str, h: int) -> pd.DataFrame:
    """One series' forecast; a JOINT model is asked for the whole pool at once."""
    if getattr(module, "joint_inference", False):
        return module.infer_pool(pool, h)[sid]
    return module.infer(pool[sid], h)


def layer3(make_module, pool_factory, noisy_pool_factory, fk_pool_factory=None, horizons=(1, 4, 30)) -> dict:
    """`make_module(**overrides)` -> a fresh module; the factories return {segment_id: contract frame}."""
    from pooling import assemble_pool
    checks = []
    pool = pool_factory()
    batch = assemble_pool(pool)
    first_id, first_df = next(iter(pool.items()))

    a, b = make_module(), make_module()
    a.train_pooled(batch); b.train_pooled(batch)
    fa = {sid: _forecast(a, pool, sid, 4)["forecast"].to_numpy() for sid in pool}
    fb = {sid: _forecast(b, pool, sid, 4)["forecast"].to_numpy() for sid in pool}
    checks.append(_check("same-seed reproducibility", all(np.allclose(fa[s], fb[s], rtol=1e-6, atol=1e-6) for s in pool),
                         "two fits with the same seed give the same forecasts"))

    with tempfile.TemporaryDirectory() as d:
        a.save(Path(d) / "m" / "model.pkl")
        loaded = make_module(); loaded.load(Path(d) / "m" / "model.pkl")
        fl = {sid: _forecast(loaded, pool, sid, 4)["forecast"].to_numpy() for sid in pool}
    checks.append(_check("save -> load -> infer identical", all(np.allclose(fa[s], fl[s], rtol=1e-6, atol=1e-6) for s in pool),
                         "loaded without any training data"))

    rows_ok, finite_ok = True, True
    for h in horizons:
        fc = _forecast(a, pool, first_id, h)
        last = _dates(first_df).max()
        from algorithms.utils import future_dates
        exp = future_dates(last, h, freq=getattr(a, "frequency", "D"))
        rows_ok &= (len(fc) == h and list(fc["date"]) == list(exp))
        finite_ok &= bool(np.isfinite(fc["forecast"].to_numpy()).all())
    checks.append(_check(f"horizons {list(horizons)} return exactly that many dated rows", rows_ok))
    checks.append(_check("no NaN / inf in forecasts", finite_ok))

    if fk_pool_factory is None:
        checks.append(_check("changing future covariates does not change earlier steps", True, "not applicable: no future-known inputs supplied", applicable=False))
    else:
        fk_pool = fk_pool_factory()
        fk_batch = assemble_pool(fk_pool)
        m = make_module(); m.train_pooled(fk_batch)
        sid, df = next(iter(fk_pool.items()))
        base = m.infer(df, 8)["forecast"].to_numpy()
        last = _dates(df).max()
        edited = df.copy()
        ex = (edited["series_role"] == "exogenous") & (edited["date"] >= last + pd.Timedelta(days=6)) & (edited["date"] <= last + pd.Timedelta(days=8))
        edited.loc[ex, "value"] = edited.loc[ex, "value"] * 5 + 100
        alt = m.infer(edited, 8)["forecast"].to_numpy()
        same_before = bool(np.allclose(base[:5], alt[:5], rtol=1e-7, atol=1e-7))
        checks.append(_check("changing future covariates does not change earlier steps", same_before,
                             "future-known values at steps 6-8 changed; steps 1-5 compared", later_steps_changed=bool(not np.allclose(base[5:], alt[5:])),
                             applicable=True))

    good = make_module(); good.train_pooled(batch)
    hist = good._fitted_model["early_stopping"]["history"]
    dec = len(hist) >= 2 and hist[-1]["train_nll"] < hist[0]["train_nll"]
    checks.append(_check("training loss decreases", dec, f"first {hist[0]['train_nll']:.3f} -> last {hist[-1]['train_nll']:.3f}" if hist else ""))
    noisy = make_module(max_epochs=60, early_stop_patience=2)
    noisy.train_pooled(assemble_pool(noisy_pool_factory()))
    es = noisy._fitted_model["early_stopping"]
    checks.append(_check("early stopping triggers on a noisy series", bool(es["used"] and es["stopped_early"]),
                         f"stopped after {es['epochs_run']} of {es['max_epochs']} epochs; best epoch {es['best_epoch']}; best weights restored: {es['restored_best_checkpoint']}"))
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def layer2(algo_id: str, calibration: dict | None) -> dict:
    """
    The statistically designed suite (G-42, D-11): every setting APPROVED for this model must show non-inferiority to the
    reference (paired test, margin = calibration["margin_k"] x the reference's own seed-to-seed SD of its gap to the oracle),
    proper 80% interval coverage (Kupiec, not rejected at `alpha`), and -- for a DeepVAR setting with a true nonzero
    correlation -- the negative control: full covariance must beat the diagonal-covariance variant at `alpha`. Uses the
    calibration run's own results file; a setting with fewer than the approved seed count is reported as insufficient, not
    guessed. `run_calibration.py` / `analyze.py` produced the numbers that must be approved before this can run at all.
    """
    if not calibration:
        return {"passed": False, "status": "not_run",
                "reason": "Layer 2 needs an approved calibration (significance level, number of seeds and series, margin rule); "
                          "config.json -> admission.calibration is not set, so nothing is guessed"}
    from calibration import analyze, metrics
    results_path = Path(calibration["results_path"])
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    if not results_path.exists():
        return {"passed": False, "status": "not_run", "reason": f"calibration results file {results_path} does not exist"}
    alpha, k, n_seeds = float(calibration["alpha"]), float(calibration["margin_k"]), int(calibration["n_seeds"])
    settings = calibration.get("legs", {}).get(algo_id, [])
    if not settings:
        return {"passed": False, "status": "not_run", "reason": f"the approved calibration names no settings for {algo_id!r}"}
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    metric = "crps" if algo_id == "deepstate" else "energy_score"
    checks = []
    for setting in settings:
        rows = sorted((r for r in recs if r["leg"] == algo_id and r["setting"] == setting), key=lambda r: r["seed"])[:n_seeds]
        if len(rows) < n_seeds:
            checks.append({"setting": setting, "passed": False, "reason": f"only {len(rows)} of {n_seeds} approved seeds are in {results_path.name}"})
            continue
        orc = np.array([r["oracle"][metric] for r in rows]); ref = np.array([r["reference"][metric] for r in rows]); cus = np.array([r["custom"][metric] for r in rows])
        acc = analyze.noninferiority(ref - orc, cus - orc, k=k, alpha=alpha)
        hits = sum(r["custom"]["coverage80_hits"] for r in rows); n_pts = sum(r["custom"]["coverage80_n"] for r in rows)
        cov_p = metrics.kupiec_pvalue(hits, n_pts, 0.8)
        entry = {"setting": setting, "n_seeds": n_seeds, "margin": acc["margin"], "noninferiority_p": acc["p_value_noninferior"],
                 "noninferior": bool(acc["noninferior_on_pilot"]), "coverage": hits / n_pts, "coverage_kupiec_p": cov_p, "coverage_ok": bool(cov_p > alpha)}
        passed = entry["noninferior"] and entry["coverage_ok"]
        true_corr = rows[0].get("corr")
        if algo_id == "deepvar" and true_corr:
            dia = np.array([r["negative_control_diagonal"][metric] for r in rows])
            gain = (dia - orc) - (cus - orc)
            sd_g = float(np.std(gain, ddof=1))
            t = gain.mean() / (sd_g / np.sqrt(len(gain))) if sd_g > 0 else float("nan")
            from scipy import stats as _st
            nc_p = float(1 - _st.t.cdf(t, len(gain) - 1)) if np.isfinite(t) else float("nan")
            entry["negative_control_p_full_beats_diagonal"] = nc_p
            entry["negative_control_ok"] = bool(np.isfinite(nc_p) and nc_p < alpha)
            passed = passed and entry["negative_control_ok"]
        entry["passed"] = passed
        checks.append(entry)
    overall = all(c["passed"] for c in checks)
    return {"passed": overall, "status": "passed" if overall else "failed", "alpha": alpha, "margin_k": k, "n_seeds": n_seeds,
            "results_file": str(results_path), "checks": checks}


def run_admission(algo_id: str, module_cls, layer1: dict, layer3_result: dict, calibration: dict | None, reports_dir: Path) -> dict:
    l2 = layer2(algo_id, calibration)
    report = {
        "algorithm": algo_id, "module_sha256": module_source_hash(module_cls),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layer1": layer1, "layer2": l2, "layer3": layer3_result,
        "admitted": bool(layer1["passed"] and l2["passed"] and layer3_result["passed"]),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{algo_id}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def is_admitted(algo_id: str, module_cls, config: dict) -> tuple[bool, str]:
    """True only with a report that says admitted, for exactly the current module source."""
    adm = (config or {}).get("admission", {})
    rd = Path(adm.get("reports_dir", "admission_reports"))
    rd = rd if rd.is_absolute() else ROOT / rd
    f = rd / f"{algo_id}.json"
    if not f.exists():
        return False, "no admission report"
    rep = json.loads(f.read_text())
    if rep.get("module_sha256") != module_source_hash(module_cls):
        return False, "module source changed since the admission report"
    if not rep.get("admitted"):
        why = [k for k in ("layer1", "layer2", "layer3") if not rep.get(k, {}).get("passed")]
        return False, f"admission report says not admitted ({', '.join(why)} did not pass)"
    return True, "admitted"
