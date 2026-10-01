"""
Layer 2 CALIBRATION RUN (G-42, decision D-11): measures the seed-to-seed variability of the reference implementation and of the custom
model against the true-parameter oracle, on controlled synthetic data, so that the significance level, the number of seeds and series and
the margin rule can be PROPOSED from data and then approved. It admits nothing. Resumable: results are appended to a JSON-lines file and
finished (leg, seed, setting) triples are skipped.

    python -m calibration.run_calibration --leg deepstate --seeds 0 5 --out calibration/results.jsonl
    python -m calibration.run_calibration --leg deepvar   --seeds 0 5 --out calibration/results.jsonl

Fairness: reference and custom get the SAME optimiser budget (epochs x batches per epoch, batch size 32, Adam, same learning rate), the same
data and the same horizon; the custom model's early stopping is switched off for the calibration so the budgets match.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calibration import sim, metrics                                   # noqa: E402

DEFAULTS = {"H": 7, "epochs": 8, "batches": 20, "lr": 1e-2, "cells": 16, "num_samples": 100, "n_series": 6, "T": 350, "context": 40, "rank": 2}


def _frames(Y: np.ndarray, T_train: int):
    """Contract frames (one per series) with the simulated values as the endogenous series; used to feed the custom modules."""
    from adapters.synthetic import SyntheticAdapter
    out = {}
    for i in range(Y.shape[0]):
        df = SyntheticAdapter().extract(company_code=f"{1000 * (i + 1)}", currency="INR", process="AR", n_days=Y.shape[1], seed=i, start=sim.START)
        df = df[df["series_role"] == "endogenous"].copy().sort_values("date")
        df["value"] = Y[i]
        df = df[df["date"].isin(sorted(df["date"].unique())[:T_train])]
        out[df["segment_id"].iloc[0]] = df
    return out


def _ref(kind: str, Y_train: np.ndarray, cfg: dict, seed: int, python: str):
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / "in.npz", Path(d) / "out.npz"
        np.savez(inp, kind=kind, Y=Y_train, H=cfg["H"], seed=seed, epochs=cfg["epochs"], batches=cfg["batches"], num_samples=cfg["num_samples"],
                 lr=cfg["lr"], cells=cfg["cells"], context=cfg["context"], rank=cfg["rank"])
        r = subprocess.run([python, str(Path(__file__).parent / "reference_worker.py"), str(inp), str(outp)], capture_output=True, text=True, timeout=3000)
        if r.returncode != 0:
            raise RuntimeError("reference worker failed: " + (r.stderr or r.stdout)[-600:])
        z = np.load(outp)
        return z["samples"], float(z["seconds"])


def run_deepstate(seed: int, cfg: dict, python: str) -> dict:
    from pooling import assemble_pool
    from algorithms.deepstate import DeepStateModule
    Y, truth = sim.sim_deepstate(seed, n_series=cfg["n_series"], T=cfg["T"])
    H, T = cfg["H"], Y.shape[1]
    Tt = T - H
    obs = Y[:, Tt:]
    # oracle (true parameters)
    orc_m, orc_v = zip(*[sim.oracle_deepstate(Y[i, :Tt], H, truth["sigma_level"], truth["sigma_obs"]) for i in range(len(Y))])
    rng = np.random.default_rng(seed)
    orc = np.stack([rng.normal(m, np.sqrt(v), (cfg["num_samples"], H)) for m, v in zip(orc_m, orc_v)], 1)          # (ns, n, H)
    # custom
    frames = _frames(Y, Tt)
    t0 = time.time()
    mod = DeepStateModule({"horizon": H, "context_length": cfg["context"], "hidden_size": cfg["cells"], "max_epochs": cfg["epochs"],
                           "steps_per_epoch": cfg["batches"], "batch_size": 32, "learning_rate": cfg["lr"], "early_stop_patience": 0, "random_seed": seed})
    mod.train_pooled(assemble_pool(frames))
    cus = np.stack([mod.sample_paths(df, H, n=cfg["num_samples"], seed=seed) for df in frames.values()], 1)         # (ns, n, H)
    t_custom = time.time() - t0
    ref, t_ref = _ref("deepstate", Y[:, :Tt], cfg, seed, python)
    ref = np.transpose(ref, (1, 0, 2))                                                                              # (ns, n, H)
    res = {"leg": "deepstate", "seed": seed, "setting": "local_level_weekly", "n_series": len(Y), "H": H, "seconds": {"custom": t_custom, "reference": t_ref}}
    for name, smp in (("oracle", orc), ("reference", ref), ("custom", cus)):
        crps = metrics.crps_samples(smp, obs)
        hits = metrics.central_interval_hits(smp, obs, 0.8)
        res[name] = {"crps": float(crps.mean()), "mae": float(np.abs(np.median(smp, 0) - obs).mean()), "coverage80_hits": int(hits.sum()), "coverage80_n": int(hits.size),
                     "step1_predictive_std": float(smp[:, :, 0].std(0, ddof=1).mean())}
    res["truth"] = {"step1_predictive_std_oracle": float(np.sqrt(np.array(orc_v)[:, 0]).mean()), "sigma_obs": truth["sigma_obs"]}
    return res


def run_deepvar(seed: int, corr: float, cfg: dict, python: str) -> dict:
    from pooling import assemble_pool
    from algorithms.deepvar import DeepVARModule
    Y, truth = sim.sim_var1(seed, T=cfg["T"] + 50, corr=corr)
    H, T = cfg["H"], Y.shape[1]
    Tt = T - H
    obs = Y[:, Tt:]                                                                                                 # (dim, H)
    orc = sim.oracle_var1(Y[:, :Tt], H, truth, n_samples=cfg["num_samples"], seed=seed)                             # (ns, H, dim)
    frames = _frames(Y, Tt)
    hp = {"horizon": H, "context_length": cfg["context"], "hidden_size": cfg["cells"], "rank": cfg["rank"], "max_epochs": cfg["epochs"],
          "steps_per_epoch": cfg["batches"], "batch_size": 32, "learning_rate": cfg["lr"], "early_stop_patience": 0, "random_seed": seed, "num_samples": cfg["num_samples"]}
    out = {}
    t_custom = {}
    for label, extra in (("custom", {}), ("negative_control_diagonal", {"covariance": "diagonal"})):
        t0 = time.time()
        m = DeepVARModule({**hp, **extra})
        m.train_pooled(assemble_pool(frames))
        out[label] = m.sample_paths(frames, H, n=cfg["num_samples"], seed=seed)                                     # (ns, H, dim)
        t_custom[label] = time.time() - t0
    ref, t_ref = _ref("deepvar", Y[:, :Tt], cfg, seed, python)                                                       # (ns, H, dim)
    res = {"leg": "deepvar", "seed": seed, "setting": f"var1_corr{corr}", "corr": corr, "n_series": Y.shape[0], "H": H,
           "seconds": {**t_custom, "reference": t_ref}}
    for name, smp in (("oracle", orc), ("reference", ref), ("custom", out["custom"]), ("negative_control_diagonal", out["negative_control_diagonal"])):
        crps = metrics.crps_samples(np.transpose(smp, (0, 2, 1)), obs)                                              # per dim/step
        hits = metrics.central_interval_hits(np.transpose(smp, (0, 2, 1)), obs, 0.8)
        es = float(np.mean([metrics.energy_score(smp[:, k, :], obs[:, k], seed=seed) for k in range(H)]))
        c01 = float(np.corrcoef(smp[:, 0, 0], smp[:, 0, 1])[0, 1])
        res[name] = {"crps": float(crps.mean()), "energy_score": es, "coverage80_hits": int(hits.sum()), "coverage80_n": int(hits.size), "step1_corr_dim01": c01}
    res["truth"] = {"innovation_corr": corr}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["deepstate", "deepvar"], required=True)
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 3], help="first and last+1 seed")
    ap.add_argument("--out", default=str(ROOT / "calibration" / "results.jsonl"))
    ap.add_argument("--python", default="/tmp/mxenv/bin/python", help="interpreter of the isolated MXNet environment")
    ap.add_argument("--set", nargs="*", default=[], help="override defaults, e.g. epochs=4 batches=10")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("="); cfg[k] = type(DEFAULTS[k])(v)
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add((r["leg"], r["seed"], r["setting"], json.dumps(r.get("cfg"), sort_keys=True)))
    for seed in range(*a.seeds):
        settings = [("local_level_weekly", None)] if a.leg == "deepstate" else [("var1_corr0.0", 0.0), ("var1_corr0.6", 0.6)]
        for setting, corr in settings:
            key = (a.leg, seed, setting, json.dumps(cfg, sort_keys=True))
            if key in done:
                continue
            t0 = time.time()
            res = run_deepstate(seed, cfg, a.python) if a.leg == "deepstate" else run_deepvar(seed, corr, cfg, a.python)
            res["cfg"] = cfg; res["wall_seconds"] = time.time() - t0
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{a.leg} seed {seed} {setting}: {res['wall_seconds']:.0f}s  crps oracle/ref/custom = "
                  f"{res['oracle']['crps']:.3f}/{res['reference']['crps']:.3f}/{res['custom']['crps']:.3f}", flush=True)


if __name__ == "__main__":
    main()
