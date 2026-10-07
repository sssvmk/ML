"""Human-readable experiment report and model card, generated only from measured values."""
from __future__ import annotations

from pathlib import Path

from .utils import load_json


def write_reports(run_dir: Path, cfg: dict, ctx: dict) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    sel, test = load_json(run_dir / "selection.json"), load_json(run_dir / "test_evaluation.json")
    pop = load_json(run_dir / "splits" / "population_report.json")
    eda_p = run_dir / "eda" / "eda_summary.json"
    L = ["# Experiment report", "", f"* run `{run_dir.name}`, data hash `{ctx['data_hash'][:16]}`, seed {cfg['project']['seed']}",
         f"* rows {ctx['rows']:,} (train {ctx['n_train']:,} / validation {ctx['n_val']:,} / test {ctx['n_test']:,})",
         f"* primary metric **{cfg['metric']['primary']}**; target {cfg['metric']['target_value'] or 'NOT SET (see TODO)'}", ""]
    if eda_p.exists():
        L += ["## 1. Data understanding", ""] + [f"* {f}" for f in load_json(eda_p)["findings"]] + ["", "Plots: `eda/plots/`.", ""]
    L += ["## 2. Splits", "", f"* sizes {pop['sizes']}; representative: **{pop['representative']}**; adversarial AUC train-vs-validation {ctx.get('adversarial_auc', float('nan')):.3f}", ""]
    L += ["## 3. Stage 1: autoencoders (unsupervised)", ""]
    for kind in cfg["autoencoder"]["kinds"]:
        p = run_dir / "autoencoders" / kind / "result.json"
        if not p.exists():
            continue
        r = load_json(p)
        if r["status"] != "ok":
            L += [f"### {r['display_name']}: {r['status']}", r["message"], ""]
            continue
        s = r["reconstruction"]
        bad = [c for c in r["sanity"] if not c["passed"]]
        L += [f"### {r['display_name']} (`{kind}`)", "", f"* params `{r['params']}`", f"* search: {r['search']['method']}, {r['search']['n_trials_completed']} trials, {r['epochs_trained']} epochs trained",
              f"* validation reconstruction: continuous R² {s['mean_continuous_r2']:.3f}, categorical accuracy {s['mean_categorical_accuracy']:.3f} "
              f"(majority baseline {s['mean_majority_baseline_accuracy']:.3f}), exact categorical row match {s['exact_row_match_categorical']:.3f}",
              f"* sanity checks: {len(r['sanity']) - len(bad)}/{len(r['sanity'])} passed" + ("; flagged: " + "; ".join(f"{c['name']} ({c['detail']})" for c in bad) if bad else ""),
              f"* plots: `autoencoders/{kind}/plots/`; decoded examples: `autoencoders/{kind}/reconstruction_examples.csv`", ""]
    L += ["## 4. Stage 2: classifiers on the encoder", ""]
    for r in sorted((load_json(p) for p in (run_dir / "candidates").glob("*/result.json")), key=lambda r: r["name"]):
        if r["status"] != "ok":
            L += [f"* {r['name']}: {r['status']} - {r['message']}"]
            continue
        L.append(f"* **{r['display_name']}**: validation {r['primary_metric']} {r['val_primary']:.4f}, train {r['train_primary']:.4f}, gap {r['overfit_gap']:+.4f}; {r['diagnosis']['verdict']}")
    L += ["", "## 5. Winner and analysis", "", f"**{sel['winner']}** (raw best {sel['raw_best']}).", ""] + [f"* {t}" for t in sel["decision_trace"]]
    if sel["pretraining_benefit"]:
        L += ["", "Encoder benefit vs the matching no-encoder control:"] + [f"* {b['candidate']} vs {b['control']}: {b['diff_vs_scratch']:+.4f} {b['ci95']} -> {b['verdict']}" for b in sel["pretraining_benefit"]]
    L += ["", "## 6. One-shot test evaluation (winner only)", "",
          f"* {cfg['metric']['primary']} **{test['primary_value']:.4f}**; accuracy {test['metrics']['accuracy']:.4f}; F1 {test['metrics']['f1']:.4f}; ECE {test['metrics']['ece']:.4f}",
          f"* meets target: {test['meets_target'] if test['meets_target'] is not None else 'n/a (no target configured)'}; max slice accuracy gap {test['max_slice_accuracy_gap']:.4f}"]
    rc = test.get("pretrained_autoencoder_reconstruction_on_test")
    if rc:
        L.append(f"* reconstruction on test: R² {rc['summary']['mean_continuous_r2']:.3f}, categorical accuracy {rc['summary']['mean_categorical_accuracy']:.3f}")
    rp = run_dir / "reports" / "experiment_report.md"
    rp.parent.mkdir(exist_ok=True, parents=True)
    rp.write_text("\n".join(L) + "\n", encoding="utf-8")
    M = [f"# Model card: {sel['winner']}", "", "> Auto-generated from measured values; TODO markers need a human owner.", "",
         "## Intended use", "", "Predict whether an airline passenger reports being satisfied, via an encoder learned by reconstructing the passenger record. TODO: owner to confirm intended and out-of-scope uses.", "",
         "## Data", "", f"* {ctx['rows']:,} rows, hash `{ctx['data_hash'][:16]}`, target `{cfg['data']['target']}`. TODO: provenance, collection period, known gaps.", "",
         "## Performance", "", f"* validation {sel['primary_metric']}: {next(t['val_primary'] for t in sel['table'] if t['name'] == sel['winner']):.4f}; test {test['primary_value']:.4f} (n={test['n_test_rows']:,}, evaluated once)",
         "", "### Slices (test)", ""] + [f"* {c}: accuracy gap {s['accuracy_gap']:.4f} (" + ", ".join(f"{k}: {v['accuracy']:.3f}" for k, v in s["levels"].items()) + ")" for c, s in test["slice_metrics"].items()]
    M += ["", "## Limitations", "", "* Self-reported survey data; rating 0 = not applicable.", "* No passenger/flight/date column: repeat passengers and drift are not controlled.",
          "* The classifier only sees the latent code; a small latent can discard information that raw-feature models keep (see the control and comparison table).",
          "* TODO: fairness thresholds for slice gaps; monitoring owner and retraining trigger."]
    mp = run_dir / "reports" / "MODEL_CARD.md"
    mp.write_text("\n".join(M) + "\n", encoding="utf-8")
    return rp, mp
