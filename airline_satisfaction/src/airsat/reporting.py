"""Human-readable artefacts: experiment report and model card (generated only from measured values)."""
from __future__ import annotations

from pathlib import Path

from .utils import load_json


def _fmt(v, nd=4):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_experiment_report(run_dir: Path, cfg: dict, context: dict) -> Path:
    run_dir = Path(run_dir)
    sel = load_json(run_dir / "selection.json")
    test = load_json(run_dir / "test_evaluation.json")
    eda = load_json(run_dir / "eda" / "eda_summary.json") if (run_dir / "eda" / "eda_summary.json").exists() else None
    pop = load_json(run_dir / "splits" / "population_report.json")
    L = ["# Experiment report", "", f"* run: `{run_dir.name}`  * data hash: `{context['data_hash'][:16]}`  * seed: {cfg['project']['seed']}",
         f"* rows: {context['rows']:,} (train {context['n_train']:,} / validation {context['n_val']:,} / test {context['n_test']:,})",
         f"* primary metric: **{cfg['metric']['primary']}**; target: {cfg['metric']['target_value'] or 'NOT SET (see TODO)'}", ""]
    if eda:
        L += ["## 1. Data understanding", ""] + [f"* {f}" for f in eda["findings"]] + ["", "Plots: `eda/plots/`.", ""]
    L += ["## 2. Split representativeness", "",
          f"* sizes: {pop['sizes']}",
          f"* worst KS {pop['worst']['ks']:.4f}, worst Cramér's V {pop['worst']['cramers_v']:.4f}, worst prevalence difference {pop['worst']['prevalence_diff']:.4f}",
          f"* representative (all below thresholds {pop['thresholds']}): **{pop['representative']}**",
          f"* adversarial validation AUC train-vs-validation: {_fmt(context.get('adversarial_auc'), 3)} (0.5 = indistinguishable)", ""]
    L += ["## 3. Algorithms", ""]
    results = [load_json(p) for p in sorted((run_dir / "algorithms").glob("*/result.json"))]
    for r in results:
        L += [f"### {r['display_name']} ({r['family']}) - status: {r['status']}", ""]
        if r["status"] != "ok":
            L += [f"{r['message']}", ""]
            continue
        L += [f"* loss: {r['loss_function']}", f"* optimisation: {r['optimisation_notes']}",
              f"* search: {r['search'].get('method')} - {r['search'].get('n_trials_completed', 0)} trials; best params: `{r['best_params']}`",
              f"* validation {r['primary_metric']}: **{r['val_primary']:.4f}**, train: {r['train_primary']:.4f}, gap {r['overfit_gap']:+.4f}; diagnosis: {r['diagnosis']['verdict']}"]
        for note in r["diagnosis"]["notes"]:
            L.append(f"  * {note}")
        bad = [a for a in r["assumptions"] if not a["passed"]]
        L.append(f"* assumptions: {len(r['assumptions']) - len(bad)}/{len(r['assumptions'])} passed" + ("; flagged: " + "; ".join(f"{a['name']} ({a['detail']})" for a in bad) if bad else ""))
        L += ["", f"![curve]({Path(r['artifacts'].get('training_curve_plot', '')).relative_to(run_dir) if r['artifacts'].get('training_curve_plot') else ''})", ""]
    L += ["## 4. Winner", "", f"**{sel['winner']}** (raw best: {sel['raw_best']}).", "", "Decision trace:", ""] + [f"* {t}" for t in sel["decision_trace"]]
    L += ["", "See `model_comparison.md` for the full table.", "", "## 5. Final test evaluation (winner only, evaluated once)", "",
          f"* {cfg['metric']['primary']}: **{test['primary_value']:.4f}**; accuracy {test['metrics']['accuracy']:.4f}; F1 {test['metrics']['f1']:.4f}; "
          f"ROC-AUC {test['metrics']['roc_auc']:.4f}; ECE {test['metrics']['ece']:.4f}",
          f"* meets target: {test['meets_target'] if test['meets_target'] is not None else 'n/a (no target configured)'}",
          f"* largest accuracy gap between slices: {test['max_slice_accuracy_gap']:.4f}", ""]
    path = run_dir / "reports" / "experiment_report.md"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_model_card(run_dir: Path, cfg: dict, context: dict) -> Path:
    run_dir = Path(run_dir)
    sel, test = load_json(run_dir / "selection.json"), load_json(run_dir / "test_evaluation.json")
    win = load_json(run_dir / "algorithms" / sel["winner"] / "result.json")
    L = [f"# Model card: {win['display_name']}", "",
         "> Auto-generated from measured values. Items marked TODO need a human owner.", "",
         "## Intended use", "", "Predict whether an airline passenger reports being satisfied, from flight and survey features. "
         "TODO: owner to confirm intended use, decision supported and out-of-scope uses.", "",
         "## Data", "", f"* rows: {context['rows']:,}; data hash `{context['data_hash'][:16]}`; target `{cfg['data']['target']}`",
         "* TODO: owner to document data source, collection period and known gaps.", "",
         "## Training procedure", "", f"* algorithm: {win['display_name']}; loss: {win['loss_function']}",
         f"* hyper-parameters: `{win['best_params']}`", f"* seed {cfg['project']['seed']}; git `{context.get('git_commit')}`", "",
         "## Performance", "", f"* validation {win['primary_metric']}: {win['val_primary']:.4f}",
         f"* test {test['primary_metric']}: {test['primary_value']:.4f} (n={test['n_test_rows']:,}, evaluated once)",
         f"* accuracy {test['metrics']['accuracy']:.4f}, F1 {test['metrics']['f1']:.4f}, Brier {test['metrics']['brier']:.4f}, ECE {test['metrics']['ece']:.4f}", "",
         "### Slices (test)", ""]
    for col, s in test["slice_metrics"].items():
        L.append(f"* {col}: accuracy gap {s['accuracy_gap']:.4f} (" + ", ".join(f"{k}: {v['accuracy']:.3f}" for k, v in s["levels"].items()) + ")")
    L += ["", "## Limitations", "", "* Survey data: satisfaction is self-reported; ratings of 0 mean 'not applicable'.",
          "* No temporal or passenger/flight identifier was available, so repeated passengers or time drift cannot be controlled.",
          "* Fairness: slice gaps are measured above; no fairness threshold has been agreed. TODO: governance owner.", "",
          "## Monitoring", "", "Run `airsat-predict ... --drift-report` on each scoring batch (PSI vs training reference). TODO: thresholds and retraining owner.", ""]
    path = run_dir / "reports" / "MODEL_CARD.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
