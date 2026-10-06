"""Read-only tools the explanation agents use to fetch FACTS from a finished run.
The LLM never computes or recalls numbers: every figure in an explanation comes from one of these functions."""
from __future__ import annotations

import json
from pathlib import Path

from ..utils import NumpyEncoder, load_json

GLOSSARY = {
    "roc_auc": "ROC-AUC: probability that a randomly chosen satisfied passenger gets a higher score than a randomly chosen unsatisfied one. 0.5 = coin flip, 1.0 = perfect ranking. Threshold-free.",
    "pr_auc": "PR-AUC (average precision): area under the precision-recall curve; focuses on the positive class and is more sensitive than ROC-AUC under class imbalance.",
    "log_loss": "Log loss: penalises confident wrong probabilities; the quantity most models directly minimise. Lower is better.",
    "brier": "Brier score: mean squared error of the predicted probability. Lower is better; measures calibration plus sharpness.",
    "ece": "Expected calibration error: average gap between predicted probability and observed frequency. 0 = probabilities can be read at face value.",
    "f1": "F1: harmonic mean of precision and recall at the chosen decision threshold.",
    "mcc": "Matthews correlation: balanced single-number summary of the confusion matrix; -1..1.",
    "overfit_gap": "Train metric minus validation metric. A large positive gap means the model memorised training rows.",
    "bootstrap_ci": "95% bootstrap interval: resample the validation rows many times (same rows for every model) and take the 2.5th/97.5th percentiles of the metric. Shows how much of a difference could be sampling luck.",
    "mcnemar": "McNemar test: compares two classifiers' errors on the same rows; a small p-value means one makes significantly fewer mistakes.",
    "tie_break": "If the best model and a simpler/faster one are statistically indistinguishable (or differ by less than the practical margin), the simpler/faster one is preferred because it is easier to operate and less likely to fail on new data.",
    "adversarial_validation": "A classifier is trained to tell training rows from validation rows. AUC near 0.5 means the two sets look alike (good).",
    "vif": "Variance inflation factor: how much a predictor is explained by the others. Large values make unpenalised linear coefficients unstable (predictions are unaffected).",
    "psi": "Population stability index: how much a feature's distribution moved vs training. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.",
}


def _js(obj) -> str:
    return json.dumps(obj, cls=NumpyEncoder, indent=1)


class RunArtifacts:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

    # ---- plain python accessors (also used by the offline explainer) -----------------------------------------------
    def algorithms(self) -> list[dict]:
        out = []
        for p in sorted((self.run_dir / "algorithms").glob("*/result.json")):
            r = load_json(p)
            out.append({"name": r["name"], "display_name": r["display_name"], "family": r["family"], "status": r["status"],
                        "val_primary": r.get("val_primary"), "message": r.get("message", "")})
        return out

    def algorithm(self, name: str) -> dict:
        p = self.run_dir / "algorithms" / name / "result.json"
        if not p.exists():
            raise KeyError(f"unknown algorithm '{name}'. Known: {[a['name'] for a in self.algorithms()]}")
        r = load_json(p)
        if r["status"] != "ok":
            return {"name": name, "status": r["status"], "message": r["message"]}
        checks = [{"check": a["name"], "outcome": "PASS" if a["passed"] else a["severity"].upper(), "measured": a["detail"],
                   "why_it_matters": a["why_it_matters"], "mitigation": a["mitigation"]} for a in r["assumptions"]]
        vm = r["val_metrics"]
        slices = {c: round(s["accuracy_gap"], 4) for c, s in r["slices"].items()}
        return {
            "name": r["name"], "display_name": r["display_name"], "family": r["family"], "what_it_is": r["description"],
            "loss_function": r["loss_function"], "optimisation": r["optimisation_notes"],
            "hyperparameter_meaning": r["hyperparameter_docs"], "best_hyperparameters": r["best_params"],
            "search": {k: v for k, v in r["search"].items() if k != "hyperparameter_importance"},
            "hyperparameter_importance": r["search"].get("hyperparameter_importance", {}),
            "assumption_checks": checks,
            "performance": {"primary_metric": r["primary_metric"], "train": r["train_primary"], "validation": r["val_primary"],
                            "gap": r["overfit_gap"], "cv": r.get("cv") or None,
                            "validation_metrics": {k: round(v, 4) for k, v in vm.items()}},
            "diagnosis": r["diagnosis"], "top_features_permutation": dict(list(r["importance"]["permutation_auc_drop"].items())[:8]),
            "slice_accuracy_gaps": slices, "latency_ms_per_1000_rows": r["latency_ms_per_1000_rows"],
            "fit_seconds": r["fit_seconds"], "encoded_features": r["n_features_encoded"],
        }

    def comparison(self) -> dict:
        sel = load_json(self.run_dir / "selection.json")
        keep = ("display_name", "family", "val_primary", "ci95_low", "ci95_high", "train_primary", "overfit_gap", "cv_mean",
                "cv_std", "accuracy", "f1", "brier", "latency_ms_per_1000", "complexity_rank", "disqualified", "flags")
        return {"primary_metric": sel["primary_metric"], "validation_rows": sel["validation_rows"],
                "bootstrap_samples": sel["bootstrap_samples"], "table": {t["name"]: {k: t[k] for k in keep} for t in sel["table"]},
                "pairwise_vs_best": sel["pairwise_vs_best"], "raw_best": sel["raw_best"], "winner": sel["winner"]}

    def winner_decision(self) -> dict:
        sel = load_json(self.run_dir / "selection.json")
        return {"winner": sel["winner"], "raw_best": sel["raw_best"], "tie_set": sel["tie_set"], "rules": sel["rules"],
                "relaxed_guardrails": sel["relaxed_guardrails"], "decision_trace": sel["decision_trace"]}

    def test_evaluation(self) -> dict:
        t = load_json(self.run_dir / "test_evaluation.json")
        return {k: t[k] for k in ("algorithm", "n_test_rows", "primary_metric", "primary_value", "target_value", "meets_target",
                                  "max_slice_accuracy_gap", "test_set_evaluations")} | {"metrics": {k: round(v, 4) for k, v in t["metrics"].items()}}

    def data_findings(self) -> dict:
        p = self.run_dir / "eda" / "eda_summary.json"
        pop = load_json(self.run_dir / "splits" / "population_report.json")
        return {"eda_findings": load_json(p)["findings"] if p.exists() else [], "split": {"sizes": pop["sizes"], "representative": pop["representative"], "worst": pop["worst"]}}

    # ---- JSON-string tools exposed to the LLM -----------------------------------------------------------------------------
    def tool_list_algorithms(self) -> str:
        """List every algorithm in this experiment with status and validation primary metric."""
        return _js(self.algorithms())

    def tool_get_algorithm_report(self, name: str) -> str:
        """Full factual report for ONE algorithm: loss, hyper-parameters, assumption checks, metrics, diagnosis, features."""
        return _js(self.algorithm(name))

    def tool_get_comparison(self) -> str:
        """Comparison table across algorithms with bootstrap confidence intervals and paired differences vs the best."""
        return _js(self.comparison())

    def tool_get_winner_decision(self) -> str:
        """The rules and the step-by-step decision trace that produced the declared winner."""
        return _js(self.winner_decision())

    def tool_get_test_evaluation(self) -> str:
        """Final one-shot evaluation of the winner on the untouched test set."""
        return _js(self.test_evaluation())

    def tool_get_data_findings(self) -> str:
        """Key exploratory-data-analysis findings and split representativeness."""
        return _js(self.data_findings())

    def tool_glossary(self, term: str) -> str:
        """Plain-language definition of a metric/technique. term examples: roc_auc, log_loss, bootstrap_ci, tie_break, vif, psi."""
        key = term.strip().lower().replace(" ", "_").replace("-", "_")
        return GLOSSARY.get(key, f"unknown term '{term}'. Known: {sorted(GLOSSARY)}")

    def llm_tools(self) -> list:
        """Plain functions (AG2 needs real functions with type hints, not bound methods)."""
        art = self

        def list_algorithms() -> str:
            """List every algorithm in this experiment with status and validation primary metric."""
            return art.tool_list_algorithms()

        def get_algorithm_report(name: str) -> str:
            """Full factual report for ONE algorithm (use the exact name from list_algorithms): loss, hyper-parameters, assumption checks, metrics, diagnosis, top features."""
            return art.tool_get_algorithm_report(name)

        def get_comparison() -> str:
            """Comparison table across algorithms with bootstrap confidence intervals and paired differences vs the best model."""
            return art.tool_get_comparison()

        def get_winner_decision() -> str:
            """The rules and the step-by-step decision trace that produced the declared winner."""
            return art.tool_get_winner_decision()

        def get_test_evaluation() -> str:
            """Final one-shot evaluation of the winner on the untouched test set."""
            return art.tool_get_test_evaluation()

        def get_data_findings() -> str:
            """Key exploratory-data-analysis findings and split representativeness."""
            return art.tool_get_data_findings()

        def glossary(term: str) -> str:
            """Plain-language definition of a metric or technique, e.g. roc_auc, log_loss, bootstrap_ci, tie_break, vif, psi."""
            return art.tool_glossary(term)

        return [list_algorithms, get_algorithm_report, get_comparison, get_winner_decision, get_test_evaluation,
                get_data_findings, glossary]
