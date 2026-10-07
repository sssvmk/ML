"""Read-only tools the explanation agents use to fetch FACTS from a finished run. The LLM never computes or recalls numbers."""
from __future__ import annotations

import json
from pathlib import Path

from ..utils import NumpyEncoder, load_json

GLOSSARY = {
    "autoencoder": "Autoencoder: a network that squeezes each passenger record into a short code (the latent vector) and then tries to rebuild the record from that code. It learns without using the satisfaction label.",
    "denoising_autoencoder": "Denoising autoencoder: the input is deliberately damaged (noise, hidden values) and the network must rebuild the CLEAN record, which forces it to learn how the columns relate rather than copy them.",
    "vae": "Variational autoencoder: like an autoencoder but each record maps to a small probability cloud; a KL penalty (weight beta) keeps the codes smooth. Too large a beta can make the code carry almost no information (posterior collapse).",
    "latent": "Latent code: the short numeric summary of a passenger produced by the encoder; the classifier only sees this.",
    "reconstruction_loss": "Reconstruction loss: squared error for continuous columns plus cross-entropy for categorical columns (including the 0-5 ratings, which are treated as classes). Lower = the record is rebuilt more faithfully.",
    "r2": "R²: share of a continuous column's variation the reconstruction explains; 0 = no better than always predicting the average, 1 = perfect.",
    "frozen": "Frozen encoder: encoder weights fixed after pretraining; only the classification head learns (a 'probe' of how useful the code is).",
    "finetune": "Fine-tuned encoder: the encoder keeps learning from the satisfaction labels, with a smaller learning rate than the head.",
    "scratch_control": "Control: the same network trained from random weights on the labels only. If pretraining does not beat it, the autoencoder step added nothing.",
    "lgbm_latent": "LightGBM on the latent code: gradient-boosted trees trained on the encoder's short code only; the encoder stays frozen.",
    "lgbm_hybrid": "LightGBM on latent + raw features: trees see the latent code AND the original encoded columns; an upper bound on what the encoder can add.",
    "lgbm_raw": "Raw-feature LightGBM control: trees on the original columns with no encoder at all. If the encoder-based models do not beat it, the latent code adds nothing for tree models.",
    "label_efficiency": "Label efficiency: validation AUC when only a fraction of the labelled rows is used for the supervised stage. Pretraining is expected to help most when labels are scarce.",
    "roc_auc": "ROC-AUC: probability that a random satisfied passenger gets a higher score than a random unsatisfied one. 0.5 = coin flip, 1 = perfect.",
    "log_loss": "Log loss: penalises confident wrong probabilities. Lower is better.",
    "brier": "Brier score: mean squared error of predicted probabilities. Lower is better.",
    "ece": "Expected calibration error: gap between predicted probability and observed frequency. 0 = probabilities can be read at face value.",
    "bootstrap_ci": "95% bootstrap interval: resample the validation rows many times (same rows for every model) and take the 2.5th/97.5th percentiles; shows how much of a difference could be luck.",
    "tie_break": "If the best model and a smaller/faster one are statistically indistinguishable (or differ by less than the practical margin), the smaller/faster one is preferred.",
    "posterior_collapse": "Posterior collapse: the VAE ignores the input and the latent carries no information; detected by counting latent dimensions that actually vary.",
}


def _js(o) -> str:
    return json.dumps(o, cls=NumpyEncoder, indent=1)


class RunArtifacts:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def candidates(self) -> list[dict]:
        return [load_json(p) for p in sorted((self.run_dir / "candidates").glob("*/result.json"))]

    def autoencoder(self, kind: str) -> dict:
        p = self.run_dir / "autoencoders" / kind / "result.json"
        if not p.exists():
            raise KeyError(f"unknown autoencoder '{kind}'. Known: {[q.parent.name for q in (self.run_dir / 'autoencoders').glob('*/result.json')]}")
        r = load_json(p)
        if r["status"] != "ok":
            return {"kind": kind, "status": r["status"], "message": r["message"]}
        rep = load_json(self.run_dir / "autoencoders" / kind / "reconstruction_report.json")
        worst = sorted(rep["categorical"].items(), key=lambda kv: kv[1]["accuracy"] - kv[1]["baseline_accuracy_majority"])[:3]
        return {"kind": kind, "display_name": r["display_name"], "best_hyperparameters": r["params"], "search": {k: v for k, v in r["search"].items() if k != "hyperparameter_importance"},
                "hyperparameter_importance": r["search"].get("hyperparameter_importance", {}), "best_validation_reconstruction_loss": r["best_val_recon"],
                "epochs_trained": r["epochs_trained"], "reconstruction_summary": r["reconstruction"],
                "continuous_columns": {k: {m: round(v, 3) for m, v in c.items()} for k, c in rep["continuous"].items()},
                "weakest_categorical_columns": {k: v for k, v in worst},
                "sanity_checks": [{"check": c["name"], "outcome": "PASS" if c["passed"] else c["severity"].upper(), "measured": c["detail"], "why_it_matters": c["why_it_matters"],
                                   "mitigation": c["mitigation"]} for c in r["sanity"]],
                "loss": "weighted sum: MSE on standardised continuous columns + w_cat * mean cross-entropy on categorical columns" + (" + beta*KL" if kind == "vae" else "") + (" (input corrupted, clean target)" if kind == "dae" else "")}

    def candidate(self, name: str) -> dict:
        p = self.run_dir / "candidates" / name / "result.json"
        if not p.exists():
            raise KeyError(f"unknown candidate '{name}'. Known: {[c['name'] for c in self.candidates()]}")
        r = load_json(p)
        if r["status"] != "ok":
            return {"name": name, "status": r["status"], "message": r["message"]}
        return {"name": name, "display_name": r["display_name"], "role": r["role"], "mode": r["mode"], "pretrained_autoencoder": r.get("pretrained_autoencoder"),
                "pretrained_reconstruction": r.get("pretrained_reconstruction"), "head_hyperparameters": r["head_params"],
                "search": {k: v for k, v in r["search"].items() if k != "hyperparameter_importance"}, "loss": r["loss_function"],
                "primary_metric": r["primary_metric"], "train": r["train_primary"], "validation": r["val_primary"], "gap": r["overfit_gap"],
                "validation_metrics": {k: round(v, 4) for k, v in r["val_metrics"].items()}, "diagnosis": r["diagnosis"], "trainable_parameters": r["trainable_params"],
                "slice_accuracy_gaps": {c: round(s["accuracy_gap"], 4) for c, s in r["slices"].items()}, "latency_ms_per_1000_rows": r["latency_ms_per_1000_rows"]}

    def comparison(self) -> dict:
        s = load_json(self.run_dir / "selection.json")
        keep = ("display_name", "role", "val_primary", "ci95_low", "ci95_high", "train_primary", "overfit_gap", "accuracy", "f1", "brier", "trainable_params", "latency_ms_per_1000", "flags")
        return {"primary_metric": s["primary_metric"], "validation_rows": s["validation_rows"], "winner": s["winner"], "raw_best": s["raw_best"],
                "table": {t["name"]: {k: t[k] for k in keep} for t in s["table"]}, "pairwise_vs_best": s["pairwise_vs_best"]}

    def pretraining_benefit(self) -> dict:
        s = load_json(self.run_dir / "selection.json")
        p = self.run_dir / "label_efficiency.json"
        return {"vs_scratch_control": s["pretraining_benefit"], "label_efficiency": load_json(p) if p.exists() else None}

    def winner_decision(self) -> dict:
        s = load_json(self.run_dir / "selection.json")
        return {k: s[k] for k in ("winner", "raw_best", "tie_set", "rules", "decision_trace")}

    def test_evaluation(self) -> dict:
        t = load_json(self.run_dir / "test_evaluation.json")
        out = {k: t[k] for k in ("candidate", "n_test_rows", "primary_metric", "primary_value", "target_value", "meets_target", "max_slice_accuracy_gap", "test_set_evaluations")}
        out["metrics"] = {k: round(v, 4) for k, v in t["metrics"].items()}
        out["reconstruction_on_test"] = (t.get("pretrained_autoencoder_reconstruction_on_test") or {}).get("summary")
        return out

    def data_findings(self) -> dict:
        p = self.run_dir / "eda" / "eda_summary.json"
        pop = load_json(self.run_dir / "splits" / "population_report.json")
        return {"eda_findings": load_json(p)["findings"] if p.exists() else [], "split": {"sizes": pop["sizes"], "representative": pop["representative"]},
                "encoding": load_json(self.run_dir / "encoding.json")}

    def llm_tools(self) -> list:
        a = self

        def list_models() -> str:
            """List the autoencoders and classifier candidates in this experiment with status and validation metric."""
            return _js({"autoencoders": [q.parent.name for q in sorted((a.run_dir / "autoencoders").glob("*/result.json"))],
                        "candidates": [{"name": c["name"], "display_name": c["display_name"], "status": c["status"], "val_primary": c.get("val_primary")} for c in a.candidates()]})

        def get_autoencoder_report(kind: str) -> str:
            """Report for ONE autoencoder variant (ae, dae or vae): loss, hyper-parameters, reconstruction fidelity, sanity checks."""
            return _js(a.autoencoder(kind))

        def get_candidate_report(name: str) -> str:
            """Report for ONE classifier candidate (exact name from list_models): head hyper-parameters, metrics, diagnosis, slices."""
            return _js(a.candidate(name))

        def get_comparison() -> str:
            """Comparison table with bootstrap confidence intervals and paired differences against the best model."""
            return _js(a.comparison())

        def get_pretraining_benefit() -> str:
            """Does autoencoder pretraining beat the no-pretraining control? Includes the label-efficiency study."""
            return _js(a.pretraining_benefit())

        def get_winner_decision() -> str:
            """The rules and step-by-step decision trace that produced the declared winner."""
            return _js(a.winner_decision())

        def get_test_evaluation() -> str:
            """Final one-shot evaluation of the winner on the untouched test set."""
            return _js(a.test_evaluation())

        def get_data_findings() -> str:
            """Key exploratory-data-analysis findings, split representativeness and how columns were encoded."""
            return _js(a.data_findings())

        def glossary(term: str) -> str:
            """Plain-language definition, e.g. autoencoder, denoising_autoencoder, vae, latent, reconstruction_loss, r2, frozen, finetune, scratch_control, label_efficiency, roc_auc, bootstrap_ci, tie_break."""
            k = term.strip().lower().replace(" ", "_").replace("-", "_")
            return GLOSSARY.get(k, f"unknown term '{term}'. Known: {sorted(GLOSSARY)}")

        return [list_models, get_autoencoder_report, get_candidate_report, get_comparison, get_pretraining_benefit, get_winner_decision,
                get_test_evaluation, get_data_findings, glossary]
