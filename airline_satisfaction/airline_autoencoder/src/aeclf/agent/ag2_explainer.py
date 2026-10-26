"""AG2 agents that interpret every encoder-decoder and classifier result and explain how the winner was declared.

Agents (classic AG2 ConversableAgent API, ag2>=0.9,<1.0)
  tool_executor  - UserProxyAgent that executes tool calls (no code execution, no human input)
  interpreter    - explains one topic at a time to a non-expert, using ONLY facts fetched through tools
  auditor        - re-checks the interpreter's draft against the same tools and reports VERIFIED / ISSUES

Modes
  --mode llm       real LLM (OpenAI-compatible endpoint configured through environment variables)
  --mode offline   deterministic template explanations built from the same facts (no network, used by CI and as fallback)
  --interactive    ask follow-up questions about the run in a terminal chat

    export LLM_MODEL=...  LLM_API_KEY=...  [LLM_BASE_URL=https://gateway.example/v1]  [LLM_DEFAULT_HEADERS='{"x":"y"}']
    python -m aeclf.agent.ag2_explainer --run runs/<run_id> --mode llm
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import load_config
from .tools import GLOSSARY, RunArtifacts

TERMINATE = "TERMINATE"

INTERPRETER_PROMPT = f"""You are an ML-experiment interpreter. You explain the results of an airline-passenger-satisfaction modelling \
experiment to a business stakeholder who has no machine-learning background, and you also stay precise enough for a data scientist.

HARD RULES
1. FACTS COME FROM TOOLS. Call the tools to fetch facts. Never invent, recall or estimate a number, parameter or test result.
2. Every number you write must appear in a tool result. If a tool does not provide something, say it is not available.
3. Define each technical term in one plain sentence the first time you use it (call the glossary tool when unsure).
4. Keep FACT and INTERPRETATION visibly separate (e.g. a line starting 'Fact:' and a line starting 'What this means:').
5. Be honest about limits: unconfirmed business target, single dataset, validation vs test, statistical ties, warnings that fired.
6. Use short sections and plain language. No marketing tone, no certainty beyond the evidence.
When the explanation is complete, put the word {TERMINATE} alone on the last line."""

AUDITOR_PROMPT = f"""You are an independent auditor of ML explanations. You receive a DRAFT explanation. Use the tools to re-fetch the \
facts and check EVERY number, parameter, test outcome and causal claim in the draft.
Reply in this format:
VERDICT: VERIFIED   (or)   VERDICT: ISSUES
Then a numbered list of statements that are wrong, unsupported by tool output, or over-claimed, each with the corrected wording.
If everything is supported, say so in one sentence. Never add new claims that the tools do not support.
End with the word {TERMINATE} alone on the last line."""


# ----------------------------------------------------------------------------------------------------------------------
class MissingLLMConfig(RuntimeError):
    pass


def get_llm_config(cfg: dict):
    from autogen import LLMConfig

    l = cfg["llm"]
    model, key, base = os.environ.get(l["model_env"]), os.environ.get(l["api_key_env"]), os.environ.get(l["base_url_env"])
    if not model or not (key or base):
        raise MissingLLMConfig(f"Set {l['model_env']} and {l['api_key_env']} (and {l['base_url_env']} for a gateway/proxy) to use --mode llm.")
    entry = {"model": model, "api_type": "openai", "api_key": key or "not-needed"}
    if base:
        entry["base_url"] = base
    headers = os.environ.get(l["default_headers_env"])
    if headers:
        entry["default_headers"] = json.loads(headers)
    return LLMConfig(entry, temperature=l.get("temperature", 0.0), timeout=120)


def build_agents(art: RunArtifacts, llm_config):
    from autogen import ConversableAgent, UserProxyAgent, register_function

    executor = UserProxyAgent("tool_executor", human_input_mode="NEVER", code_execution_config=False, llm_config=False,
                              max_consecutive_auto_reply=12, is_termination_msg=lambda m: TERMINATE in (m.get("content") or ""))
    interpreter = ConversableAgent("interpreter", system_message=INTERPRETER_PROMPT, llm_config=llm_config, human_input_mode="NEVER")
    auditor = ConversableAgent("auditor", system_message=AUDITOR_PROMPT, llm_config=llm_config, human_input_mode="NEVER")
    for fn in art.llm_tools():
        for caller in (interpreter, auditor):
            register_function(fn, caller=caller, executor=executor, name=fn.__name__, description=(fn.__doc__ or fn.__name__).strip())
    return executor, interpreter, auditor


def _ask(executor, agent, message: str, max_turns: int = 14, clear: bool = True) -> str:
    res = executor.initiate_chat(agent, message=message, max_turns=max_turns, summary_method="last_msg", silent=True, clear_history=clear)
    return (res.summary or "").replace(TERMINATE, "").strip()


# ----------------------------------------------------------------------------------------------------------------------
def _autoencoder_task(kind: str) -> str:
    return (f"Explain the autoencoder variant '{kind}'. Fetch it with get_autoencoder_report. Structure the answer as:\n"
            "1. What this encoder-decoder is and how it learns without the satisfaction label (plain language).\n"
            "2. The loss function: what each term measures and why the ratings are treated as classes.\n"
            "3. Hyper-parameters the search found and what the important ones control.\n"
            "4. Sanity checks: for EACH say what it tests, the measured value, PASS/WARN/FAIL and whether it matters.\n"
            "5. Reconstruction quality: how faithfully each kind of column is rebuilt versus the trivial baseline (mean / most common value); weakest columns.\n"
            "6. What this means for using the latent code as input to a classifier.")


def _candidate_task(name: str) -> str:
    return (f"Explain classifier candidate '{name}'. Fetch it with get_candidate_report (and the glossary for terms). Cover: how it is built "
            "(encoder source, frozen/fine-tuned/scratch), the loss, head hyper-parameters, train vs validation metrics, the diagnosis, "
            "calibration/slice gaps, number of trainable parameters, and how it compares with the control.")


ANALYSIS_TASK = ("Explain how the experiment winner was declared. Use get_comparison, get_winner_decision, get_pretraining_benefit and get_test_evaluation. "
                 "Walk through: (1) the metric and why, (2) the bootstrap intervals and what a tie means, (3) why a smaller/faster model may win a tie, "
                 "(4) whether autoencoder pretraining actually helped versus the no-pretraining control and in the label-efficiency study, "
                 "(5) the final one-shot test result and what it does and does not prove, (6) caveats.")

SUMMARY_TASK = ("Write a one-page executive summary for a business sponsor: what was built (encoder-decoder pretraining then a classifier), what the data looks like, "
                "which variants were compared, who won and why, whether the autoencoder helped, performance on unseen data, and the 3 most important caveats. "
                "Use list_models, get_data_findings, get_comparison, get_pretraining_benefit and get_test_evaluation.")


def explain_run(run_dir: str | Path, cfg: dict, mode: str = "llm", audit: bool = True) -> Path:
    run_dir = Path(run_dir)
    art = RunArtifacts(run_dir)
    out = run_dir / "reports" / "agent_explanations"
    out.mkdir(parents=True, exist_ok=True)
    kinds = [p.parent.name for p in sorted((run_dir / "autoencoders").glob("*/result.json"))]
    cands = [c["name"] for c in art.candidates() if c["status"] == "ok" and c["role"] != "baseline"]
    docs: dict[str, str] = {}

    if mode == "llm":
        try:
            executor, interpreter, auditor = build_agents(art, get_llm_config(cfg))
        except MissingLLMConfig as exc:
            print(f"[agent] {exc}\n[agent] falling back to offline explanations")
            mode = "offline"
    if mode == "llm":
        tasks = {**{f"autoencoder_{k}": _autoencoder_task(k) for k in kinds}, **{f"candidate_{c}": _candidate_task(c) for c in cands},
                 "experiment_analysis": ANALYSIS_TASK, "executive_summary": SUMMARY_TASK}
        for key, task in tasks.items():
            try:
                draft = _ask(executor, interpreter, task)
                text = f"{draft}\n"
                if audit:
                    verdict = _ask(executor, auditor, f"Audit this DRAFT about '{key}':\n\n{draft}")
                    text += f"\n---\n### Independent audit\n{verdict}\n"
                docs[key] = text
            except Exception as exc:  # network/gateway/auth problems must not lose the run
                docs[key] = offline_doc(art, key) + f"\n> LLM call failed ({type(exc).__name__}: {exc}); showing the deterministic explanation instead.\n"
            print(f"[agent] wrote {key}")
    else:
        for key in [*(f"autoencoder_{k}" for k in kinds), *(f"candidate_{c}" for c in cands), "experiment_analysis", "executive_summary"]:
            docs[key] = offline_doc(art, key)

    index = ["# Agent explanations", "", f"Mode: **{mode}**", ""]
    for key, text in docs.items():
        (out / f"{key}.md").write_text(f"# {key}\n\n{text}", encoding="utf-8")
        index.append(f"* [{key}]({key}.md)")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return out


def interactive(run_dir: str | Path, cfg: dict):
    art = RunArtifacts(run_dir)
    executor, interpreter, _ = build_agents(art, get_llm_config(cfg))
    print("Ask about this run (e.g. 'why did XGBoost lose?', 'what does the VIF warning mean?'). Empty line quits.")
    first = True
    while True:
        q = input("you> ").strip()
        if not q:
            break
        print("agent>", _ask(executor, interpreter, q, clear=first), "\n")
        first = False


# ---------------------------------------------------------------------------------------------------------------------
# Deterministic explanations (same facts, fixed templates)
# ---------------------------------------------------------------------------------------------------------------------
def offline_doc(art: RunArtifacts, key: str) -> str:
    if key == "experiment_analysis":
        return _offline_analysis(art)
    if key == "executive_summary":
        return _offline_summary(art)
    if key.startswith("autoencoder_"):
        return _offline_autoencoder(art, key[len("autoencoder_"):])
    return _offline_candidate(art, key[len("candidate_"):])


def _offline_autoencoder(art: RunArtifacts, kind: str) -> str:
    r = art.autoencoder(kind)
    if r.get("status") in {"failed", "skipped"}:
        return f"Autoencoder `{kind}` was **{r['status']}**: {r['message']}\n"
    s = r["reconstruction_summary"]
    L = [f"## {r['display_name']} (`{kind}`)", "", "### 1. What it is", GLOSSARY["denoising_autoencoder" if kind == "dae" else "vae" if kind == "vae" else "autoencoder"], "",
         "### 2. Loss function", f"Fact: {r['loss']}.", GLOSSARY["reconstruction_loss"], "",
         "### 3. Hyper-parameters", f"Fact: `{r['best_hyperparameters']}` ({r['search']['method']}, {r['search']['n_trials_completed']} trials, {r['epochs_trained']} epochs trained)."]
    if r["hyperparameter_importance"]:
        L.append("Most influential on reconstruction: " + ", ".join(f"{k} ({v:.2f})" for k, v in list(r["hyperparameter_importance"].items())[:3]))
    L += ["", "### 4. Sanity checks"]
    for c in r["sanity_checks"]:
        L.append(f"* **{c['outcome']}** `{c['check']}`: {c['measured']}. Why it matters: {c['why_it_matters']}" + (f" Mitigation: {c['mitigation']}" if c["mitigation"] and c["outcome"] != "PASS" else ""))
    L += ["", "### 5. Reconstruction quality (validation)",
          f"Fact: continuous columns R² {s['mean_continuous_r2']:.3f}; categorical accuracy {s['mean_categorical_accuracy']:.3f} vs majority baseline {s['mean_majority_baseline_accuracy']:.3f}; exact categorical row match {s['exact_row_match_categorical']:.3f}.",
          "Per continuous column (original units): " + "; ".join(f"{k} RMSE {v['rmse']} vs mean-predictor {v['baseline_rmse_mean_predictor']}" for k, v in r["continuous_columns"].items()),
          "Weakest categorical columns: " + ", ".join(f"{k} ({v['accuracy']:.3f} vs {v['baseline_accuracy_majority']:.3f})" for k, v in r["weakest_categorical_columns"].items()), "",
          "### 6. What it means", "A latent code is useful to a classifier only if it keeps what predicts satisfaction; reconstruction quality is a proxy, not a guarantee. The classifier comparison decides."]
    return "\n".join(L) + "\n"


def _offline_candidate(art: RunArtifacts, name: str) -> str:
    r = art.candidate(name)
    if r.get("status") in {"failed", "skipped"}:
        return f"Candidate `{name}` was **{r['status']}**: {r['message']}\n"
    v = r["validation_metrics"]
    L = [f"## {r['display_name']} (`{name}`)", "", f"* mode: {r['mode']} - {GLOSSARY.get({'frozen': 'frozen', 'finetune': 'finetune', 'scratch': 'scratch_control'}.get(r['mode'], r['mode']), '')}",
         f"* loss: {r['loss']}", f"* head hyper-parameters: `{r['head_hyperparameters']}`; trainable parameters {r['trainable_parameters']:,}",
         f"* {r['primary_metric']}: train {r['train']:.4f}, validation {r['validation']:.4f} (gap {r['gap']:+.4f}); accuracy {v['accuracy']}, F1 {v['f1']}, Brier {v['brier']}, ECE {v['ece']}",
         f"* diagnosis: **{r['diagnosis']['verdict']}** " + " ".join(r["diagnosis"]["notes"]),
         "* slice accuracy gaps: " + (", ".join(f"{k} {x}" for k, x in r["slice_accuracy_gaps"].items()) or "n/a")]
    return "\n".join(L) + "\n"


def _offline_analysis(art: RunArtifacts) -> str:
    c, w, t, pb = art.comparison(), art.winner_decision(), art.test_evaluation(), art.pretraining_benefit()
    m = c["primary_metric"]
    L = ["## How the winner was declared", "", f"1. **Metric.** Ranked on validation **{m}**. {GLOSSARY.get(m, '')}",
         f"2. **Fair comparison.** All models scored on the same {c['validation_rows']:,} validation rows; the test set played no part in selection.",
         "3. **Evidence.** 95% bootstrap intervals and paired bootstrap differences on identical resamples.", "",
         "| Model | Validation | 95% CI | Train-val gap | Trainable params | Flags |", "|---|---|---|---|---|---|"]
    for n, r in sorted(c["table"].items(), key=lambda kv: -kv[1]["val_primary"]):
        L.append(f"| {r['display_name']} | {r['val_primary']:.4f} | {r['ci95_low']:.4f}-{r['ci95_high']:.4f} | {r['overfit_gap']:+.4f} | {r['trainable_params']:,} | {'; '.join(r['flags']) or '-'} |")
    L += ["", "### Decision trace (verbatim)", ""] + [f"* {x}" for x in w["decision_trace"]]
    L += ["", f"**Winner: `{w['winner']}`** (raw best `{w['raw_best']}`; tie set {', '.join(w['tie_set'])}). {GLOSSARY['tie_break']}", "", "### Did autoencoder pretraining help?"]
    L += [f"* {b['candidate']}: {b['diff_vs_scratch']:+.4f} vs {b.get('control', 'control')}, CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] -> {b['verdict']}" for b in pb["vs_scratch_control"]] or ["* no control available"]
    if pb["label_efficiency"]:
        L.append("Label efficiency (mean validation AUC): " + "; ".join(f"{r['setup']} @ {r['label_fraction']:g}: {r['mean']:.3f}" for r in pb["label_efficiency"]["table"]))
    L += ["", "### Final one-shot test evaluation", f"Fact: {t['primary_metric']} = {t['primary_value']:.4f} on {t['n_test_rows']:,} untouched rows (evaluated {t['test_set_evaluations']} time).",
          f"Business target: {'not configured - cannot say whether this is good enough' if t['target_value'] is None else t['target_value']}.", "",
          "### Caveats", "* A tie means the data cannot separate those models; simplicity and speed then decide.", "* Autoencoder pretraining uses all training rows' features (no labels); that is allowed but is why label efficiency is reported separately."]
    return "\n".join(L) + "\n"


def _offline_summary(art: RunArtifacts) -> str:
    d, c, t, pb = art.data_findings(), art.comparison(), art.test_evaluation(), art.pretraining_benefit()
    win = c["table"][c["winner"]]
    L = ["## Executive summary", "", f"* Data: {sum(d['split']['sizes'].values()):,} passengers; splits representative: {d['split']['representative']}. Columns encoded: {len(d['encoding']['continuous'])} continuous, {len(d['encoding']['categorical'])} categorical.",
         f"* Approach: learn a compact passenger code by reconstructing the record (autoencoder variants), then classify satisfaction from the code; compared with a same-size network trained on labels only.",
         f"* Winner: **{win['display_name']}** - validation {c['primary_metric']} {win['val_primary']:.4f}; test {t['primary_value']:.4f}, accuracy {t['metrics']['accuracy']}, F1 {t['metrics']['f1']}.",
         "* Pretraining verdicts: " + "; ".join(f"{b['candidate']}: {b['verdict']}" for b in pb["vs_scratch_control"]), "", "Data findings:"]
    L += [f"* {f}" for f in d["eda_findings"][:5]]
    L += ["", "Caveats:", "* Business target not configured." if t["target_value"] is None else f"* Target {t['target_value']}: met = {t['meets_target']}.",
          "* Survey self-reports; no passenger/flight id or date.", "* The classifier only sees the latent code."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (e.g. runs/run_2026...)")
    ap.add_argument("--mode", choices=["llm", "offline"], default="llm")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.interactive:
        interactive(a.run, cfg)
    else:
        print("wrote", explain_run(a.run, cfg, a.mode, audit=not a.no_audit))


if __name__ == "__main__":
    main()
