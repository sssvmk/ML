"""Step 16: AG2 agents that interpret every algorithm's results and explain how the winner was declared.

Agents (classic AG2 ConversableAgent API, ag2>=0.9,<1.0)
  tool_executor  - UserProxyAgent that executes tool calls (no code execution, no human input)
  interpreter    - explains one topic at a time to a non-expert, using ONLY facts fetched through tools
  auditor        - re-checks the interpreter's draft against the same tools and reports VERIFIED / ISSUES

Modes
  --mode llm       real LLM (OpenAI-compatible endpoint configured through environment variables)
  --mode offline   deterministic template explanations built from the same facts (no network, used by CI and as fallback)
  --interactive    ask follow-up questions about the run in a terminal chat

    export LLM_MODEL=...  LLM_API_KEY=...  [LLM_BASE_URL=https://gateway.example/v1]  [LLM_DEFAULT_HEADERS='{"x":"y"}']
    python -m airsat.agent.ag2_explainer --run runs/<run_id> --mode llm
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
def _algorithm_task(name: str) -> str:
    return (f"Explain the results of algorithm '{name}'. Fetch its report with get_algorithm_report. Structure the answer as:\n"
            "1. What this algorithm is and how it learns (one short paragraph, plain language).\n"
            "2. The loss function it optimises and what that means.\n"
            "3. Hyper-parameters: what the search found and what each important one controls (use hyperparameter_importance if present).\n"
            "4. Assumption checks: for EACH check say what it tests, the measured value, PASS/WARN/FAIL, and whether any warning matters.\n"
            "5. Performance and generalisation: train vs validation vs cross-validation, the diagnosis verdict, calibration, and slice gaps.\n"
            "6. Top drivers of predictions and what to watch out for.")


ANALYSIS_TASK = ("Explain how the experiment winner was declared. Use get_comparison, get_winner_decision and get_test_evaluation. "
                 "Walk through: (1) the metric and why, (2) the guard-rails, (3) the best raw score and its bootstrap confidence "
                 "interval, (4) the paired comparisons - what a tie means and why a simpler/faster model may win, (5) the final "
                 "one-shot test result and what it does and does not prove, (6) caveats (missing business target, "
                 "assumption warnings, skipped algorithms).")

SUMMARY_TASK = ("Write a one-page executive summary of the whole experiment for a business sponsor: what data, which algorithms were "
                "compared, who won and why, how good the result is on unseen data, key drivers of satisfaction, and the 3 most "
                "important caveats. Use list_algorithms, get_data_findings, get_comparison and get_test_evaluation.")


def explain_run(run_dir: str | Path, cfg: dict, mode: str = "llm", audit: bool = True) -> Path:
    run_dir = Path(run_dir)
    art = RunArtifacts(run_dir)
    out = run_dir / "reports" / "agent_explanations"
    out.mkdir(parents=True, exist_ok=True)
    algos = [a["name"] for a in art.algorithms() if a["status"] == "ok"]
    docs: dict[str, str] = {}

    if mode == "llm":
        try:
            executor, interpreter, auditor = build_agents(art, get_llm_config(cfg))
        except MissingLLMConfig as exc:
            print(f"[agent] {exc}\n[agent] falling back to offline explanations")
            mode = "offline"
    if mode == "llm":
        tasks = {**{a: _algorithm_task(a) for a in algos}, "experiment_analysis": ANALYSIS_TASK, "executive_summary": SUMMARY_TASK}
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
        for key in [*algos, "experiment_analysis", "executive_summary"]:
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
    return _offline_algorithm(art, key)


def _offline_algorithm(art: RunArtifacts, name: str) -> str:
    a = art.algorithm(name)
    if a.get("status") in {"failed", "skipped"}:
        return f"Algorithm `{name}` was **{a['status']}**: {a['message']}\n"
    p, d = a["performance"], a["diagnosis"]
    m = p["primary_metric"]
    L = [f"## {a['display_name']} ({a['family']})", "", "### 1. What it is", a["what_it_is"], "",
         "### 2. Loss function", f"Fact: {a['loss_function']}", f"Optimisation: {a['optimisation']}", "",
         "### 3. Hyper-parameters found", f"Fact: search = {a['search'].get('method')}, {a['search'].get('n_trials_completed', 0)} completed trials.",
         f"Fact: best values: `{a['best_hyperparameters']}`"]
    for k, v in a["best_hyperparameters"].items():
        if k in a["hyperparameter_meaning"]:
            L.append(f"* `{k}` = {v}: {a['hyperparameter_meaning'][k]}")
    if a["hyperparameter_importance"]:
        L.append("Most influential on the score: " + ", ".join(f"{k} ({v:.2f})" for k, v in list(a["hyperparameter_importance"].items())[:3]))
    L += ["", "### 4. Assumption checks"]
    for c in a["assumption_checks"]:
        L.append(f"* **{c['outcome']}** `{c['check']}`: {c['measured']}. Why it matters: {c['why_it_matters']}" + (f" Mitigation: {c['mitigation']}" if c["mitigation"] and c["outcome"] != "PASS" else ""))
    cv = p["cv"]
    L += ["", "### 5. Performance and generalisation",
          f"Fact: {m} train {p['train']:.4f} vs validation {p['validation']:.4f} (gap {p['gap']:+.4f})."
          + (f" Cross-validation: {cv['mean']:.4f} ± {cv['std']:.4f} over {cv['folds']} folds." if cv else ""),
          f"Fact: validation accuracy {p['validation_metrics']['accuracy']}, F1 {p['validation_metrics']['f1']}, Brier {p['validation_metrics']['brier']}, ECE {p['validation_metrics']['ece']}.",
          f"Diagnosis: **{d['verdict']}**. " + " ".join(d["notes"]),
          "Slice accuracy gaps (largest difference between segments): " + (", ".join(f"{k} {v}" for k, v in a["slice_accuracy_gaps"].items()) or "n/a"), "",
          "### 6. What drives predictions", "Permutation importance (drop in AUC when the feature is shuffled): "
          + ", ".join(f"{k} ({v:.3f})" for k, v in a["top_features_permutation"].items()), ""]
    return "\n".join(L) + "\n"


def _offline_analysis(art: RunArtifacts) -> str:
    c, w, t = art.comparison(), art.winner_decision(), art.test_evaluation()
    m = c["primary_metric"]
    L = ["## How the winner was declared", "",
         f"1. **Metric.** Models are ranked on validation **{m}** ({GLOSSARY.get(m, '')})",
         f"2. **Fair comparison.** All models were scored on the same {c['validation_rows']:,} validation rows; the test set was not used for selection.",
         "3. **Evidence.** Each score has a 95% bootstrap interval; differences between models were tested with a paired bootstrap on identical resamples.", "",
         "| Algorithm | Validation | 95% CI | Train-val gap | Flags |", "|---|---|---|---|---|"]
    for n, r in sorted(c["table"].items(), key=lambda kv: -kv[1]["val_primary"]):
        L.append(f"| {r['display_name']} | {r['val_primary']:.4f} | {r['ci95_low']:.4f}-{r['ci95_high']:.4f} | {r['overfit_gap']:+.4f} | {'; '.join(r['disqualified'] + r['flags']) or '-'} |")
    L += ["", "### Decision trace (verbatim from the pipeline)", ""] + [f"* {x}" for x in w["decision_trace"]]
    L += ["", f"**Declared winner: `{w['winner']}`** (raw best score: `{w['raw_best']}`; tie set: {', '.join(w['tie_set'])}).", "",
          GLOSSARY["tie_break"], "", "### Final one-shot test evaluation",
          f"Fact: {t['primary_metric']} = {t['primary_value']:.4f} on {t['n_test_rows']:,} untouched rows (test set evaluated {t['test_set_evaluations']} time(s)).",
          f"Business target: {'not configured - cannot say whether this is good enough' if t['target_value'] is None else t['target_value']}.", "",
          "### Caveats", "* A statistical tie means the data cannot separate those models; the choice then rests on simplicity and speed.",
          "* Assumption warnings are listed per algorithm; none were treated as proof of failure unless the report says so."]
    return "\n".join(L) + "\n"


def _offline_summary(art: RunArtifacts) -> str:
    algos, d, c, t = art.algorithms(), art.data_findings(), art.comparison(), art.test_evaluation()
    L = ["## Executive summary", "", f"* Data: {sum(d['split']['sizes'].values()):,} passengers; splits representative: {d['split']['representative']}.",
         "* Compared: " + ", ".join(f"{a['display_name']} ({a['status']})" for a in algos) + ".",
         f"* Winner: **{c['table'][c['winner']]['display_name']}** - validation {c['primary_metric']} {c['table'][c['winner']]['val_primary']:.4f}; "
         f"test {t['primary_value']:.4f}, accuracy {t['metrics']['accuracy']}, F1 {t['metrics']['f1']}.", "", "Data findings:"]
    L += [f"* {f}" for f in d["eda_findings"][:6]]
    L += ["", "Caveats:", "* Business target not configured." if t["target_value"] is None else f"* Target {t['target_value']}: met = {t['meets_target']}.",
          "* Survey self-reports; no passenger/flight id or date, so repeat passengers and drift cannot be controlled.",
          "* See the per-algorithm pages for assumption warnings."]
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
