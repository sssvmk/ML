"""
Builds TRACEABILITY.md from the running code itself (algorithms.catalog
+ each class's actual has_eligibility_condition flag), so the matrix
can't silently drift from what's really implemented. Re-run this after
adding or changing an algorithm module.
"""
from pathlib import Path
from algorithms.catalog import ALGORITHM_CATALOG, load_class

INFRA_ROWS = [
    ("contract.py", "PRD §5.2 (Design doc); PRD §3.2 step 1", "Canonical contract schema + universal pre-fit validator"),
    ("adapters/base.py", "Design doc §5.1", "Source adapter abstract interface"),
    ("adapters/synthetic.py", "Design doc §5.1", "Synthetic data generator adapter (demo/test only)"),
    ("adapters/csv_fixture.py", "PRD §4.2 #12", "CSV test-fixture adapter, own column vocabulary"),
    ("adapters/sap_stub.py", "PRD §4.2 #1-5", "Documents the SAP field mapping; no live SAP connectivity in this environment"),
    ("algorithms/base.py", "PRD v10 §5.1; §6", "Abstract base defining the ten algorithm-module interfaces"),
    ("algorithms/utils.py", "Design doc §6 (shared plumbing, not shared test code)", "Endogenous/exogenous series extraction helpers"),
    ("algorithms/catalog.py", "Design doc §5.1 (Orchestrator never knows module internals)", "id -> (module, class, PRD row) catalog for config-driven loading"),
    ("algorithms/seasonal_naive.py", "PRD v10 §3.5", "Seasonal-Naive-7 default baseline (elimination gate); also used as the Seasonal-Naive-30 challenger"),
    ("algorithms/naive_1.py", "PRD v10 §3.5", "Naive-1 challenger baseline (informational only, not the elimination gate)"),
    ("algorithms/rolling_fallback.py", "PRD §3.6", "Rolling mean/median fallback, used only when nothing survives elimination"),
    ("algorithms/_sklearn_template.py", "Shared plumbing (see algorithms/base.py note)", "Lag-feature train/save/load/infer boilerplate for sklearn-style regressors"),
    ("algorithms/_statsmodels_template.py", "Shared plumbing", "SARIMAX-backed fit/forecast boilerplate for the ARIMA-like family"),
    ("algorithms/_neural_template.py", "Implementation plan (Design doc)", "Simplified MLP stand-in backbone for the deep-learning family -- NOT the published architectures"),
    ("registry.py", "PRD §5.3; Design doc §5.2-§5.3", "Model registry: current winner (or fallback) per segment, incl. chosen window and holdout metrics"),
    ("backtest.py", "PRD v10 §3.2, §3.3.4", "Shared rolling-origin backtest harness (six metrics per fold), used by orchestrator.py and search.py"),
    ("ranking.py", "PRD v10 §3.5 steps 1-4; §3.3.6", "MASE/bias elimination + six-metric rank-sum combine, shared by algorithm selection and hyperparameter selection"),
    ("search.py", "PRD v10 §3.3", "Hyperparameter search: initial randomized search + adaptive narrowing + trial pruning, scored via ranking.py"),
    ("parallel.py", "PRD v10 §3.7", "Ray-with-fallback parallel candidate execution (falls back to a thread pool if Ray is unavailable)"),
    ("mlflow_logging.py", "PRD v10 §4.4", "MLflow run logging wrapper; no-ops gracefully when mlflow isn't installed/enabled"),
    ("orchestrator.py", "Design doc §5.3, §6.1; PRD v10 §3.2-§3.7", "Three cadences; per-algorithm window search; elimination/ranking/consistency check; holdout evaluation"),
    ("config.json", "This request: JSON configuration file", "Single source of truth for orchestration params + every algorithm's hyperparameters/enabled flag"),
    ("config.py", "This request: JSON configuration file", "Loads config.json; builds the candidate factory dict from algorithms.catalog"),
    ("demo.py", "Design doc implementation plan", "End-to-end smoke test: adapter -> validator -> orchestrator (all 3 cadences) -> registry -> logs"),
    ("monitor.py", "PRD §4.4 (last para); G-01", "Production monitoring: live six-metric rank-sum of Production vs prior version; proposes a rollback for a human to decide"),
    ("versioning.py", "PRD §4.3; G-08/G-14 area", "Dataset / code version ids recorded on every run, including the nested lineage"),
    ("charts.py", "PRD §5 item 7; G-07", "Required evaluation charts"),
    ("pooling.py", "PRD Overview §1; G-03", "Pool assembly (per-series scale, attributes, categories) and per-fold slicing incl. future-known series"),
    ("admission.py", "D-4 / D-11; G-42", "Admission validation for custom models: Layer 1 (numerical equality), Layer 3 (implementation gates), Layer 2 status; per-model report tied to the module source hash"),
    ("calibration/", "D-11; G-42", "Layer 2 calibration run: simulators with known truth, true-parameter oracles, reference worker (isolated MXNet env), analysis that proposes parameters for approval"),
    ("gap_status.py", "Traceability matrix G-01..G-42", "Status and evidence per gap; drives GAP_STATUS.md"),
    ("config.json -> algorithms.<id>.force_enabled", "This request", "Config-driven override: this algorithm is picked, trained/resolved, and installed as Production unconditionally -- competition, ranking and (for a custom model) G-42 admission are all bypassed, visibly (elimination_log.forced_override)"),
    ("config.json -> baselines", "This request", "Elimination-gate baseline, informational challengers and the fallback model are config-driven, not hard-coded in orchestrator.py"),
    ("config.json -> orchestration.frequency", "This request", "Every date this system produces (future_dates, next_period_after, calendar-feature reindexing, the vendored NeuralForecast/GluonTS freq arguments) reads self.frequency instead of assuming daily; default 'D' preserves prior behaviour."),
    ("pipeline.py (Pipeline / PipelineResult)", "This request", "Config-driven bridge: reads config.json -> data_source, instantiates the right adapter by name, extracts, validates, routes to full_train / full_train_pooled (orchestrator decides based on pooling_capable flags), runs inference. Adapter changes never propagate downstream."),
    ("config.json -> data_source", "This request", "Adapter selection: adapter name, extract() kwargs, optional segment filter. Change this block to switch source without touching any algorithm or orchestration code."),
    ("orchestrator._write_cross_model_comparison()", "This request", "Per-segment: one JSON file comparing every algorithm's metrics, hyperparameters, outcome (winner/ranked/eliminated/error) and rank after each full_train run."),
    ("orchestrator._write_cross_segment_summary()", "This request", "After full_train_pooled: one JSON file showing winner algorithm and metrics across all segments, plus winner distribution."),
    ("contract.Dataset (opened)", "This request", "Dataset is now an open string field: any non-empty string accepted from non-cashflow adapters. The 6 cashflow enum constants are preserved as Dataset.BANK_INFLOW_EBS etc. for existing adapters."),
    ("algorithms/nf_adapter.py", "G-41; D-6", "NeuralForecast models behind the ten interfaces (validation tail, best-weights restore, any horizon, gap policy)"),
    ("algorithms/nf_pooled.py", "G-03/G-25", "Pooling-capable NeuralForecast base: static covariates, positional future-known channels"),
    ("algorithms/_futr_required.py", "G-28/G-29", "Eligibility derived from future-known covariate series (structure-inferred)"),
    ("algorithms/_frames.py", "Shared plumbing", "Contract rows -> regular daily frames, calendar features, explicit gap policy"),
    ("algorithms/_vendor.py", "G-41; D-9 (S2)", "Loader for the vendored, Ray-optional NeuralForecast 3.2.2"),
    ("algorithms/_vendor_etsformer.py", "G-36; D-12", "Loader for the vendored official ETSformer under a private module name"),
    ("algorithms/torch_base.py", "Shared plumbing", "Native-PyTorch windowing, per-window scaling and trainer (TCN, ETSformer)"),
    ("algorithms/prebuilt.py", "D-8 / D-10", "Path-only pretrained-model resolution with fingerprints and the non-commercial licence guard"),
    ("vendor/nf_patched/", "G-41; D-9", "NeuralForecast 3.2.2, Apache-2.0, 3 files patched (STC_PATCH.md)"),
    ("vendor/etsformer/", "G-36; D-12", "Official ETSformer, BSD-3, unmodified (PROVENANCE.md, OSS_SCAN.md)"),
]


def write_gap_status() -> str:
    from gap_status import GAPS
    order = ["closed", "closed_with_deviation", "partial", "built_not_admitted", "built_unavailable", "open"]
    counts = {k: sum(1 for v in GAPS.values() if v[1] == k) for k in order}
    L = ["# Gap status (traceability matrix G-01 .. G-42)", "",
         "Generated by `generate_traceability.py` from `gap_status.py`; `tests/test_docs_status.py` checks that every gap is listed, that every cited test file exists and that this file is current.",
         "", "**Summary:** " + ", ".join(f"{counts[k]} {k.replace('_', ' ')}" for k in order if counts[k]), "",
         "| Gap | What | Status | Evidence (tests) | Note |", "| --- | --- | --- | --- | --- |"]
    for gid, (title, status, tests, note) in GAPS.items():
        L.append(f"| {gid} | {title} | {status.replace('_', ' ')} | {', '.join('`' + t + '`' for t in tests)} | {note} |")
    return "\n".join(L) + "\n"


def traceability_text() -> str:
    lines = [
        "# Traceability matrix",
        "",
        "Generated by `generate_traceability.py` from the running code -- not hand-maintained.",
        "",
        "## Infrastructure",
        "",
        "| File | Requirement / section | What it implements |",
        "| --- | --- | --- |",
    ]
    for f, req, desc in INFRA_ROWS:
        lines.append(f"| `{f}` | {req} | {desc} |")

    from algorithms.catalog import EXTENSION_CATALOG
    import inspect

    def traits(cls):
        t = []
        if getattr(cls, "is_simplified_stand_in", False):
            t.append("SIMPLIFIED STAND-IN (sklearn MLP)")
        if getattr(cls, "pooling_capable", False):
            t.append("pooled")
        if getattr(cls, "joint_inference", False):
            t.append("joint (infer_pool)")
        if getattr(cls, "requires_admission", False):
            t.append("requires G-42 admission")
        if getattr(cls, "prebuilt_key", None):
            t.append(f"pretrained weights from config ({cls.prebuilt_key})")
        return ", ".join(t) or "-"

    def rows(catalog):
        out = []
        for algo_id, (module_path, class_name, row_num) in sorted(catalog.items(), key=lambda kv: kv[1][2]):
            cls = load_class(algo_id)
            file_path = module_path.replace("algorithms.", "algorithms/") + ".py"
            has_elig = "yes" if cls.has_eligibility_condition else "no"
            out.append(f"| {row_num} | {class_name.replace('Module','')} | `{file_path}` | `{class_name}` | {has_elig} | {traits(cls)} |")
        return out

    head = ["| # | Algorithm | File | Class | Has eligibility condition | Implementation notes |", "| --- | --- | --- | --- | --- | --- |"]
    lines += ["", f"## Algorithm modules (the PRD §3.2 manifest of {len(ALGORITHM_CATALOG)})", ""] + head + rows(ALGORITHM_CATALOG)
    if EXTENSION_CATALOG:
        lines += ["", "## Optional roster additions OUTSIDE the PRD's manifest", "", "Kept in a separate catalog so the PRD count stays exactly what the PRD says (gap G-40 needs the PRD owner).", ""] + head + rows(EXTENSION_CATALOG)

    flagged = sorted(row for _, (_, _, row) in ALGORITHM_CATALOG.items() if load_class(next(k for k, v in ALGORITHM_CATALOG.items() if v[2] == row)).has_eligibility_condition)

    def ranges(nums):
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}" if i == j else f"{nums[i]}-{nums[j]}")
            i = j + 1
        return ", ".join(out)

    lines += [
        "",
        f"**{len(ALGORITHM_CATALOG)} algorithm modules in the PRD manifest; {len(flagged)} have a genuine eligibility condition (rows {ranges(flagged)}), "
        f"{len(ALGORITHM_CATALOG) - len(flagged)} have none.** Derived from each class's `has_eligibility_condition`; the PRD row-level table governs (decision on G-15).",
        "",
        "Every module also implements `required_observations()` and, where declared, `hyperparameter_search_space()` (§3.3.1).",
    ]

    reports = sorted(Path("admission_reports").glob("*.json")) if Path("admission_reports").exists() else []
    if reports:
        import json
        lines += ["", "## Admission reports (G-42)", "", "| Model | Layer 1 | Layer 2 | Layer 3 | Admitted | Source hash |", "| --- | --- | --- | --- | --- | --- |"]
        for f in reports:
            r = json.loads(f.read_text())
            l2 = r["layer2"].get("status", "passed" if r["layer2"].get("passed") else "failed")
            lines.append(f"| {r['algorithm']} | {'pass' if r['layer1']['passed'] else 'FAIL'} | {l2} | {'pass' if r['layer3']['passed'] else 'FAIL'} | "
                         f"{'yes' if r['admitted'] else 'no'} | `{r['module_sha256'][:12]}` |")
    return "\n".join(lines) + "\n"


def main():
    Path("TRACEABILITY.md").write_text(traceability_text())
    Path("GAP_STATUS.md").write_text(write_gap_status())
    print("wrote TRACEABILITY.md and GAP_STATUS.md")


if __name__ == "__main__":
    main()
