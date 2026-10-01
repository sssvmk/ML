"""
End-to-end pipeline demo (config.json-driven): loads the adapter named in
config.json -> data_source, extracts, validates, trains all enabled algorithms,
declares the winner per segment, then runs weekly revalidation and daily inference.
"""
from registry import ModelRegistry
from orchestrator import Orchestrator
from config import load_config, build_candidates
from pipeline import Pipeline


def main():
    config = load_config("config.json")
    orch_cfg = config["orchestration"]

    candidates = build_candidates(config)
    print(f"[config] {len(candidates)} algorithm modules enabled+admitted: {sorted(candidates)}")

    registry = ModelRegistry(orch_cfg["registry_dir"])
    orch = Orchestrator(
        registry, orch_cfg["output_dir"], candidates,
        max_parallel_workers=orch_cfg["max_parallel_workers"],
        seasonal_period=orch_cfg["seasonal_period"],
        bias_threshold=orch_cfg["bias_threshold"],
        consistency_min_pass_rate=orch_cfg["consistency_min_pass_rate"],
        holdout_periods=orch_cfg["holdout_periods"],
        window_search_enabled=orch_cfg["window_search_enabled"],
        backtest_years_of_history=orch_cfg["backtest_years_of_history"],
        backtest_step=orch_cfg.get("backtest_step"),
        hyperparameter_search=config.get("hyperparameter_search"),
        mlflow_config=config.get("mlflow"),
        baselines=config.get("baselines"),
    )

    # --- training ---
    pipe = Pipeline(config)
    result = pipe.run(orch, candidates,
                      horizon=orch_cfg["backtest_horizon"],
                      rule_version="demo-v1")
    print(f"[pipeline] {result.summary()}")
    for sid, entry in result.entries.items():
        print(f"  {sid}: winner={entry.algorithm_name} fallback={entry.is_fallback} "
              f"window={entry.window} metrics={entry.metrics}")

    # --- weekly revalidation (same pipeline, weekly cadence) ---
    print("\n[pipeline] weekly revalidation ...")
    for sid, df in result.valid_segments.items():
        e2 = orch.weekly_revalidate(sid, df, horizon=orch_cfg["backtest_horizon"], rule_version="demo-v1")
        print(f"  {sid}: {e2.algorithm_name} (fallback={e2.is_fallback})")

    # --- inference ---
    print("\n[pipeline] daily inference ...")
    forecasts = pipe.infer(orch, horizon=orch_cfg["backtest_horizon"])
    for sid, fc in forecasts.items():
        print(f"  {sid}:\n{fc.to_string(index=False)}")


if __name__ == "__main__":
    main()
