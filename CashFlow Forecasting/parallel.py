"""
Parallel candidate-training execution (PRD v10 §3.7).

  Ray
   |
   v (if Ray unavailable/uninitializable)
  best available native/platform parallel mechanism
   |
   v
  Train eligible algorithms concurrently
   |
   v
  Consolidate results

Ray is preferred and tried first; the moment it cannot be imported or
initialized, this falls back to a ThreadPoolExecutor (the "best
available native mechanism" in a plain-Python environment -- the numeric
libraries every module here uses -- statsmodels/sklearn/xgboost/
lightgbm/catboost -- release the GIL during their fit's numeric-heavy
portion, so threads still give real wall-clock parallelism). Ray is
NOT a mandatory dependency (§3.7 point 5): this module works whether or
not `ray` is installed, and the fallback is exercised automatically,
not configured by hand.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


def _try_ray():
    try:
        import ray
    except ImportError:
        return None
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level="ERROR", include_dashboard=False)
        return ray
    except Exception:
        return None


def parallel_run(jobs: dict[str, Callable[[], dict]], max_workers: int = 8) -> tuple[dict[str, dict], str]:
    """
    Runs each zero-arg callable in `jobs` concurrently. Returns
    (results, backend) where backend is "ray" or "threadpool" -- callers
    log which one actually ran, per §3.7 point 6-9 (the framework choice
    must not change eligibility/backtest/metric/selection rules, only
    execution).
    """
    ray = _try_ray()
    results: dict[str, dict] = {}

    if ray is not None:
        try:
            @ray.remote
            def _run(fn):
                return fn()

            futures = {name: _run.remote(fn) for name, fn in jobs.items()}
            for name, fut in futures.items():
                try:
                    results[name] = ray.get(fut)
                except Exception as exc:  # one task failing must not stop the others (§3.7 point 8)
                    results[name] = {"error": f"ray task raised: {exc}"}
            return results, "ray"
        except Exception:
            results = {}  # fall through to threadpool if Ray broke mid-run

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                results[name] = {"error": f"backtest raised: {exc}"}
    return results, "threadpool"
