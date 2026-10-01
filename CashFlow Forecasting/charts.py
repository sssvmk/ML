"""
Evaluation charts (PRD §5 item 7 / §5.1 evaluate(); gap G-07).

Per algorithm and window: forecast vs actual, residuals, and metric-by-fold.
Charts are written into the config-defined log folder and can be attached
to the MLflow run by the caller. matplotlib is not thread-safe, so the
orchestrator draws charts from the main thread after the parallel backtest
stage has finished (this module itself holds a lock as a second guard).
"""
from __future__ import annotations
from pathlib import Path
import threading

_LOCK = threading.Lock()


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir: Path, stem: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


def forecast_vs_actual_chart(dates, actual, forecast, out_dir: Path, stem: str, title: str) -> Path:
    with _LOCK:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(list(dates), list(actual), marker="o", label="actual")
        ax.plot(list(dates), list(forecast), marker="x", linestyle="--", label="forecast")
        ax.set_title(title)
        ax.legend()
        fig.autofmt_xdate()
        path = _save(fig, out_dir, stem)
        plt.close(fig)
        return path


def residual_chart(dates, residuals, out_dir: Path, stem: str, title: str) -> Path:
    with _LOCK:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(range(len(residuals)), list(residuals))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(range(len(residuals)))
        ax.set_xticklabels([str(d)[:10] for d in dates], rotation=45, ha="right")
        path = _save(fig, out_dir, stem)
        plt.close(fig)
        return path


def metric_by_fold_chart(folds: list[dict], out_dir: Path, stem: str, title: str, metrics=("mase", "smape", "wape")) -> Path | None:
    ok = [f for f in folds if "error" not in f]
    if not ok:
        return None
    with _LOCK:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for m in metrics:
            vals = [f.get(m) for f in ok]
            if any(v is not None for v in vals):
                ax.plot(range(1, len(vals) + 1), [float("nan") if v is None else v for v in vals], marker="o", label=m)
        if "mase" in metrics:
            ax.axhline(1.0, color="red", linewidth=0.8, linestyle=":")
        ax.set_xlabel("backtest fold")
        ax.set_title(title)
        ax.legend()
        path = _save(fig, out_dir, stem)
        plt.close(fig)
        return path
