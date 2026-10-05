#!/usr/bin/env python3
"""Classify learning curves from a metrics CSV and write a JSON verdict (plus optional plot).

Input CSV columns (header required):
    epoch, train_loss, val_loss            (required; 'epoch' may be an iteration or size index)
    train_metric, val_metric               (optional; higher-is-better unless --lower-is-better)

Usage:
    python diagnose_curves.py metrics.csv --out reports/diagnosis.json --plot reports/curves.png \
        --target 0.90 [--lower-is-better] [--patience 5]

The verdict is a hypothesis to confirm with worst-error review and the debugging checklist,
not a conclusion.
"""
import argparse
import csv
import json
import math
import sys


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("metrics file is empty")
    needed = {"epoch", "train_loss", "val_loss"}
    missing = needed - set(rows[0].keys())
    if missing:
        sys.exit(f"missing required columns: {sorted(missing)}")
    return rows


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        out.append(float(v) if v not in ("", None) else math.nan)
    return out


def slope(ys):
    """Least-squares slope of ys against index; 0 for fewer than 2 points."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def diagnose(rows, target=None, lower_is_better=False, patience=5):
    tl, vl = col(rows, "train_loss"), col(rows, "val_loss")
    n = len(rows)
    findings, verdicts = [], []

    if any(math.isnan(x) or math.isinf(x) for x in tl + vl):
        return {
            "verdict": ["unstable"],
            "findings": ["NaN or infinite loss found; check learning rate, input scaling, and numerical issues first."],
            "n_points": n,
        }

    best_i = min(range(n), key=lambda i: vl[i])
    since_best = n - 1 - best_i
    final_gap = vl[-1] - tl[-1]
    rel_gap = final_gap / max(abs(tl[-1]), 1e-12)
    tail = max(3, n // 4)
    train_tail_slope = slope(tl[-tail:])
    val_tail_slope = slope(vl[-tail:])
    scale = max(abs(tl[0]), abs(vl[0]), 1e-12)

    info = {
        "n_points": n,
        "best_val_loss": vl[best_i],
        "best_val_index": rows[best_i]["epoch"],
        "final_train_loss": tl[-1],
        "final_val_loss": vl[-1],
        "final_gap": final_gap,
        "relative_gap": rel_gap,
        "points_since_best_val": since_best,
    }

    # Divergence / oscillation
    if tl[-1] > tl[0] * 1.5 or vl[-1] > vl[0] * 1.5 and train_tail_slope > 0:
        verdicts.append("unstable")
        findings.append("Loss increased relative to the start; suspect learning rate too high, bad scaling, or a bug.")
    diffs = [abs(tl[i + 1] - tl[i]) for i in range(n - 1)]
    if n >= 6 and diffs and sum(diffs) / len(diffs) > 0.25 * scale:
        verdicts.append("unstable")
        findings.append("Training loss is oscillating strongly; lower the learning rate or clip gradients.")

    # Flat from the start (possible bug / learning rate far too low)
    if n >= 5 and abs(tl[-1] - tl[0]) < 0.01 * scale and abs(vl[-1] - vl[0]) < 0.01 * scale:
        verdicts.append("not_learning")
        findings.append("Neither loss moved. Run the tiny-batch overfit test and inspect gradients and data flow before tuning.")

    # Overfitting: val best earlier than last, and validation rising while train falls
    overfit = since_best >= patience and val_tail_slope > 0 and train_tail_slope <= 0
    big_gap = rel_gap > 0.3 and final_gap > 0.02 * scale
    if overfit or (big_gap and val_tail_slope >= 0 and train_tail_slope < 0):
        verdicts.append("overfitting")
        findings.append(
            f"Validation loss bottomed at index {rows[best_i]['epoch']} ({since_best} points ago) while train loss kept falling; "
            "use early stopping at the best point, add regularization or data, or reduce capacity."
        )
    elif big_gap:
        verdicts.append("overfitting")
        findings.append(f"Large train/validation gap (relative {rel_gap:.2f}); consider more data or regularization.")

    # Underfitting: small gap, and either target missed or both still high/flat
    metric_info = None
    if "val_metric" in rows[0] and "train_metric" in rows[0]:
        vm, tm = col(rows, "val_metric"), col(rows, "train_metric")
        if not any(math.isnan(x) for x in vm + tm):
            sign = -1.0 if lower_is_better else 1.0
            metric_info = {"final_train_metric": tm[-1], "final_val_metric": vm[-1]}
            if target is not None:
                train_meets = sign * tm[-1] >= sign * target
                val_meets = sign * vm[-1] >= sign * target
                metric_info.update({"target": target, "train_meets_target": train_meets, "val_meets_target": val_meets})
                if not train_meets and "overfitting" not in verdicts:
                    verdicts.append("underfitting")
                    findings.append(
                        "Training metric itself misses the target and there is no large gap; capacity, training time, or features are limiting. "
                        "Rule out a bug or data defect first (worst-error review)."
                    )
                elif train_meets and not val_meets and "overfitting" not in verdicts:
                    verdicts.append("overfitting")
                    findings.append("Training metric meets the target but validation does not: a generalization problem.")
                elif train_meets and val_meets:
                    findings.append("Both training and validation metrics meet the target; confirm once on the untouched test set.")
    if metric_info is None and not verdicts:
        still_improving = train_tail_slope < -0.01 * scale / tail
        if not big_gap and still_improving:
            verdicts.append("still_improving")
            findings.append("Both losses are still falling with a small gap; train longer before changing anything else.")
        elif not big_gap and abs(train_tail_slope) < 1e-3 * scale and tl[-1] > 0.5 * tl[0]:
            verdicts.append("underfitting")
            findings.append("Train loss has plateaued high with a small gap; likely capacity-limited. Check for bugs and data defects first.")

    if not verdicts:
        verdicts.append("healthy")
        findings.append("No red flags in the curves. Compare against the target and the baseline.")

    # Instability or a flat run points to a bug or bad optimization setting; fix that before reasoning about capacity.
    if "unstable" in verdicts or "not_learning" in verdicts:
        verdicts = [v for v in verdicts if v != "underfitting"]
        findings = [f for f in findings if not f.startswith("Training metric itself misses the target")]

    out = {"verdict": sorted(set(verdicts)), "findings": findings, "metrics": info}
    if metric_info:
        out["target_check"] = metric_info
    out["note"] = "Heuristic first pass; confirm with worst-error review and the debugging checklist."
    return out


def plot(rows, path, target=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return False
    x = [r["epoch"] for r in rows]
    has_metric = "val_metric" in rows[0] and "train_metric" in rows[0]
    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(11 if has_metric else 6, 4), squeeze=False)
    ax = axes[0][0]
    ax.plot(x, col(rows, "train_loss"), label="train")
    ax.plot(x, col(rows, "val_loss"), label="validation")
    ax.set_xlabel("epoch / iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    if has_metric:
        ax2 = axes[0][1]
        ax2.plot(x, col(rows, "train_metric"), label="train")
        ax2.plot(x, col(rows, "val_metric"), label="validation")
        if target is not None:
            ax2.axhline(target, linestyle="--", color="gray", label="target")
        ax2.set_xlabel("epoch / iteration")
        ax2.set_ylabel("metric")
        ax2.set_title("Metric")
        ax2.legend()
    # keep tick labels readable for many points
    for a in axes[0]:
        if len(x) > 12:
            a.set_xticks(a.get_xticks()[:: max(1, len(a.get_xticks()) // 8)])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default="diagnosis.json")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()

    rows = read_rows(a.csv)
    result = diagnose(rows, a.target, a.lower_is_better, a.patience)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    if a.plot:
        plot(rows, a.plot, a.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
