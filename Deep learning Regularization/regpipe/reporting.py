"""Plots, diagnosis, comparison tables and the model card."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def diagnose(history: list) -> dict:
    """First-look verdict from the learning curves; a hypothesis, not a conclusion."""
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    ta = [h["train_acc"] for h in history]
    va = [h["val_acc"] for h in history]
    verdict, notes = [], []
    if any(math.isnan(x) or math.isinf(x) for x in tl + vl):
        return {"verdict": ["unstable"], "notes": ["non-finite loss"]}
    rise = vl[-1] - min(vl)
    gap = ta[-1] - va[-1]
    if rise > 0.02 and vl.index(min(vl)) < len(vl) - 1:
        verdict.append("overfitting")
        notes.append(f"validation loss rose {rise:.3f} above its minimum (epoch {vl.index(min(vl)) + 1})")
    if gap > 0.03:
        verdict.append("generalization_gap")
        notes.append(f"train accuracy exceeds validation accuracy by {gap:.3f}")
    if ta[-1] < 0.90:
        verdict.append("underfitting")
        notes.append(f"train accuracy only {ta[-1]:.3f}")
    if len(vl) > 3 and sum(1 for a, b in zip(vl[1:], vl[:-1]) if a > b * 1.2) > len(vl) / 3:
        verdict.append("unstable")
        notes.append("validation loss jumps repeatedly")
    return {"verdict": verdict or ["healthy"], "notes": notes,
            "final": {"train_loss": tl[-1], "val_loss": vl[-1], "train_acc": ta[-1], "val_acc": va[-1]}}


def plot_curves(history: list, path: Path, title: str) -> Path:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train (cross-entropy)")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation (cross-entropy)")
    ax[0].plot(ep, [h["train_obj"] for h in history], ls=":", label="optimised objective")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch"); ax[1].legend()
    ax[2].plot(ep, [h["lr"] for h in history]); ax[2].set(title="Learning rate", xlabel="epoch")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_overview(histories: dict, path: Path) -> Path:
    n = len(histories)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, h) in zip(axes.flat, histories.items()):
        ax.axis("on")
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], label="train")
        ax.plot(ep, [r["val_loss"] for r in h], label="val")
        ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Train vs validation cross-entropy per method"); fig.tight_layout()
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


def plot_comparison(rows: list, path: Path) -> Path:
    names = [r["method"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.38 * len(rows) + 2))
    acc = [r["test"]["accuracy"] for r in rows]
    se = [r["test"]["accuracy_se"] for r in rows]
    colors = ["tab:gray" if r["method"] == "baseline" else ("tab:orange" if not r["champion_eligible"] else "tab:blue")
              for r in rows]
    ax[0].barh(names, acc, xerr=se, color=colors); ax[0].invert_yaxis()
    ax[0].set(title="Test accuracy ± SE (orange = reduced-data regime)", xlim=(min(acc) - 0.02, 1.0))
    auc = [r["test"]["auc"] for r in rows]
    ax[1].barh(names, auc, color=colors); ax[1].invert_yaxis()
    ax[1].set(title="Test AUC (macro one-vs-rest)", xlim=(min(auc) - 0.005, 1.0)); ax[1].set_yticklabels([])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def results_markdown(rows: list, baseline_acc: float | None) -> str:
    head = ("| # | Method | Book § | Regime | Val acc | Test acc ± SE | Test AUC | Test loss | FGSM acc (val) | Tuned hyperparameters |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for i, r in enumerate(rows):
        t = r["test"]
        hp = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["best_hp"].items())
        lines.append(f"| {i} | {r['title']} | {r['section']} | {r['regime']} | {r['val']['accuracy']:.4f} | "
                     f"{t['accuracy']:.4f} ± {t['accuracy_se']:.4f} | {t['auc']:.5f} | {t['loss']:.4f} | "
                     f"{r['val_fgsm_accuracy']:.3f} | {hp} |")
    return head + "\n".join(lines) + "\n"


def model_card(summary: dict) -> str:
    b = summary["best"]
    return f"""# Model card: MNIST digit classifier ({b['title']})

*Generated from logged results. Items marked TODO need a human owner.*

## Model
- Fully connected ReLU network trained with SGD + momentum and exponential learning-rate decay.
- Selected regularization method: **{b['title']}** (book section {b['section']}), selected by validation accuracy among full-data methods.
- Hyperparameters (tuned with Optuna on the validation set): `{b['best_hp']}`

## Intended use
- TODO: owner to complete. This pipeline is a study of regularization methods on MNIST; it is not validated for any production decision.

## Data
- MNIST, source `{summary['data']['source']}`; the dataset ships pre-split into train/test only, so a stratified validation set was carved from train.
- Sizes: {summary['data']['sizes']}; data hash `{summary['data']['data_hash']}`.
- Audit: duplicates train/val={summary['audit']['duplicates_train_val']}, train/test={summary['audit']['duplicates_train_test']}, val/test={summary['audit']['duplicates_val_test']}.

## Metrics (test set, evaluated once for this model)
- Accuracy: {b['test']['accuracy']:.4f} ± {b['test']['accuracy_se']:.4f} (SE)
- AUC (macro one-vs-rest): {b['test']['auc']:.5f}, 95% bootstrap CI {b['test'].get('auc_ci95')}
- Cross-entropy: {b['test']['loss']:.4f}
- FGSM (eps={summary['robustness_eps']}) accuracy on validation: {b['val_fgsm_accuracy']:.3f}

## Limitations
- Handwritten digits only; no evidence about other image types or distribution shift.
- Slice, calibration and fairness analyses were not performed. TODO: owner to decide whether they are needed.

## Monitoring plan
- TODO: owner to define input-drift and performance-decay thresholds.

## Version
- Code commit: `{summary['tags'].get('git_commit')}`; torch {summary['tags'].get('torch')}; mlflow {summary['tags'].get('mlflow')}.
"""
