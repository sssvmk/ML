"""
MLflow helpers for the MLP pipeline: experiment setup, run tags/params, per-epoch
logging, evaluation artifacts (confusion matrix / residual plots / worst errors),
and the model registry with a promotion gate.
"""
import csv
import io
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from metrics import classification_metrics, paired_mse_difference, regression_metrics
from mlp_core import PRIMARY, make_loss

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


# --------------------------------------------------------------------------- #
# Setup, tags, params
# --------------------------------------------------------------------------- #
def setup(tracking_uri, experiment, default_dir=None):
    """Point MLflow at a tracking store and select the experiment.

    Store: the explicit `tracking_uri`, else $MLFLOW_TRACKING_URI, else (if `default_dir` is given) a
    SQLite file inside that folder with artifacts next to it, so the whole run lives in one result
    folder; else sqlite:///mlflow.db in the working directory.
    """
    user_env = os.environ.get("MLFLOW_TRACKING_URI")  # what the caller set, if anything
    uri = tracking_uri or user_env
    artifact_location = None
    if not uri and default_dir is not None:
        folder = Path(default_dir).resolve()
        uri = "sqlite:///" + (folder / "mlflow.db").as_posix()
        artifact_location = (folder / "mlartifacts").as_uri()
    uri = uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    # mlflow.set_tracking_uri also WRITES the MLFLOW_TRACKING_URI environment variable. Undo that, otherwise a
    # second run in the same process would mistake it for a user-supplied store and ignore its own result folder.
    if user_env is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = user_env
    if artifact_location and MlflowClient().get_experiment_by_name(experiment) is None:
        MlflowClient().create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)
    return uri


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def env_tags(device):
    return {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mlflow": mlflow.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }


def flat_params(cfg):
    """Config dict -> MLflow-loggable params (lists become strings)."""
    out = {}
    for k, v in cfg.items():
        out[k] = str(list(v)) if isinstance(v, (list, tuple)) else v
    out["n_hidden_layers"] = len(cfg["hidden"])
    return out


def log_epoch(row):
    """fit() callback: one set of metrics per epoch, with step=epoch."""
    mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=row["epoch"])


# --------------------------------------------------------------------------- #
# Figures / diagnosis
# --------------------------------------------------------------------------- #
def curves_figure(rows, task):
    ep = [r["epoch"] for r in rows]
    key = PRIMARY[task]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="validation")
    ax[0].set(title="Loss", xlabel="epoch",
              ylabel="cross-entropy" if task == "classification" else "MSE (standardized target)")
    ax[1].plot(ep, [r[f"train_{key}"] for r in rows], label="train (running avg)")
    ax[1].plot(ep, [r[f"val_{key}"] for r in rows], label="validation")
    ax[1].set(title=key, xlabel="epoch", ylabel=key)
    ax[2].plot(ep, [r["lr"] for r in rows], color="tab:green")
    ax[2].set(title="Learning rate", xlabel="epoch", yscale="log")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    return fig


def diagnose(rows, best_epoch):
    """Transparent heuristic only: compares train and validation loss at the best epoch."""
    b = rows[best_epoch - 1]
    ratio = b["val_loss"] / max(b["train_loss"], 1e-12)
    return {
        "best_epoch": best_epoch, "epochs_run": len(rows),
        "train_loss_at_best": b["train_loss"], "val_loss_at_best": b["val_loss"],
        "val_over_train_loss": ratio,
        "verdict": "possible overfitting (val/train loss > 1.3)" if ratio > 1.3
                   else "no large train/validation gap",
        "note": "heuristic; train loss is a running average over the epoch (with dropout active)",
    }


# --------------------------------------------------------------------------- #
# Evaluation (final test pass) with logged artifacts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).cpu())
        ys.append(y.cpu())
    return torch.cat(outs), torch.cat(ys)


def _csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def evaluate_and_log(model, loader, info, device, prefix="test", n_boot=200, seed=0):
    """Evaluate once on `loader`, log metrics + diagnostic artifacts to the active run.

    classification: ROC-AUC (macro one-vs-rest, bootstrap CI), accuracy +- SE, log-loss +- SE,
                    macro-F1, per-class AUC / accuracy, ROC curves, confusion matrix, worst errors
    regression    : MSE +- SE, RMSE/MAE +- SE, R2, paired MSE difference vs the linear baseline
                    (when available), predicted-vs-actual and residual plots, largest errors
    Returns the scalar metrics dict (python floats).
    """
    task = info["task"]
    out, y = _forward(model, loader, device)
    loss = make_loss(task)(out, y).item()

    if task == "classification":
        probs = torch.softmax(out, dim=1)
        pred = probs.argmax(1)
        metrics = classification_metrics(probs.numpy(), y.numpy(), n_boot=n_boot, seed=seed)
        k = info["out_dim"]
        cm = np.zeros((k, k), dtype=int)
        for t, p_ in zip(y.numpy(), pred.numpy()):
            cm[t, p_] += 1
        per_class = {f"{prefix}_acc_class_{i}": float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(k)}

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set(title=f"Confusion matrix ({prefix})", xlabel="predicted", ylabel="true",
               xticks=range(k), yticks=range(k))
        if k <= 12:
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        mlflow.log_figure(fig, "evaluation/confusion_matrix.png")
        plt.close(fig)
        mlflow.log_text(_csv_text(["true\\pred"] + list(range(k)),
                                  [[i] + cm[i].tolist() for i in range(k)]),
                        "evaluation/confusion_matrix.csv")

        # One-vs-rest ROC curves
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(6, 5))
        for c in range(k):
            pos = (y.numpy() == c)
            if 0 < pos.sum() < len(pos):
                fpr, tpr, _ = roc_curve(pos.astype(int), probs[:, c].numpy())
                ax.plot(fpr, tpr, lw=1, label=f"class {c} (AUC {metrics.get(f'auc_class_{c}', float('nan')):.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=f"One-vs-rest ROC ({prefix}); macro AUC {metrics['auc_macro_ovr']:.4f}",
               xlabel="false positive rate", ylabel="true positive rate")
        if k <= 12:
            ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "evaluation/roc_curves.png")
        plt.close(fig)

        wrong = (pred != y).nonzero(as_tuple=True)[0]
        order = wrong[probs[wrong].max(1).values.argsort(descending=True)][:25]
        mlflow.log_text(_csv_text(
            ["row_index", "true", "predicted", "confidence"],
            [[int(i), int(y[i]), int(pred[i]), round(float(probs[i].max()), 4)] for i in order]),
            "evaluation/worst_errors.csv")
        extra = per_class
    else:
        p = (out * info["y_std"] + info["y_mean"]).numpy().ravel()
        t = (y * info["y_std"] + info["y_mean"]).numpy().ravel()
        metrics = regression_metrics(p, t)
        base = info.get("baseline_test_pred")
        if base is not None and len(base) == len(p):
            diff, diff_se = paired_mse_difference(p, base, t)
            metrics["mse_diff_vs_linear"] = diff
            metrics["mse_diff_vs_linear_se"] = diff_se
        res = p - t
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(t, p, s=4, alpha=0.4)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax[0].plot(lim, lim, color="tab:red", lw=1)
        lab = info.get("target_label", "target")
        ax[0].set(title=f"Predicted vs actual ({prefix})", xlabel=f"actual: {lab}", ylabel=f"predicted: {lab}")
        ax[1].hist(res, bins=50)
        ax[1].set(title="Residuals (predicted - actual)", xlabel=f"error: {lab}")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "evaluation/regression_diagnostics.png")
        plt.close(fig)
        order = np.argsort(-np.abs(res))[:25]
        mlflow.log_text(_csv_text(
            ["row_index", "actual", "predicted", "error"],  # original target units
            [[int(i), round(float(t[i]), 4), round(float(p[i]), 4), round(float(res[i]), 4)] for i in order]),
            "evaluation/worst_errors.csv")
        extra = {}

    metrics = {k_: float(v) for k_, v in metrics.items()}
    logged = {f"{prefix}_loss": loss, **{f"{prefix}_{k_}": v for k_, v in metrics.items()}, **extra}
    mlflow.log_metrics(logged)
    return {"loss": float(loss), **metrics}


# --------------------------------------------------------------------------- #
# Registry: verification and promotion gate
# --------------------------------------------------------------------------- #
def verify_registered_model(model_uri, example_input, expected, atol=1e-4):
    """Reload the model from the registry and check it reproduces `expected` on the example."""
    loaded = mlflow.pyfunc.load_model(model_uri)
    got = loaded.predict(example_input)
    cols = [c for c in expected.columns if c in got.columns]
    if not cols:
        return False, "output has none of the expected columns"
    for c in cols:
        if c == "label":
            ok = (got[c].to_numpy() == expected[c].to_numpy()).all()
        else:
            ok = np.allclose(got[c].to_numpy(dtype=float), expected[c].to_numpy(dtype=float), atol=atol)
        if not ok:
            return False, f"column {c!r} differs from the original model's output"
    return True, "reloaded model matches the original on the input example"


def check_targets(task, test_metrics, min_auc=None, max_mse=None):
    """Optional acceptance targets on the final test metrics. Returns (ok, message).

    The targets themselves come from the problem (what the model must achieve), not from
    the data, so they are set by the user: --min-auc (classification), --max-mse (regression).
    """
    checks = []
    if task == "classification" and min_auc is not None:
        v = test_metrics["auc_macro_ovr"]
        checks.append((v >= min_auc, f"test AUC {v:.4f} (95% CI {test_metrics['auc_ci95_low']:.4f}-"
                                     f"{test_metrics['auc_ci95_high']:.4f}) vs required >= {min_auc}"))
    if task == "regression" and max_mse is not None:
        v = test_metrics["mse"]
        checks.append((v <= max_mse, f"test MSE {v:.4f} +- {test_metrics['mse_se']:.4f} (SE) "
                                     f"vs required <= {max_mse}"))
    if not checks:
        return True, "no acceptance target set"
    return all(ok for ok, _ in checks), "; ".join(m for _, m in checks)


def register_and_promote(client, model_name, version, run_id, final_val_loss, fingerprint,
                         verified, verify_msg, log=print, target_ok=True,
                         target_msg="no acceptance target set", margin=0.001):
    """Tag the new version, set alias `challenger`, and promote to `champion` if gates pass.

    Gates (all must hold):
      1. reload-and-predict verification passed
      2. the optional acceptance target (check_targets) is met
      3. no champion yet, OR same data fingerprint as the champion and a validation loss that is
         lower by at least `margin` (relative, default 0.1%), so retraining noise cannot swap champions
    Validation loss (not test) is compared, so the test set is not used for repeated selection.
    """
    version = str(version)
    for k, v in {"validated": str(verified).lower(), "validation_note": verify_msg,
                 "final_val_loss": f"{final_val_loss:.6f}", "data_fingerprint": fingerprint,
                 "acceptance_target": target_msg}.items():
        client.set_model_version_tag(model_name, version, k, v)
    client.set_registered_model_alias(model_name, "challenger", version)

    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        champ = None

    if not verified:
        decision = "not promoted: reload-and-predict verification failed"
    elif not target_ok:
        decision = f"not promoted: acceptance target not met ({target_msg})"
    elif champ is None:
        decision = "promoted: first version, no champion yet"
    elif champ.tags.get("data_fingerprint") != fingerprint:
        decision = ("not promoted: data/split fingerprint differs from the champion "
                    "(validation losses are not comparable); promote manually if intended")
    else:
        champ_loss = float(champ.tags.get("final_val_loss", "inf"))
        # Both sides at the precision stored in the tag (6 decimals), so an identical model ties.
        beats = round(final_val_loss, 6) < champ_loss * (1.0 - margin)
        decision = (f"promoted: val loss {final_val_loss:.4f} beats champion v{champ.version} "
                    f"{champ_loss:.4f} by >= {margin:.1%}" if beats else
                    f"not promoted: val loss {final_val_loss:.4f} does not beat champion "
                    f"v{champ.version} {champ_loss:.4f} by the required {margin:.1%} margin")

    promoted = decision.startswith("promoted")
    if promoted:
        client.set_registered_model_alias(model_name, "champion", version)
    client.set_model_version_tag(model_name, version, "promotion_decision", decision)
    log(f"registry: {model_name} v{version} -> {decision}")
    return promoted, decision
