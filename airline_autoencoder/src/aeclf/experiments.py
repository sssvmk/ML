"""Steps 4-8: run every experiment under one protocol and log everything.

Experiment grid
  Stage 1  for kind in {ae, dae, vae}: tune + train the autoencoder (unsupervised, reconstruction loss)
  Stage 2  for kind x mode in {frozen, finetune}: tune + train the classification head on the encoder
  Control  scratch: SAME architecture, random initialisation, trained only on the labels -> does pretraining help at all?
  Floor    majority-class baseline
  Extra    label-efficiency curves (AUC vs fraction of labels used) for the best AE-based model vs the scratch control

Hyper-parameter search: Optuna TPE (Bayesian) with a median pruner on per-epoch validation loss; the documented defaults are
enqueued as trial 0, so the search can never end up worse than the starting point.
"""
from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .bundle import ModelBundle, compute_reference_stats
from .encoding import TabularEncoder
from .metrics import choose_threshold, compute_metrics, primary_value, slice_metrics
from .plots import plot_calibration, plot_confusion, plot_curve, plot_optuna_history, plot_roc_pr
from .schema import Schema
from .tracking import Tracker
from .training import (Tensors, build_ae, post_training_checks, pre_training_checks, predict_proba, reconstruction_report,
                       latent_stats, train_autoencoder, train_classifier)
from .lgbm_head import LGBM_KEYS, LGBM_MODES, build_features, leaf_count_proxy, lgbm_space, require_lightgbm, train_lgbm
from .models import count_trainable
from .utils import dump_json

AE_BASE_KEYS = ["hidden", "depth", "latent", "dropout", "lr", "weight_decay", "batch_size", "w_cat"]
AE_EXTRA = {"ae": [], "dae": ["noise_sigma", "mask_p"], "vae": ["beta"]}
KIND_NAMES = {"ae": "Autoencoder", "dae": "Denoising autoencoder", "vae": "Variational autoencoder"}
MODE_NAMES = {"frozen": "frozen encoder", "finetune": "fine-tuned encoder", "scratch": "no pretraining (control)",
              "lgbm_latent": "LightGBM on latent code", "lgbm_hybrid": "LightGBM on latent + raw features", "lgbm_raw": "LightGBM on raw features, no encoder (control)"}


@dataclass
class ExperimentData:
    X_train: pd.DataFrame            # RAW rows (before encoding), TRAIN
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    schema: Schema
    encoder: TabularEncoder
    tr: Tensors
    va: Tensors
    cfg: dict
    run_meta: dict = field(default_factory=dict)

    @property
    def arch_static(self) -> dict:
        return {"n_cont": len(self.encoder.cont_cols), "cards": self.encoder.cards,
                "emb_max": self.cfg["encoding"]["embedding_max_dim"]}


# ------------------------------------------------------------------------------------------------ search spaces
def ae_space(trial, kind: str) -> dict:
    p = {"hidden": trial.suggest_categorical("hidden", [64, 128, 256]), "depth": trial.suggest_int("depth", 1, 3),
         "latent": trial.suggest_categorical("latent", [8, 16, 32]), "dropout": trial.suggest_float("dropout", 0.0, 0.3),
         "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True), "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
         "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]), "w_cat": trial.suggest_float("w_cat", 0.25, 4.0, log=True)}
    if kind == "dae":
        p["noise_sigma"] = trial.suggest_float("noise_sigma", 0.05, 0.5)
        p["mask_p"] = trial.suggest_float("mask_p", 0.05, 0.3)
    if kind == "vae":
        p["beta"] = trial.suggest_float("beta", 1e-3, 0.1, log=True)      # capped: reconstruction-only tuning would otherwise push beta -> 0
    return p


def head_space(trial, mode: str) -> dict:
    p = {"head_hidden": trial.suggest_categorical("head_hidden", [0, 32, 64]), "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),
         "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True), "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])}
    p["enc_lr_mult"] = trial.suggest_float("enc_lr_mult", 0.02, 1.0, log=True) if mode == "finetune" else 1.0
    return p


def run_search(objective, defaults: dict, n_trials: int, timeout: int, seed: int, direction: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3))
    study.enqueue_trial(defaults)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))
    done = [t for t in study.trials if t.value is not None]
    if not done:
        reason = next((t.system_attrs.get("fail_reason") for t in study.trials if t.system_attrs.get("fail_reason")), "unknown")
        raise RuntimeError(f"hyper-parameter search produced no successful trial; first failure: {reason}")
    try:
        imp = optuna.importance.get_param_importances(study) if len(done) >= 4 else {}
    except Exception:
        imp = {}
    info = {"method": "Optuna TPE (Bayesian) + median pruner", "n_trials_requested": n_trials, "n_trials_completed": len(done),
            "n_trials_pruned_or_failed": len(study.trials) - len(done), "best_value": study.best_value, "hyperparameter_importance": imp}
    return dict(study.best_params), info, study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))


def _subsample(t: Tensors, n: int, seed: int) -> Tensors:
    if len(t) <= n:
        return t
    return t.take(torch.randperm(len(t), generator=torch.Generator().manual_seed(seed))[:n])


# ------------------------------------------------------------------------------------------------ stage 1
def pretrain(kind: str, data: ExperimentData, tracker: Tracker, out_dir: Path, device: str) -> dict:
    cfg, seed = data.cfg, data.cfg["project"]["seed"]
    acfg = cfg["autoencoder"]
    d = out_dir / "autoencoders" / kind
    (d / "plots").mkdir(parents=True, exist_ok=True)
    keys = AE_BASE_KEYS + AE_EXTRA[kind]
    defaults = {k: acfg["defaults"][k] for k in keys}
    res: dict = {"kind": kind, "display_name": KIND_NAMES[kind], "status": "ok", "message": ""}
    t0 = time.perf_counter()
    try:
        with tracker.run(f"pretrain_{kind}", nested=True, tags={"phase": "autoencoder", "kind": kind}):
            pre = pre_training_checks(kind, data.arch_static, {**acfg["defaults"]}, data.tr, cfg, seed, device)
            blocking = [c for c in pre if c["blocking"]]
            if blocking:
                raise RuntimeError("blocking sanity check failed: " + "; ".join(c["detail"] for c in blocking))
            sub, vsub = _subsample(data.tr, acfg["search"]["train_subsample"], seed), _subsample(data.va, 50000, seed)

            def objective(trial):
                p = {**acfg["defaults"], **ae_space(trial, kind)}
                return train_autoencoder(kind, p, data.arch_static, sub, vsub, cfg, seed, device, trial=trial)[2]

            best, info, trials = run_search(objective, defaults, acfg["search"]["n_trials"], acfg["search"]["timeout_seconds"], seed, "minimize")
            params = {**acfg["defaults"], **best}
            model, hist, best_val = train_autoencoder(kind, params, data.arch_static, data.tr, data.va, cfg, seed, device)
            trials.to_csv(d / "search_trials.csv", index=False)
            hist.to_csv(d / "training_curve.csv", index=False)
            rep, examples = reconstruction_report(model, data.encoder, data.X_val, "cpu", cfg["reconstruction"]["examples"])
            post = post_training_checks(kind, rep, model, data.va, cfg)
            examples.to_csv(d / "reconstruction_examples.csv", index=False)
            dump_json(rep, d / "reconstruction_report.json")
            dump_json(pre + post, d / "sanity_checks.json")
            mu, y = latent_stats(model, data.va)
            res.update(params={k: params[k] for k in keys}, search=info, best_val_recon=best_val, reconstruction=rep["summary"],
                       reconstruction_columns=rep, sanity=pre + post, epochs_trained=len(hist),
                       fit_seconds=time.perf_counter() - t0, state={k: v for k, v in model.state_dict().items()},
                       latent_dim=int(params["latent"]),
                       arch={**data.arch_static, "kind": kind, "hidden": params["hidden"], "depth": params["depth"],
                             "latent": params["latent"], "dropout": params["dropout"]})
            res["artifacts"] = {
                "training_curve_plot": plot_curve(hist, "iteration", ["train_loss", "val_loss"], d / "plots" / "training_curve.png",
                                                  f"{KIND_NAMES[kind]}: reconstruction loss", "weighted reconstruction loss"),
                "search_history_plot": plot_optuna_history(trials.assign(value=-trials["value"]), d / "plots" / "search_history.png", f"{kind}: -val recon (higher=better)"),
                "latent_plot": _plot_latent(mu, y, d / "plots" / "latent_pca.png", kind),
                "reconstruction_plot": _plot_reconstruction(rep, d / "plots" / "reconstruction_fidelity.png", kind),
                "reconstruction_examples": str(d / "reconstruction_examples.csv")}
            tracker.log_params({f"best.{k}": v for k, v in res["params"].items()})
            tracker.log_metrics({"val_recon": best_val, **{f"recon.{k}": v for k, v in rep["summary"].items()}})
            for k in ("training_curve_plot", "latent_plot", "reconstruction_plot"):
                tracker.log_artifact(res["artifacts"][k], "plots")
    except Exception as exc:
        res.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        (d / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    public = {k: v for k, v in res.items() if k not in ("state", "reconstruction_columns")}
    dump_json(public, d / "result.json")
    return res


def _plot_latent(mu: np.ndarray, y, path, kind) -> str:
    from sklearn.decomposition import PCA

    z = PCA(n_components=2).fit_transform(mu) if mu.shape[1] > 2 else mu
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, s=3, alpha=0.4, cmap="coolwarm")
    ax.set(title=f"{KIND_NAMES[kind]}: latent space (PCA of validation codes)", xlabel="PC1", ylabel="PC2")
    fig.colorbar(sc, label="satisfied")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)


def _plot_reconstruction(rep: dict, path, kind) -> str:
    cont = pd.Series({k: v["r2"] for k, v in rep["continuous"].items()})
    cat = pd.Series({k: v["accuracy"] - v["baseline_accuracy_majority"] for k, v in rep["categorical"].items()})
    fig, ax = plt.subplots(1, 2, figsize=(11, max(3.5, 0.3 * len(cat))))
    cont.sort_values().plot.barh(ax=ax[0]); ax[0].set(title="Continuous: R² vs mean predictor", xlabel="R² (0 = trivial)")
    cat.sort_values().plot.barh(ax=ax[1]); ax[1].set(title="Categorical: accuracy gain over majority level", xlabel="accuracy - majority baseline")
    fig.suptitle(f"{KIND_NAMES[kind]} reconstruction fidelity (validation)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------------------ stage 2
def _auc(y, p) -> float:
    return primary_value("roc_auc", y, p)


def run_candidate(name: str, kind: str | None, mode: str, ae: dict | None, data: ExperimentData, tracker: Tracker,
                  out_dir: Path, device: str, role: str) -> dict:
    """One classifier candidate: tune head (or LightGBM) -> train -> evaluate -> bundle.

    mode: frozen | finetune | scratch (neural head)   or   lgbm_latent | lgbm_hybrid | lgbm_raw (LightGBM head)."""
    cfg, seed = data.cfg, data.cfg["project"]["seed"]
    ccfg = cfg["classifier"]
    is_lgbm = mode in LGBM_MODES
    d = out_dir / "candidates" / name
    (d / "plots").mkdir(parents=True, exist_ok=True)
    if mode == "scratch":
        display = "Same network, no pretraining (control)"
    elif mode == "lgbm_raw":
        display = "LightGBM on raw features, no encoder (control)"
    else:
        display = f"{KIND_NAMES[kind]} + {MODE_NAMES[mode]}"
    res = {"name": name, "kind": kind, "mode": mode, "role": role, "family": "control" if role == "control" else "autoencoder", "display_name": display,
           "status": "ok", "message": "", "primary_metric": cfg["metric"]["primary"],
           "loss_function": ("binary logistic loss (LightGBM, gradient-boosted trees) + L1/L2 leaf penalties" if is_lgbm else "binary cross-entropy (BCEWithLogits) on the satisfaction logit")}
    try:
        with tracker.run(name, nested=True, tags={"phase": "classifier", "mode": mode, "kind": kind or "none"}):
            if is_lgbm:
                require_lightgbm()
            arch = ae["arch"] if ae else None
            state = ae["state"] if (ae and mode not in ("scratch", "lgbm_raw")) else None
            sub, vsub = _subsample(data.tr, ccfg["search"]["train_subsample"], seed), _subsample(data.va, 50000, seed)
            ae_model = build_ae(arch, state).eval() if (is_lgbm and mode != "lgbm_raw") else None
            if is_lgbm:
                keys = LGBM_KEYS
                defaults = {k: ccfg["lgbm_defaults"][k] for k in keys}
                Fsub, Fvsub = build_features(ae_model, sub, mode), build_features(ae_model, vsub, mode)
                ysub, yvsub = sub.y.numpy().astype(int), vsub.y.numpy()

                def objective(trial):
                    m, _ = train_lgbm(lgbm_space(trial), Fsub, ysub, seed)
                    return _auc(yvsub, m.predict_proba(Fvsub)[:, 1])
            else:
                keys = ["head_hidden", "head_dropout", "lr", "weight_decay", "batch_size"] + (["enc_lr_mult"] if mode == "finetune" else [])
                defaults = {k: ccfg["defaults"][k] for k in keys}

                def objective(trial):
                    p = {**ccfg["defaults"], **head_space(trial, mode)}
                    clf, _ = train_classifier(arch, state, mode, p, sub, vsub, cfg, seed, device, trial=trial)
                    return _auc(vsub.y.numpy(), predict_proba(clf, vsub, device))

            best, info, trials = run_search(objective, defaults, ccfg["search"]["n_trials"], ccfg["search"]["timeout_seconds"], seed, "maximize")
            t0 = time.perf_counter()
            sub_tr = _subsample(data.tr, 50000, seed)
            if is_lgbm:
                params = {**ccfg["lgbm_defaults"], **best}
                Ftr, Fva, Fsub_tr = build_features(ae_model, data.tr, mode), build_features(ae_model, data.va, mode), build_features(ae_model, sub_tr, mode)
                model, hist = train_lgbm(params, Ftr, data.tr.y.numpy().astype(int), seed)
                res["fit_seconds"] = time.perf_counter() - t0
                p_val, p_tr = model.predict_proba(Fva)[:, 1], model.predict_proba(Fsub_tr)[:, 1]
                res["trainable_params"] = leaf_count_proxy(model, params)
                res["trainable_params_note"] = "LightGBM: trees x max leaves (upper-bound proxy, not comparable one-to-one with neural weights)"
                n_lat = min(5000, len(data.va))
                t1 = time.perf_counter()
                model.predict_proba(build_features(ae_model, data.va.take(range(n_lat)), mode))
                clf = None
            else:
                params = {**ccfg["defaults"], **best}
                if mode != "finetune":
                    params["enc_lr_mult"] = 1.0
                clf, hist = train_classifier(arch, state, mode, params, data.tr, data.va, cfg, seed, device)
                res["fit_seconds"] = time.perf_counter() - t0
                p_val, p_tr = predict_proba(clf, data.va), predict_proba(clf, sub_tr)
                res["trainable_params"] = count_trainable(clf, mode)
                n_lat = min(5000, len(data.va))
                t1 = time.perf_counter()
                predict_proba(clf, data.va.take(range(n_lat)))
            res["latency_ms_per_1000_rows"] = (time.perf_counter() - t1) / n_lat * 1e6
            thr = choose_threshold(data.y_val, p_val, cfg["metric"]["threshold_rule"], cfg["metric"]["fixed_threshold"])
            res["val_metrics"] = compute_metrics(data.y_val, p_val, thr)
            res["train_metrics"] = compute_metrics(sub_tr.y.numpy(), p_tr, thr)
            m = cfg["metric"]["primary"]
            key, sign = ("log_loss", -1.0) if m == "neg_log_loss" else (m, 1.0)
            res["val_primary"], res["train_primary"] = sign * res["val_metrics"][key], sign * res["train_metrics"][key]
            res["overfit_gap"] = res["train_primary"] - res["val_primary"]
            res["head_params"] = {k: params[k] for k in keys}
            res["search"] = info
            res["epochs_trained"] = len(hist)
            res["slices"] = slice_metrics(data.X_val, data.y_val, p_val, thr, cfg["diagnostics"]["slice_columns"])
            res["pretrained_autoencoder"] = (ae["params"] if ae else None)
            res["pretrained_reconstruction"] = (ae["reconstruction"] if ae else None)
            res["diagnosis"] = _diagnose(res, hist, cfg, cap=None if is_lgbm else ccfg["max_epochs"])
            trials.to_csv(d / "search_trials.csv", index=False)
            hist.to_csv(d / "training_curve.csv", index=False)
            np.savez_compressed(d / "val_predictions.npz", p=p_val, y=data.y_val)
            res["artifacts"] = {
                "roc_pr_plot": plot_roc_pr(data.y_val, p_val, d / "plots" / "roc_pr.png", f"({name})"),
                "calibration_plot": plot_calibration(data.y_val, p_val, d / "plots" / "calibration.png", f"({name})"),
                "confusion_plot": plot_confusion(data.y_val, (p_val >= thr).astype(int), d / "plots" / "confusion.png", f"(thr={thr:.2f})"),
                "training_curve_plot": plot_curve(hist, "iteration", ["train_loss", "val_loss"], d / "plots" / "training_curve.png", f"{name}: log loss", "log loss"),
                "search_history_plot": plot_optuna_history(trials, d / "plots" / "search_history.png", name),
                "val_predictions": str(d / "val_predictions.npz")}
            bundle = ModelBundle(kind or "none", mode, data.encoder, data.schema, arch, state, params,
                                 None if is_lgbm else {k: v.clone() for k, v in clf.state_dict().items()}, thr,
                                 {"candidate": name, "val_metrics": res["val_metrics"], "primary_metric": m, **data.run_meta},
                                 compute_reference_stats(data.X_train, data.schema), head_type=("torch" if not is_lgbm else mode), lgbm=(model if is_lgbm else None))
            bundle.save(d / "bundle")
            res["artifacts"]["bundle_dir"] = str(d / "bundle")
            tracker.log_params({f"best.{k}": v for k, v in res["head_params"].items()})
            tracker.log_metrics({**{f"val.{k}": v for k, v in res["val_metrics"].items()}, "overfit_gap": res["overfit_gap"]})
            for k in ("roc_pr_plot", "calibration_plot", "training_curve_plot"):
                tracker.log_artifact(res["artifacts"][k], "plots")
            res["_clf_params"] = params
    except Exception as exc:
        res.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        (d / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    dump_json({k: v for k, v in res.items() if not k.startswith("_")}, d / "result.json")
    return res


def _diagnose(res: dict, hist: pd.DataFrame, cfg: dict, cap: int | None = None) -> dict:
    notes, verdict = [], "healthy"
    gap, mx = res["overfit_gap"], cfg["selection"]["max_overfit_gap"]
    if gap > mx:
        verdict = "overfitting"
        notes.append(f"train-validation gap {gap:.4f} exceeds {mx}.")
    best_it = int(hist["val_loss"].idxmin())
    if best_it < len(hist) - 1 - 3:
        notes.append(f"validation loss was lowest at epoch {best_it}; early stopping restored those weights.")
    if cap is not None and len(hist) >= cap:
        verdict = "possibly_undertrained" if verdict == "healthy" else verdict
        notes.append("hit the epoch cap while still improving: raise classifier.max_epochs.")
    return {"verdict": verdict, "gap": gap, "notes": notes}


def baseline_candidate(data: ExperimentData, out_dir: Path) -> dict:
    cfg = data.cfg
    d = out_dir / "candidates" / "majority_baseline"
    d.mkdir(parents=True, exist_ok=True)
    prior = float(np.mean(data.y_train))
    p_val, p_tr = np.full(len(data.y_val), prior), np.full(len(data.y_train), prior)
    thr = choose_threshold(data.y_val, p_val, cfg["metric"]["threshold_rule"], cfg["metric"]["fixed_threshold"])
    vm, tm = compute_metrics(data.y_val, p_val, thr), compute_metrics(data.y_train, p_tr, thr)
    m = cfg["metric"]["primary"]
    key, sign = ("log_loss", -1.0) if m == "neg_log_loss" else (m, 1.0)
    np.savez_compressed(d / "val_predictions.npz", p=p_val, y=data.y_val)
    res = {"name": "majority_baseline", "kind": None, "mode": "baseline", "role": "baseline", "family": "baseline", "status": "ok", "message": "",
           "display_name": "Majority-class baseline", "primary_metric": m, "loss_function": "none (constant prior)",
           "val_metrics": vm, "train_metrics": tm, "val_primary": sign * vm[key], "train_primary": sign * tm[key], "overfit_gap": sign * (tm[key] - vm[key]),
           "trainable_params": 0, "latency_ms_per_1000_rows": 0.0, "fit_seconds": 0.0, "slices": {}, "search": {}, "head_params": {},
           "diagnosis": {"verdict": "reference", "gap": 0.0, "notes": []}, "artifacts": {"val_predictions": str(d / "val_predictions.npz")}}
    dump_json(res, d / "result.json")
    return res


# ------------------------------------------------------------------------------------------------ label efficiency
def label_efficiency(data: ExperimentData, ae_results: dict, cands: list[dict], device: str, out_dir: Path) -> dict | None:
    lcfg, cfg = data.cfg["label_efficiency"], data.cfg
    if not lcfg["enabled"]:
        return None
    ok = [c for c in cands if c["status"] == "ok"]
    ae_best = max((c for c in ok if c["role"] == "candidate" and c["mode"] in ("frozen", "finetune")), key=lambda c: c["val_primary"], default=None)
    scratch = next((c for c in ok if c["mode"] == "scratch"), None)
    if ae_best is None or scratch is None:
        return None
    rows = []
    for label, cand in (("pretrained: " + ae_best["name"], ae_best), ("scratch control", scratch)):
        ae = ae_results.get(cand["kind"] or ae_best["kind"])
        for frac in lcfg["fractions"]:
            for s in lcfg["seeds"]:
                g = torch.Generator().manual_seed(1000 + s)
                n = max(50, int(len(data.tr) * frac))
                idx = torch.randperm(len(data.tr), generator=g)[:n]
                clf, _ = train_classifier(ae["arch"], ae["state"] if cand["mode"] != "scratch" else None, cand["mode"], cand["_clf_params"],
                                          data.tr.take(idx), data.va, cfg, cfg["project"]["seed"] + s, device, max_epochs=cfg["classifier"]["max_epochs"])
                rows.append({"setup": label, "label_fraction": frac, "labels": n, "seed": s, "val_auc": _auc(data.y_val, predict_proba(clf, data.va))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "label_efficiency.csv", index=False)
    g = df.groupby(["setup", "label_fraction"])["val_auc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for setup, sub in g.groupby("setup"):
        ax.errorbar(sub["label_fraction"], sub["mean"], yerr=sub["std"].fillna(0), marker="o", label=setup, capsize=3)
    ax.set(xscale="log", xlabel="fraction of labelled training rows", ylabel="validation ROC-AUC", title="Label efficiency: does pretraining help?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (out_dir / "reports").mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "reports" / "label_efficiency.png", dpi=110); plt.close(fig)
    summary = {"pretrained_candidate": ae_best["name"], "table": g.to_dict("records"),
               "note": "Pretraining always sees ALL training rows' features (no labels); only the supervised head/finetune stage is label-limited."}
    dump_json(summary, out_dir / "label_efficiency.json")
    return summary


# ------------------------------------------------------------------------------------------------ orchestration
def run_all(data: ExperimentData, tracker: Tracker, out_dir: Path, log=print) -> tuple[list[dict], dict, dict | None]:
    from .training import pick_device

    cfg = data.cfg
    device = pick_device(cfg["project"]["device"])
    log(f"device: {device}")
    cands = [baseline_candidate(data, out_dir)]
    ae_results: dict = {}
    for kind in cfg["autoencoder"]["kinds"]:
        log(f"--> autoencoder '{kind}'")
        ae_results[kind] = pretrain(kind, data, tracker, out_dir, device)
        r = ae_results[kind]
        log(f"    {r['status']}" + (f": val recon {r['best_val_recon']:.4f}, R2 {r['reconstruction']['mean_continuous_r2']:.3f}, "
                                    f"cat acc {r['reconstruction']['mean_categorical_accuracy']:.3f}" if r["status"] == "ok" else f": {r['message']}"))
    for kind, ae in ae_results.items():
        if ae["status"] != "ok":
            continue
        for mode in cfg["classifier"]["modes"]:
            log(f"--> classifier {kind}_{mode}")
            c = run_candidate(f"{kind}_{mode}", kind, mode, ae, data, tracker, out_dir, device, "candidate")
            cands.append(c)
            log(f"    {c['status']}" + (f": val {c['val_primary']:.4f} (train {c['train_primary']:.4f})" if c["status"] == "ok" else f": {c['message']}"))
    if cfg["classifier"]["include_scratch_control"]:
        ref = next((k for k in ("ae", *ae_results) if k in ae_results and ae_results[k]["status"] == "ok"), None)
        if ref:
            log("--> control: same architecture, no pretraining")
            c = run_candidate("scratch_mlp", ref, "scratch", ae_results[ref], data, tracker, out_dir, device, "control")
            cands.append(c)
            log(f"    {c['status']}" + (f": val {c['val_primary']:.4f}" if c["status"] == "ok" else f": {c['message']}"))
    if cfg["classifier"].get("include_lgbm_raw_control", False):
        log("--> control: LightGBM on raw features, no encoder")
        c = run_candidate("lgbm_raw_control", None, "lgbm_raw", None, data, tracker, out_dir, device, "control")
        cands.append(c)
        log(f"    {c['status']}" + (f": val {c['val_primary']:.4f}" if c["status"] == "ok" else f": {c['message']}"))
    log("--> label-efficiency study")
    le = label_efficiency(data, {k: v for k, v in ae_results.items() if v["status"] == "ok"}, cands, device, out_dir)
    return cands, ae_results, le
