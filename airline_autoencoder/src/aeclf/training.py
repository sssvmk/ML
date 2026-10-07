"""Training loops (steps 4-5) plus reconstruction evaluation and sanity checks.

Stage 1  train_autoencoder(): unsupervised, the target is never touched. Early stopping on VALIDATION reconstruction loss.
Stage 2  train_classifier(): frozen / finetune / scratch. Early stopping on VALIDATION log loss.
Validation is used for early stopping and hyper-parameter search in both stages (standard); the TEST set is never used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .models import (MixedAutoencoder, SatisfactionClassifier, corrupt, kl_divergence, reconstruction_loss)


# ------------------------------------------------------------------------------------------------------ data
@dataclass
class Tensors:
    xc: torch.Tensor
    xk: torch.Tensor
    y: torch.Tensor | None = None

    def __len__(self):
        return len(self.xk)

    def take(self, idx) -> "Tensors":
        idx = torch.as_tensor(idx)
        return Tensors(self.xc[idx], self.xk[idx], None if self.y is None else self.y[idx])


def to_tensors(encoder, df: pd.DataFrame, y=None) -> Tensors:
    xc, xk = encoder.transform(df)
    return Tensors(torch.from_numpy(xc), torch.from_numpy(xk), None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.float32))


def pick_device(name: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def build_ae(arch: dict, state: dict | None = None) -> MixedAutoencoder:
    ae = MixedAutoencoder(**arch)
    if state is not None:
        ae.load_state_dict(state)
    return ae


def _cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------------------------------------------ stage 1
@torch.no_grad()
def eval_recon(model: MixedAutoencoder, data: Tensors, w_cat: float, device: str, batch: int = 8192) -> dict:
    model.eval()
    n, tot, cm, ce, kl = 0, 0.0, 0.0, 0.0, 0.0
    for i in range(0, len(data), batch):
        xc, xk = data.xc[i:i + batch].to(device), data.xk[i:i + batch].to(device)
        out = model(xc, xk)
        t, c, k = reconstruction_loss(out, xc, xk, 1.0, w_cat)
        b = len(xk)
        n, tot, cm, ce = n + b, tot + t.item() * b, cm + c.item() * b, ce + k.item() * b
        if out["logvar"] is not None:
            kl += kl_divergence(out["mu"], out["logvar"]).item() * b
    return {"recon": tot / n, "cont_mse": cm / n, "cat_ce": ce / n, "kl": kl / n}


def train_autoencoder(kind: str, params: dict, arch_static: dict, tr: Tensors, va: Tensors, cfg: dict, seed: int,
                      device: str, trial=None, max_epochs: int | None = None):
    """Returns (model on cpu, history DataFrame, best validation reconstruction loss)."""
    torch.manual_seed(seed)
    acfg = cfg["autoencoder"]
    max_epochs = max_epochs or acfg["max_epochs"]
    arch = {**arch_static, "kind": kind, "hidden": params["hidden"], "depth": params["depth"], "latent": params["latent"],
            "dropout": params["dropout"]}
    model = MixedAutoencoder(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    unk = torch.tensor(arch_static["cards"], device=device) - 1
    gen = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(tr), generator=gen)[: min(20000, len(tr))]
    tr_sub = tr.take(sub)
    best, best_state, bad, hist = math.inf, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        beta = params["beta"] * min(1.0, (epoch + 1) / max(1, acfg["vae_kl_warmup_epochs"])) if kind == "vae" else 0.0
        perm = torch.randperm(len(tr), generator=gen)
        bs = min(params["batch_size"], max(64, len(tr) // 16))
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xc, xk = tr.xc[idx].to(device), tr.xk[idx].to(device)
            xin_c, xin_k = corrupt(xc, xk, params["noise_sigma"], params["mask_p"], unk) if kind == "dae" else (xc, xk)
            out = model(xin_c, xin_k)
            loss, _, _ = reconstruction_loss(out, xc, xk, 1.0, params["w_cat"])      # target = CLEAN row
            if kind == "vae":
                loss = loss + beta * kl_divergence(out["mu"], out["logvar"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = eval_recon(model, va, params["w_cat"], device)
        t = eval_recon(model, tr_sub, params["w_cat"], device)
        hist.append({"iteration": epoch, "train_loss": t["recon"], "val_loss": v["recon"], "val_cont_mse": v["cont_mse"],
                     "val_cat_ce": v["cat_ce"], "val_kl": v["kl"], "lr": opt.param_groups[0]["lr"]})
        sched.step(v["recon"])
        if trial is not None:
            import optuna
            trial.report(-v["recon"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if v["recon"] < best - 1e-5:
            best, bad, best_state = v["recon"], 0, _cpu_state(model)
        else:
            bad += 1
            if bad >= acfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), pd.DataFrame(hist), best


# ------------------------------------------------------------------------------------------------------ stage 2
@torch.no_grad()
def predict_proba(clf: SatisfactionClassifier, data: Tensors, device: str = "cpu", batch: int = 16384) -> np.ndarray:
    clf.eval()
    clf.to(device)
    out = [torch.sigmoid(clf(data.xc[i:i + batch].to(device), data.xk[i:i + batch].to(device))).cpu().numpy()
           for i in range(0, len(data), batch)]
    return np.concatenate(out).astype(float)


def train_classifier(ae_arch: dict, ae_state: dict | None, mode: str, params: dict, tr: Tensors, va: Tensors, cfg: dict,
                     seed: int, device: str, trial=None, max_epochs: int | None = None):
    """mode: frozen (encoder fixed) | finetune (encoder trained with lr*enc_lr_mult) | scratch (random init, all trained)."""
    torch.manual_seed(seed)
    ccfg = cfg["classifier"]
    max_epochs = max_epochs or ccfg["max_epochs"]
    ae = build_ae(ae_arch, ae_state if mode != "scratch" else None)
    clf = SatisfactionClassifier(ae, params["head_hidden"], params["head_dropout"]).to(device)
    head_params = list(clf.head.parameters())
    if mode == "frozen":
        for p in ae.parameters():
            p.requires_grad_(False)
        groups = [{"params": head_params, "lr": params["lr"]}]
    elif mode == "finetune":
        groups = [{"params": head_params, "lr": params["lr"]},
                  {"params": ae.encoder_parameters(), "lr": params["lr"] * params["enc_lr_mult"]}]
    else:
        groups = [{"params": head_params + ae.encoder_parameters(), "lr": params["lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    gen = torch.Generator().manual_seed(seed)
    yv = va.y.numpy()
    best, best_state, bad, hist = math.inf, None, 0, []
    for epoch in range(max_epochs):
        clf.train()
        if mode == "frozen":
            clf.ae.eval()                                  # no dropout / no updates in the fixed encoder
        perm = torch.randperm(len(tr), generator=gen)
        bs = min(params["batch_size"], max(64, len(tr) // 16))
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xc, xk, y = tr.xc[idx].to(device), tr.xk[idx].to(device), tr.y[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(clf(xc, xk), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
        pv = np.clip(predict_proba(clf, va, device), 1e-7, 1 - 1e-7)
        vloss = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
        sub = torch.arange(min(20000, len(tr)))
        ptr = np.clip(predict_proba(clf, tr.take(sub), device), 1e-7, 1 - 1e-7)
        ytr = tr.y[sub].numpy()
        hist.append({"iteration": epoch, "train_loss": float(-np.mean(ytr * np.log(ptr) + (1 - ytr) * np.log(1 - ptr))),
                     "val_loss": vloss, "val_auc": float(roc_auc_score(yv, pv)), "lr": opt.param_groups[0]["lr"]})
        sched.step(vloss)
        if trial is not None:
            import optuna
            trial.report(-vloss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if vloss < best - 1e-5:
            best, bad, best_state = vloss, 0, _cpu_state(clf)
        else:
            bad += 1
            if bad >= ccfg["patience"]:
                break
    clf.load_state_dict(best_state)
    return clf.cpu().eval(), pd.DataFrame(hist)


# ------------------------------------------------------------------------------------------------------ reconstruction
@torch.no_grad()
def reconstruct_full(model: MixedAutoencoder, enc, df: pd.DataFrame, device: str = "cpu"):
    """Decode rows back to the ORIGINAL columns/units. Returns (decoded frame, per-row error, predicted categorical codes)."""
    data = to_tensors(enc, df)
    model.eval().to(device)
    conts, codes, err = [], [], []
    for i in range(0, len(data), 8192):
        xc, xk = data.xc[i:i + 8192].to(device), data.xk[i:i + 8192].to(device)
        out = model(xc, xk)
        row = ((out["cont"] - xc) ** 2).mean(1) if out["cont"] is not None else torch.zeros(len(xk), device=device)
        ce = torch.stack([F.cross_entropy(lg, xk[:, j], reduction="none") for j, lg in enumerate(out["logits"])]).mean(0)
        err.append((row + ce).cpu().numpy())
        if out["cont"] is not None:
            conts.append(out["cont"].cpu().numpy())
        codes.append(torch.stack([lg.argmax(1) for lg in out["logits"]], 1).cpu().numpy())
    cont_df = enc.inverse_cont(np.concatenate(conts)) if conts else pd.DataFrame(index=range(len(df)))
    all_codes = np.concatenate(codes)
    cat_df = enc.decode_cat(all_codes)
    return pd.concat([cont_df, cat_df], axis=1), np.concatenate(err), all_codes


def reconstruct(model: MixedAutoencoder, enc, df: pd.DataFrame, device: str = "cpu") -> tuple[pd.DataFrame, np.ndarray]:
    dec, err, _ = reconstruct_full(model, enc, df, device)
    return dec, err


def reconstruction_report(model: MixedAutoencoder, enc, df: pd.DataFrame, device: str = "cpu", n_examples: int = 15) -> tuple[dict, pd.DataFrame]:
    """Per-column fidelity in ORIGINAL units vs the trivial reconstruction (train mean / train majority level)."""
    dec, row_err, codes = reconstruct_full(model, enc, df, device)
    xc, xk = enc.transform(df)
    rep: dict = {"continuous": {}, "categorical": {}, "n_rows": int(len(df))}
    for c in enc.cont_cols:
        truth = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(truth)
        pred = dec[c].to_numpy(float)
        mean = enc.cont_stats[c]["orig_mean"]
        sse, sst = np.sum((truth[ok] - pred[ok]) ** 2), np.sum((truth[ok] - mean) ** 2)
        rep["continuous"][c] = {"rmse": float(np.sqrt(np.mean((truth[ok] - pred[ok]) ** 2))), "mae": float(np.mean(np.abs(truth[ok] - pred[ok]))),
                                "baseline_rmse_mean_predictor": float(np.sqrt(np.mean((truth[ok] - mean) ** 2))),
                                "r2": float(1 - sse / sst) if sst > 0 else float("nan")}
    maj = enc.majority_codes()
    exact = np.ones(len(df), dtype=bool)
    for j, c in enumerate(enc.cat_cols):
        pred_codes = codes[:, j]
        hit = pred_codes == xk[:, j]
        exact &= hit
        entry = {"accuracy": float(hit.mean()), "baseline_accuracy_majority": float((xk[:, j] == maj[j]).mean())}
        if c in enc.ordinal_cats:
            entry["mae_levels"] = float(np.mean(np.abs(pred_codes - xk[:, j])))
        rep["categorical"][c] = entry
    r2s = [v["r2"] for v in rep["continuous"].values() if np.isfinite(v["r2"])]
    rep["summary"] = {"mean_continuous_r2": float(np.mean(r2s)) if r2s else float("nan"),
                      "mean_categorical_accuracy": float(np.mean([v["accuracy"] for v in rep["categorical"].values()])),
                      "mean_majority_baseline_accuracy": float(np.mean([v["baseline_accuracy_majority"] for v in rep["categorical"].values()])),
                      "exact_row_match_categorical": float(exact.mean()), "mean_row_error": float(np.mean(row_err))}
    show = list(df.columns)
    ex = pd.concat([df[show].head(n_examples).reset_index(drop=True).add_prefix("orig | "),
                    dec.reindex(columns=[c for c in show if c in dec.columns]).head(n_examples).add_prefix("recon | ")], axis=1)
    ex = ex[[c for pair in zip([f"orig | {s}" for s in show if s in dec.columns], [f"recon | {s}" for s in show if s in dec.columns]) for c in pair]]
    return rep, ex


@torch.no_grad()
def latent_stats(model: MixedAutoencoder, data: Tensors, device: str = "cpu", max_rows: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    d = data.take(torch.arange(min(max_rows, len(data))))
    mu, _ = model.encode(d.xc.to(device), d.xk.to(device))
    return mu.cpu().numpy(), (d.y.numpy() if d.y is not None else None)


# ------------------------------------------------------------------------------------------------------ sanity checks
def _check(name, ok, value, detail, why, mitigation="", blocking=False):
    return {"name": name, "passed": bool(ok), "severity": "pass" if ok else ("fail" if blocking else "warn"),
            "value": None if value is None else float(value), "detail": detail, "why_it_matters": why,
            "mitigation": mitigation, "blocking": bool(blocking and not ok)}


def pre_training_checks(kind: str, arch_static: dict, params: dict, tr: Tensors, cfg: dict, seed: int, device: str = "cpu") -> list[dict]:
    """Run BEFORE the search: finite inputs and 'can the network overfit 64 rows?' (a wiring/bug detector)."""
    bad = int((~torch.isfinite(tr.xc)).sum())
    checks = [_check("inputs_finite", bad == 0, bad, f"{bad} NaN/inf values in encoded continuous inputs",
                     "NaN/inf corrupt every gradient step.", "Median imputation inside the encoder.", blocking=True)]
    maxcode = [int(tr.xk[:, j].max()) for j in range(tr.xk.shape[1])]
    in_range = all(m < c for m, c in zip(maxcode, arch_static["cards"]))
    checks.append(_check("category_codes_in_range", in_range, float(in_range), "all categorical codes lie inside each embedding table",
                         "Out-of-range codes crash or silently corrupt embeddings.", blocking=True))
    torch.manual_seed(seed)
    idx = torch.randperm(len(tr))[:64]
    xc, xk = tr.xc[idx], tr.xk[idx]
    arch = {**arch_static, "kind": kind, "hidden": params["hidden"], "depth": params["depth"], "latent": params["latent"], "dropout": 0.0}
    m = MixedAutoencoder(**arch)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(400):
        out = m(xc, xk)
        loss, _, _ = reconstruction_loss(out, xc, xk, 1.0, params["w_cat"])
        if kind == "vae":
            loss = loss + 1e-4 * kl_divergence(out["mu"], out["logvar"])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ratio = loss.item() / first
    checks.append(_check("can_overfit_tiny_batch", ratio < 0.25, ratio, f"loss fell to {ratio:.1%} of its initial value on 64 rows in 400 steps (need < 25%)",
                         "A net that cannot memorise 64 rows has a bug in the model, loss or data pipeline.", blocking=True))
    return checks


def post_training_checks(kind: str, recon: dict, model: MixedAutoencoder, va: Tensors, cfg: dict, device: str = "cpu") -> list[dict]:
    s, rc = recon["summary"], cfg["reconstruction"]
    out = [_check("beats_mean_predictor_continuous", not np.isfinite(s["mean_continuous_r2"]) or s["mean_continuous_r2"] > rc["min_r2"],
                  s["mean_continuous_r2"], f"mean R2 of continuous columns {s['mean_continuous_r2']:.3f} (mean predictor = 0)",
                  "If the decoder cannot beat 'always predict the mean' the latent code carries no information about these columns.",
                  "Larger latent, more epochs, lower dropout."),
           _check("beats_majority_predictor_categorical", s["mean_categorical_accuracy"] > s["mean_majority_baseline_accuracy"],
                  s["mean_categorical_accuracy"], f"mean categorical accuracy {s['mean_categorical_accuracy']:.3f} vs majority baseline {s['mean_majority_baseline_accuracy']:.3f}",
                  "Reconstruction must beat the trivial per-column mode.", "Larger latent / capacity.")]
    mu, _ = latent_stats(model, va, device)
    active = int((mu.var(axis=0) > 0.01).sum())
    out.append(_check("no_latent_collapse", active >= rc["min_active_latent_dims"], active,
                      f"{active} of {mu.shape[1]} latent dimensions vary across passengers (variance > 0.01)",
                      "Collapsed latents (typical for VAEs with a large beta) carry almost no information for the classifier.",
                      "Lower beta, longer KL warm-up, smaller latent."))
    return out
