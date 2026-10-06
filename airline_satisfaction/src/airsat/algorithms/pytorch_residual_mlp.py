"""Residual MLP for tabular data (PyTorch): the deep-learning candidate.

Architecture   Linear(n_in -> h) -> [x + Dropout(Linear(ReLU(BatchNorm(x))))] x n_blocks -> BN -> ReLU -> Linear(h -> 1)
Loss           binary cross-entropy on logits (BCEWithLogitsLoss, numerically stable)
Optimiser      AdamW (decoupled weight decay), ReduceLROnPlateau, gradient clipping at 1.0
Regularisation dropout + weight decay + early stopping on log loss of a 10% hold-out taken from TRAIN
Search         Optuna TPE with a median pruner on per-epoch validation loss (the only module that benefits from pruning)
Honest prior   on small, mostly-ordinal tabular data, gradient boosting usually matches or beats a tuned MLP; it is
               included because the task asks for a deep-learning contender and the experiment should decide.

Sanity checks that run before training (debugging checklist, Goodfellow et al. Ch. 11):
  * initial loss ~ ln 2 for a balanced binary problem (else the output layer / loss is mis-wired)
  * the net must be able to over-fit a tiny batch to near-zero loss (else there is a bug in model/loss/data pipeline)
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from .. import assumptions as A
from .base import AlgorithmModule

try:
    import torch
    from torch import nn
except Exception:  # torch is optional
    torch = None
    nn = None


if torch is not None:

    class _ResBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.norm = nn.BatchNorm1d(width)
            self.fc = nn.Linear(width, width)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.fc(torch.relu(self.norm(x))))

    class _Net(nn.Module):
        def __init__(self, n_in: int, hidden: int, n_blocks: int, dropout: float):
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.blocks = nn.Sequential(*[_ResBlock(hidden, dropout) for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x):
            return self.head(self.blocks(self.inp(x))).squeeze(-1)


def n_parameters(n_in: int, hidden: int, n_blocks: int) -> int:
    inp = n_in * hidden + hidden
    block = hidden * hidden + hidden + 2 * hidden          # linear + batchnorm affine
    head = 2 * hidden + hidden + 1
    return inp + n_blocks * block + head


class TorchResidualMLPClassifier:
    """sklearn-style wrapper (fit / predict_proba) so the shared protocol and the ModelBundle treat it like any model."""

    def __init__(self, hidden=128, n_blocks=2, dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=1024,
                 max_epochs=40, patience=6, seed=42):
        self.hidden, self.n_blocks, self.dropout = hidden, n_blocks, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.net_ = None
        self.state_ = None
        self.n_in_ = None
        self.history_: list[dict] = []
        self.classes_ = np.array([0, 1])

    # -- helpers -------------------------------------------------------------------------------------------
    @staticmethod
    def _device() -> str:
        forced = os.environ.get("AIRSAT_DEVICE")
        if forced:
            return forced
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _new_net(self, n_in: int, dropout: float | None = None):
        return _Net(n_in, self.hidden, self.n_blocks, self.dropout if dropout is None else dropout)

    @staticmethod
    def _as_tensor(a, dtype=None):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32 if dtype is None else dtype))

    def _loss_on(self, net, X, y, device, batch=8192) -> float:
        net.eval()
        total = 0.0
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb, yb = X[i:i + batch].to(device), y[i:i + batch].to(device)
                total += loss_fn(net(xb), yb).item()
        return total / len(X)

    # -- training --------------------------------------------------------------------------------------------
    def fit(self, X, y, X_es=None, y_es=None, trial=None):
        torch.manual_seed(self.seed)
        device = self._device()
        Xt, yt = self._as_tensor(X), self._as_tensor(y)
        Xe = self._as_tensor(X_es) if X_es is not None else None
        ye = self._as_tensor(y_es) if y_es is not None else None
        self.n_in_ = Xt.shape[1]
        net = self._new_net(self.n_in_).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)
        sub = torch.randperm(len(Xt), generator=gen)[: min(20_000, len(Xt))]   # fixed subset for eval-mode train loss
        best, best_state, bad = math.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:                       # BatchNorm needs more than one row
                    continue
                xb, yb = Xt[idx].to(device), yt[idx].to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            train_loss = self._loss_on(net, Xt[sub], yt[sub], device)
            val_loss = self._loss_on(net, Xe, ye, device) if Xe is not None else train_loss
            sched.step(val_loss)
            self.history_.append({"iteration": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                  "lr": opt.param_groups[0]["lr"]})
            if trial is not None:
                import optuna

                trial.report(-val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.state_ = best_state
        self.net_ = net.cpu().eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xt = self._as_tensor(X)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 16384):
                out.append(torch.sigmoid(self.net_(Xt[i:i + 16384])).numpy())
        p = np.concatenate(out).astype(float)
        return np.column_stack([1 - p, p])

    # -- sanity checks used by check_assumptions() -----------------------------------------------------------
    def initial_loss(self, X, y) -> float:
        torch.manual_seed(self.seed)
        net = self._new_net(np.asarray(X).shape[1])
        return self._loss_on(net, self._as_tensor(X), self._as_tensor(y), "cpu")

    def overfit_tiny_batch(self, X, y, steps: int = 400) -> float:
        torch.manual_seed(self.seed)
        Xt, yt = self._as_tensor(X), self._as_tensor(y)
        net = self._new_net(Xt.shape[1], dropout=0.0)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(net(Xt), yt)
            loss.backward()
            opt.step()
        return float(loss.item())

    # -- pickling: persist weights, not the live module graph ----------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["net_"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.state_ is not None and self.n_in_ is not None:
            net = self._new_net(self.n_in_)
            net.load_state_dict(self.state_)
            self.net_ = net.eval()


class Algorithm(AlgorithmModule):
    name = "pytorch_residual_mlp"
    display_name = "Residual MLP (PyTorch)"
    family = "deep"
    description = ("Fully connected network with residual blocks, batch-norm, dropout and AdamW. Learns non-linear feature "
                   "interactions end-to-end; needs scaled inputs and more tuning than trees.")
    complexity_rank = 5
    loss_function = "Binary cross-entropy on logits (BCEWithLogitsLoss) + decoupled weight decay (AdamW)."
    optimisation_notes = ("Mini-batch AdamW, ReduceLROnPlateau, gradient clipping 1.0, early stopping on a 10% TRAIN hold-out; "
                          "TPE + median pruner on per-epoch validation loss.")
    uses_early_stopping = True
    iterative = True
    nominal_encoding = "onehot"
    scale_inputs = True

    def is_available(self):
        return (torch is not None, "PyTorch is not installed (pip install torch); module skipped")

    def default_params(self) -> dict:
        return {"hidden": 128, "n_blocks": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024}

    def hyperparameter_docs(self) -> dict:
        return {"hidden": "Width of every layer.", "n_blocks": "Number of residual blocks (depth).",
                "dropout": "Dropout probability inside blocks (regularisation).",
                "lr": "AdamW initial learning rate (log-uniform) - usually the most important knob.",
                "weight_decay": "Decoupled L2 shrinkage.", "batch_size": "Mini-batch size."}

    def search_space(self, trial) -> dict:
        return {"hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
                "n_blocks": trial.suggest_int("n_blocks", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048])}

    def build_estimator(self, params, seed):
        return TorchResidualMLPClassifier(seed=seed, **params)

    def fit_estimator(self, est, X, y, X_es=None, y_es=None, trial=None):
        return est.fit(X, y, X_es, y_es, trial=trial)

    def training_curve(self, est):
        return pd.DataFrame(est.history_)

    def check_assumptions(self, ctx):
        d = self.default_params()
        n_params = n_parameters(ctx.Xt.shape[1], d["hidden"], d["n_blocks"])
        idx = ctx.sample_idx(2000)
        est = self.build_estimator(d, ctx.seed)
        init = est.initial_loss(ctx.Xt[idx], ctx.y[idx])
        init_check = A._r("initial_loss_near_ln2", 0.5 <= init <= 1.0, init,
                          f"loss at initialisation {init:.3f} (expected ~{math.log(2):.3f} for a balanced binary problem)",
                          "A wildly different initial loss reveals a mis-wired output layer or loss.", warn_only=False, blocking=True)
        tiny = ctx.sample_idx(64)
        final = est.overfit_tiny_batch(ctx.Xt[tiny], ctx.y[tiny])
        tiny_check = A._r("can_overfit_tiny_batch", final < 0.05, final,
                          f"loss after 400 steps on 64 rows = {final:.4f} (must be < 0.05)",
                          "If the net cannot memorise 64 rows there is a bug in the model, loss or data pipeline.",
                          warn_only=False, blocking=True)
        return [A.check_finite_inputs(ctx), A.check_both_classes_present(ctx), A.check_class_balance(ctx),
                A.check_sample_size(ctx), A.check_duplicates(ctx), A.check_single_feature_leakage(ctx),
                A.check_train_validation_shift(ctx), A.check_scaled_inputs(ctx), A.check_params_vs_samples(ctx, n_params),
                init_check, tiny_check]
