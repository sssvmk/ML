"""
Shared plumbing for the custom PyTorch modules (TCN #23 now; DeepState #26 and DeepVAR #27 later): data windowing,
per-window scaling, and the trainer loop (C-1 allows shared plumbing, never a shared model class).

  * Windows are (lookback, features) with the target in column 0 and historical exogenous / calendar covariates after it.
  * Scaling is PER WINDOW, fitted on that window's own inputs only, and the target is scaled with the same statistics
    (no leakage, C-5). Forecasts are mapped back to original units.
  * Training loss and validation loss are both computed on the SCALED target, so the train-vs-validation gap recorded
    in `_fitted_model["early_stopping"]` is a like-for-like overfitting signal (G-10).
  * Chronological validation tail, early stopping with patience (in epochs) and best-weights restore (C-2); seeded
    shuffling and initialisation, gradient clipping, CPU device (C-5).
  * save() persists state_dict + architecture config + hyperparameters; load() rebuilds the network without any
    training data (C-3). Any horizon is served: direct h-step forecasts rolled forward in chunks when longer (C-3).
"""

from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import pandas as pd

from .base import AlgorithmModule
from .utils import future_dates
from ._frames import DailyFrameMixin, CALENDAR_COLS, calendar_features


class TorchWindowModule(DailyFrameMixin, AlgorithmModule):
    """Subclass: set `name`, implement `lookback()`, `build_network(n_features)` (nn.Module: (B, L, F) -> (B, h)),
    and the eligibility / required-observation rules."""

    algorithm_version = "torch-1"
    default_max_epochs = 30
    needs_horizon = True  # built for a fixed h: config.build_candidates injects the configured backtest horizon
    default_use_calendar = False
    context_family_floor = 0

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        hp = self.hyperparameters
        self.horizon = int(hp.get("horizon", 4))
        self._use_calendar = bool(hp.get("use_calendar", self.default_use_calendar))
        self._net = None
        self._exog_cols: list[str] = []
        self._n_features = 0

    # ---- subclass hooks --------------------------------------------------------------------------------
    def lookback(self) -> int:
        raise NotImplementedError

    def build_network(self, n_features: int):
        raise NotImplementedError

    # ---- data -----------------------------------------------------------------------------------------
    def _matrix(self, segment_df: pd.DataFrame, exog_cols: list[str] | None):
        frame = self._frame(segment_df, exog_cols=exog_cols)
        cols = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
        parts = [frame[["y"]].to_numpy(dtype=np.float32), frame[cols].to_numpy(dtype=np.float32) if cols else np.zeros((len(frame), 0), np.float32)]
        if self._use_calendar:
            parts.append(calendar_features(frame["ds"]).to_numpy(dtype=np.float32))
        return frame, cols, np.concatenate(parts, axis=1)

    @staticmethod
    def _scale_windows(X: np.ndarray, Y: np.ndarray | None):
        """Per-window standardisation from the window's own inputs; target uses the y-column statistics."""
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        Xs = (X - mu) / sd
        Ys = None if Y is None else (Y - mu[:, :, 0]) / sd[:, :, 0]
        return Xs.astype(np.float32), (None if Ys is None else Ys.astype(np.float32)), mu[:, :, 0], sd[:, :, 0]

    def _windows(self, M: np.ndarray):
        L, h = self.lookback(), self.horizon
        n = M.shape[0] - L - h + 1
        if n <= 0:
            raise ValueError(f"series of {M.shape[0]} days too short for lookback {L} + horizon {h}")
        idx = np.arange(L)[None, :] + np.arange(n)[:, None]
        X = M[idx]                                             # (n, L, F)
        Y = np.stack([M[i + L: i + L + h, 0] for i in range(n)])
        return X, Y

    # ---- optional hooks ----------------------------------------------------------------------------------
    def _fit_enabled(self) -> bool:
        """False for a zero-shot module: train() then only builds/loads the network (no fitting)."""
        return True

    def _extra_state(self) -> dict:
        """Extra JSON-plain state persisted by save() (e.g. the architecture config of a library model)."""
        return {}

    def _load_extra_state(self, extra: dict) -> None:
        """Restore what _extra_state() saved, before the network is rebuilt in load()."""

    # ---- train ----------------------------------------------------------------------------------------
    def train(self, segment_df: pd.DataFrame) -> None:
        import torch
        hp = self.hyperparameters
        frame, cols, M = self._matrix(segment_df, exog_cols=None)
        self._exog_cols = cols
        self._n_features = M.shape[1]
        X, Y = self._windows(M)
        Xs, Ys, _, _ = self._scale_windows(X, Y)
        seed = int(hp.get("random_seed", 1))
        torch.manual_seed(seed)
        net = self.build_network(self._n_features)
        if not self._fit_enabled():
            net.eval()
            self._net = net
            self._fitted_model = {"lookback": self.lookback(), "horizon": self.horizon, "hyperparameters": self.hyperparameters,
                                  "n_parameters": self.n_parameters(), "exogenous_columns": cols,
                                  "early_stopping": {"used": False, "fine_tuned": False, "reason": "zero-shot module: no fitting"},
                                  **self._extra_fitted()}
            self._store_resid(Xs, X, Y)
            return
        n = len(Xs)
        patience = int(hp.get("early_stop_patience", 5))
        n_val = int(max(10, float(hp.get("validation_fraction", 0.15)) * n)) if patience > 0 else 0
        if n_val and n - n_val < 50:
            n_val = 0
        Xt, Yt = torch.tensor(Xs[: n - n_val]), torch.tensor(Ys[: n - n_val])
        Xv, Yv = (torch.tensor(Xs[n - n_val:]), torch.tensor(Ys[n - n_val:])) if n_val else (None, None)
        opt = torch.optim.Adam(net.parameters(), lr=float(hp.get("learning_rate", 1e-3)))
        loss_fn = torch.nn.L1Loss()
        clip = float(hp.get("gradient_clip_val", 1.0))
        bs = int(hp.get("batch_size", 32))
        gen = torch.Generator().manual_seed(seed)
        best = {"loss": float("inf"), "epoch": -1, "state": None}
        bad = 0
        epochs_run = 0
        train_hist, val_hist = [], []
        for epoch in range(int(hp.get("max_epochs", self.default_max_epochs))):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(perm), bs):
                b = perm[i: i + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]), Yt[b])
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
            epochs_run = epoch + 1
            net.eval()
            with torch.no_grad():
                tl = float(loss_fn(net(Xt), Yt))
                train_hist.append(tl)
                if n_val:
                    vl = float(loss_fn(net(Xv), Yv))
                    val_hist.append(vl)
                    if vl < best["loss"]:
                        best = {"loss": vl, "epoch": epoch, "state": copy.deepcopy(net.state_dict())}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= patience:
                            break
        restored = False
        if n_val and best["state"] is not None:
            net.load_state_dict(best["state"])          # best-weights restore (C-2)
            restored = True
        net.eval()
        self._net = net
        with torch.no_grad():
            tl_final = float(loss_fn(net(Xt), Yt))
        es = {"used": bool(n_val), "validation_rows": int(n_val), "epochs_run": epochs_run,
              "max_epochs": int(hp.get("max_epochs", self.default_max_epochs)), "restored_best_checkpoint": restored,
              "final_train_loss": tl_final, "loss_units": "MAE on per-window-scaled target"}
        if n_val:
            es.update({"best_epoch": best["epoch"] + 1, "best_validation_loss": best["loss"],
                       "train_validation_loss_gap": best["loss"] - train_hist[best["epoch"]]})
        self._fitted_model = {"lookback": self.lookback(), "horizon": self.horizon, "hyperparameters": self.hyperparameters,
                              "n_parameters": self.n_parameters(), "early_stopping": es, "exogenous_columns": cols,
                              **self._extra_fitted()}
        self._store_resid(Xs, X, Y)

    def _extra_fitted(self) -> dict:
        return {}

    def _store_resid(self, Xs, X, Y) -> None:
        import torch
        try:
            with torch.no_grad():
                p = self._net(torch.tensor(Xs)).numpy()
            _, _, mu, sd = self._scale_windows(X, None)
            self._resid = (Y[:, 0] - (p[:, 0] * sd[:, 0] + mu[:, 0])).astype(float)
        except Exception:
            self._resid = None

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters())) if self._net is not None else 0

    def network(self):
        return self._net

    # ---- infer ----------------------------------------------------------------------------------------
    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        import torch
        if self._net is None:
            raise RuntimeError("load() or train() must run before infer()")
        frame, _, M = self._matrix(segment_df, exog_cols=self._exog_cols)
        L = self.lookback()
        if len(M) < L:
            raise ValueError(f"history of {len(M)} days shorter than lookback {L}")
        dates = pd.DatetimeIndex(frame["ds"])
        preds: list[float] = []
        while len(preds) < horizon:
            window = M[-L:][None]
            Xs, _, mu, sd = self._scale_windows(window, None)
            with torch.no_grad():
                out = self._net(torch.tensor(Xs)).numpy()[0]
            yhat = out * sd[0, 0] + mu[0, 0]
            preds.extend(float(v) for v in yhat)
            if len(preds) >= horizon:
                break
            new_dates = future_dates(dates[-1], len(yhat), freq=self.frequency)
            rows = np.zeros((len(yhat), M.shape[1]), np.float32)
            rows[:, 0] = yhat
            n_ex = len(self._exog_cols)
            if n_ex:
                rows[:, 1: 1 + n_ex] = M[-1, 1: 1 + n_ex]        # historical exogenous carried forward when rolling beyond h
            if self._use_calendar:
                rows[:, 1 + n_ex:] = calendar_features(new_dates).to_numpy(dtype=np.float32)
            M = np.concatenate([M, rows])
            dates = dates.append(new_dates)
        return pd.DataFrame({"date": future_dates(frame["ds"].max(), horizon, freq=self.frequency), "forecast": preds[:horizon]})

    # ---- save / load ----------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "hyperparameters": _plain(self.hyperparameters),
                    "exog_cols": list(self._exog_cols), "n_features": int(self._n_features), "horizon": self.horizon,
                    "use_calendar": self._use_calendar, "class": type(self).__name__, "fitted": _plain(self._fitted_model),
                    "extra": _plain(self._extra_state())}, path)

    def load(self, path: Path) -> None:
        import torch
        st = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.hyperparameters = st["hyperparameters"]
        self.horizon, self._use_calendar = int(st["horizon"]), bool(st["use_calendar"])
        self._exog_cols, self._n_features = list(st["exog_cols"]), int(st["n_features"])
        self._load_extra_state(st.get("extra") or {})
        self._reconfigure()
        self._net = self.build_network(self._n_features)
        self._net.load_state_dict(st["state_dict"])
        self._net.eval()
        self._fitted_model = st.get("fitted")

    def _reconfigure(self) -> None:
        """Re-derive attributes that depend on hyperparameters after load() replaced them."""


def _plain(obj):
    """JSON-plain copy (tuples -> lists, numpy scalars -> python) so weights_only loading accepts it."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
