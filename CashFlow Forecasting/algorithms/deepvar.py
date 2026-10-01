"""
DeepVAR (PRD §3.2 #27; gap G-27; decisions D-4 / D-11 and the joint-inference decision) -- native PyTorch, written from the paper's
formulation (Salinas et al. 2019, low-rank Gaussian copula-free variant) with no code ported from any reference implementation.
CUSTOM model: it must pass the G-42 admission validation before it competes (`requires_admission`).

Model. The pooled series are the DIMENSIONS of one vector z_t in R^m (series scaled by their own scale; no FX conversion). On top of that
every window is divided, per dimension, by the mean absolute value of ITS OWN context part (the "item scale" of DeepAR/DeepVAR), so a
trending series is seen at a comparable level in training and at forecast time. A single
LSTM reads the previous vector z_{t-1} and the calendar features of t and emits the parameters of a multivariate Gaussian for z_t:
    mean mu_t (m),  diagonal variance d_t (m, softplus),  low-rank factor V_t (m x r)      Sigma_t = diag(d_t) + V_t V_t^T
trained on the exact negative log-likelihood (Woodbury / matrix-determinant-lemma form, implemented here and verified against scipy in
Layer 1). Forecasts are ANCESTRAL SAMPLES of the joint path (each step's sample is fed back), so cross-series dependence is carried; the
median over samples is each series' point forecast.

Joint inference (decision: pool-level). Forecasting one series needs the recent history of ALL the others, so the model is served through
`infer_pool(segments, horizon)` -- one call for the whole pool, used by the pooled backtest, the pooled holdout and
`Orchestrator.daily_infer_pool`. The single-segment `infer` refuses. Every series the model was trained on must be supplied and all of
them must end on the same date.

Limits, stated: no future-known covariates (calendar features only); the in-sample train-metric check and production monitoring work per
segment and therefore do not run for a joint model.

Eligibility (PRD): a genuinely multivariate target -- >= 2 series in the pool -- and pooled N >= 1,000.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .base import AlgorithmModule, EligibilityResult
from ._frames import DailyFrameMixin, calendar_features
from .utils import future_dates

LOG2PI = math.log(2.0 * math.pi)


def lowrank_gaussian_logpdf(x, mu, V, d):
    """
    log N(x; mu, diag(d) + V V^T), last dim m; V (..., m, r); d (..., m) > 0. Uses the matrix-determinant lemma and the Woodbury
    identity, so it costs O(m r^2) instead of O(m^3), and is differentiable.
    """
    e = x - mu
    m = e.shape[-1]
    r = V.shape[-1]
    Dinv_V = V / d[..., None]
    A = torch.eye(r, dtype=V.dtype, device=V.device) + V.transpose(-1, -2) @ Dinv_V          # I_r + V^T D^-1 V
    logdet = torch.log(d).sum(-1) + torch.logdet(A)
    Dinv_e = e / d
    t = (V.transpose(-1, -2) @ Dinv_e[..., None])                                             # (..., r, 1)
    quad = (e * Dinv_e).sum(-1) - (t.transpose(-1, -2) @ torch.linalg.solve(A, t)).squeeze(-1).squeeze(-1)
    return -0.5 * (m * LOG2PI + logdet + quad)


def sample_lowrank(mu, V, d, eps_r, eps_m):
    """x = mu + V eps_r + sqrt(d) * eps_m -- the sampling map of N(mu, diag(d) + V V^T); linear in the noise, so its covariance is exact."""
    return mu + (V @ eps_r[..., None]).squeeze(-1) + torch.sqrt(d) * eps_m


class DeepVARNet(nn.Module):
    def __init__(self, m: int, n_cov: int, hidden: int, layers: int, dropout: float, rank: int, dtype=torch.float32, diagonal: bool = False):
        super().__init__()
        self.m, self.rank, self.n_cov, self.diagonal = m, rank, n_cov, diagonal
        self.rnn = nn.LSTM(m + n_cov, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head_mu = nn.Linear(hidden, m)
        self.head_d = nn.Linear(hidden, m)
        self.head_v = nn.Linear(hidden, m * rank)
        with torch.no_grad():
            self.head_d.bias.fill_(-2.0)
            self.head_v.weight.mul_(0.1)
            self.head_v.bias.zero_()
        self.to(dtype)

    def params(self, h):
        mu = self.head_mu(h)
        d = Fn.softplus(self.head_d(h)) + 1e-4
        V = self.head_v(h).reshape(*h.shape[:-1], self.m, self.rank)
        if self.diagonal:
            V = torch.zeros_like(V)          # negative-control variant: no cross-series covariance (G-42 Layer 2); never the default
        return mu, d, V

    def nll(self, z, cov):
        """z (B, W, m) scaled; cov (B, W, n_cov). Teacher forcing: step t reads z_{t-1}; mean NLL per time step."""
        h, _ = self.rnn(torch.cat([z[:, :-1], cov[:, 1:]], dim=-1))
        mu, d, V = self.params(h)
        return -lowrank_gaussian_logpdf(z[:, 1:], mu, V, d).mean()


class DeepVARModule(DailyFrameMixin, AlgorithmModule):
    name = "deepvar"
    algorithm_version = "deepvar-native-1"
    has_eligibility_condition = True
    pooling_capable = True
    pooled_n_basis = "pooled_total"
    joint_inference = True
    requires_admission = True
    needs_horizon = True
    supports_hist_exog = False
    default_max_epochs = 30
    _use_calendar = True

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._net = None
        self._meta: dict = {}
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.horizon = int(hp.get("horizon", 4))
        self.context = int(hp.get("context_length", 56))

    # ---- contract / eligibility ------------------------------------------------------------------------------------------
    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return 1000     # PRD v10 §3.2 row #27: Pooled N >= 1,000

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        return EligibilityResult(False, "joint model: needs >= 2 series forecast together "
                                        "(train it through Orchestrator.full_train_pooled); one segment is not a multivariate target")

    def hyperparameter_search_space(self) -> dict:
        return {
            "hidden_size": {"type": "choice", "choices": [16, 32, 64]},
            "num_layers": {"type": "int", "low": 1, "high": 2},
            "rank": {"type": "int", "low": 1, "high": 4},
            "context_length": {"type": "int", "low": 28, "high": 112},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 2e-2, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }

    # ---- data ------------------------------------------------------------------------------------------------------------------
    def _aligned(self, segments: dict, ids: list[str], scales: dict):
        """(dates, Z scaled (T, m)) over the dates on which EVERY series is observed."""
        cols = {}
        for sid in ids:
            fr = self._frame(segments[sid], series_id=sid)
            cols[sid] = pd.Series(fr["y"].to_numpy(dtype=np.float64) / scales[sid], index=pd.DatetimeIndex(fr["ds"]))
        idx = None
        for s in cols.values():
            idx = s.index if idx is None else idx.intersection(s.index)
        idx = idx.sort_values()
        return idx, np.stack([cols[sid].reindex(idx).to_numpy() for sid in ids], axis=1)

    @staticmethod
    def _window_scale(context_part: np.ndarray) -> np.ndarray:
        """Mean |z| over the context axis (-2), kept as a broadcastable axis; 1 where a dimension is (numerically) all zero."""
        s = np.abs(context_part).mean(axis=-2, keepdims=True)
        return np.where(s < 1e-6, 1.0, s)

    def train(self, segment_df) -> None:
        raise RuntimeError("deepvar is a pooled joint model: train it with train_pooled(batch) / Orchestrator.full_train_pooled")

    def train_pooled(self, batch) -> None:
        hp = self.hyperparameters
        seed = int(hp.get("random_seed", 1))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dtype = torch.float64 if str(hp.get("dtype", "float32")) == "float64" else torch.float32
        ids = list(batch.segment_ids)
        if len(ids) < 2:
            raise ValueError("DeepVAR needs at least 2 series (dimensions)")
        dates, Z = self._aligned(batch.segments, ids, batch.scales)
        W = self.context + self.horizon
        if len(dates) < 2 * W + 20:
            raise ValueError(f"only {len(dates)} dates on which all {len(ids)} series are observed; need >= {2 * W + 20}")
        cov = calendar_features(dates).to_numpy(dtype=np.float64)
        n_cov = cov.shape[1]
        self._net = DeepVARNet(len(ids), n_cov, int(hp.get("hidden_size", 32)), int(hp.get("num_layers", 1)),
                               float(hp.get("dropout", 0.0)), int(hp.get("rank", 2)), dtype,
                               diagonal=str(hp.get("covariance", "low_rank")) == "diagonal")
        patience = int(hp.get("early_stop_patience", 5))
        n = len(dates)
        n_val = int(max(2 * W, float(hp.get("validation_fraction", 0.15)) * n)) if patience > 0 else 0
        cut = n - n_val
        opt = torch.optim.Adam(self._net.parameters(), lr=float(hp.get("learning_rate", 5e-3)))
        clip, bs, steps = float(hp.get("gradient_clip_val", 1.0)), int(hp.get("batch_size", 32)), int(hp.get("steps_per_epoch", 30))

        def make_batch(region: str, size: int, local_rng):
            lo, hi = (0, cut) if region == "train" else (cut, n)
            starts = local_rng.integers(lo, hi - W + 1, size=size)
            zb = np.stack([Z[s:s + W] for s in starts]); cb = np.stack([cov[s:s + W] for s in starts])
            zb = zb / self._window_scale(zb[:, :self.context, :])                 # per-window, per-dimension scale from the context part only
            return torch.tensor(zb, dtype=dtype), torch.tensor(cb, dtype=dtype)

        val_batch = make_batch("val", 4 * bs, np.random.default_rng(seed + 12345)) if patience > 0 else None
        best, bad, history = None, 0, []
        max_epochs = int(hp.get("max_epochs", self.default_max_epochs))
        for epoch in range(max_epochs):
            self._net.train()
            losses = []
            for _ in range(steps):
                zb, cb = make_batch("train", bs, rng)
                opt.zero_grad()
                loss = self._net.nll(zb, cb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), clip)
                opt.step()
                losses.append(float(loss))
            rec = {"epoch": epoch, "train_nll": float(np.mean(losses))}
            if val_batch is not None:
                self._net.eval()
                with torch.no_grad():
                    va = float(self._net.nll(*val_batch))
                rec["val_nll"] = va
                if best is None or va < best[0] - 1e-9:
                    best, bad = (va, copy.deepcopy({k: t.detach().clone() for k, t in self._net.state_dict().items()}), epoch), 0
                else:
                    bad += 1
            history.append(rec)
            if val_batch is not None and bad >= patience:
                break
        restored = False
        if best is not None:
            self._net.load_state_dict(best[1]); restored = True
        self._net.eval()
        self._meta = {"series": ids, "scales": dict(batch.scales), "n_cov": n_cov}
        self._fitted_model = {
            "hyperparameters": self.hyperparameters, "n_parameters": self.n_parameters(),
            "pooled": {"series": ids, "attributes": {sid: dict(a) for sid, a in batch.attributes.items()}, "n_series": len(ids),
                       "joint": True, "aligned_dates": int(len(dates))},
            "early_stopping": {"used": val_batch is not None, "epochs_run": len(history), "max_epochs": max_epochs,
                               "stopped_early": len(history) < max_epochs, "best_epoch": best[2] if best else None,
                               "restored_best_checkpoint": restored,
                               "loss_units": "negative log-likelihood per time step (vector of all series), scaled series; train and validation alike",
                               "history": history}}
        self._resid = None

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters())) if self._net is not None else 0

    def network(self):
        return self._net

    # ---- joint inference ---------------------------------------------------------------------------------------------------------
    def infer(self, segment_df, horizon: int):
        raise RuntimeError("deepvar forecasts the pooled series jointly and needs the history of every one of them: "
                           "use infer_pool(segments, horizon) / Orchestrator.daily_infer_pool")

    def _context(self, segments: dict):
        if self._net is None:
            raise RuntimeError("load() or train_pooled() must run before infer_pool()")
        ids = self._meta["series"]
        missing, extra = [i for i in ids if i not in segments], [i for i in segments if i not in ids]
        if missing or extra:
            raise ValueError(f"the joint model was trained on {ids}; missing {missing}, unexpected {extra}")
        dates, Z = self._aligned(segments, ids, self._meta["scales"])
        ends = {sid: self._frame(segments[sid], series_id=sid)["ds"].max() for sid in ids}
        if len(set(ends.values())) != 1:
            raise ValueError(f"a joint forecast needs every series to end on the same date; got {({k: str(v.date()) for k, v in ends.items()})}")
        if len(dates) < self.context or dates.max() != next(iter(ends.values())):
            raise ValueError(f"only {len(dates)} common dates ending {dates.max().date() if len(dates) else None}; need the last {self.context} for every series")
        return ids, dates[-self.context:], Z[-self.context:]

    def sample_paths(self, segments: dict, horizon: int, n: int | None = None, seed: int | None = None) -> np.ndarray:
        """Joint sample paths in ORIGINAL units, shape (n, horizon, m), series in `self._meta['series']` order."""
        hp = self.hyperparameters
        n = int(n or hp.get("num_samples", 100))
        seed = int(hp.get("random_seed", 1) if seed is None else seed)
        ids, ctx_dates, Zc = self._context(segments)
        net = self._net
        dt = next(net.parameters()).dtype
        fut = pd.DatetimeIndex(future_dates(ctx_dates.max(), horizon, freq=self.frequency))
        cov = torch.tensor(calendar_features(ctx_dates.append(fut)).to_numpy(), dtype=dt)
        wscale = self._window_scale(Zc)                                        # (1, m): the forecast is un-scaled with the same factor
        z = torch.tensor(Zc / wscale, dtype=dt)
        C = z.shape[0]
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            h, state = net.rnn(torch.cat([z[:-1], cov[1:C]], dim=-1)[None])         # warm-up over the context (steps 1..C-1)
            state = tuple(s.repeat(1, n, 1) for s in state)
            prev = z[-1][None].repeat(n, 1)
            out = []
            for k in range(horizon):
                x = torch.cat([prev, cov[C + k][None].repeat(n, 1)], dim=-1)[:, None, :]
                hk, state = net.rnn(x, state)
                mu, d, V = net.params(hk[:, 0])
                eps_r = torch.randn(n, net.rank, generator=gen, dtype=dt)
                eps_m = torch.randn(n, net.m, generator=gen, dtype=dt)
                prev = sample_lowrank(mu, V, d, eps_r, eps_m)
                out.append(prev)
        paths = torch.stack(out, 1).numpy() * wscale[None]                       # back to series-scaled units
        scales = np.array([self._meta["scales"][i] for i in ids])
        return paths * scales

    def infer_pool(self, segments: dict, horizon: int) -> dict:
        ids = self._meta["series"] if self._net is not None else []
        paths = self.sample_paths(segments, horizon)
        _, ctx_dates, _ = self._context(segments)
        fut = future_dates(ctx_dates.max(), horizon, freq=self.frequency)
        med = np.median(paths, axis=0)                                                # (horizon, m)
        return {sid: pd.DataFrame({"date": fut, "forecast": med[:, i]}) for i, sid in enumerate(ids)}

    def cross_series_correlation(self, segments: dict, horizon: int, n: int = 500) -> pd.DataFrame:
        """Correlation between the series' forecast errors implied by the joint samples (last horizon step)."""
        paths = self.sample_paths(segments, horizon, n=n)
        return pd.DataFrame(np.corrcoef(paths[:, -1, :].T), index=self._meta["series"], columns=self._meta["series"])

    # ---- save / load -------------------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from .torch_base import _plain
        dt = next(self._net.parameters()).dtype
        torch.save({"state_dict": self._net.state_dict(), "hyperparameters": _plain(self.hyperparameters), "meta": _plain(self._meta),
                    "dtype": str(dt).replace("torch.", ""), "fitted": _plain(self._fitted_model)}, path)

    def load(self, path: Path) -> None:
        st = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.hyperparameters = st["hyperparameters"]
        self._reconfigure()
        self._meta = st["meta"]
        hp = self.hyperparameters
        dtype = torch.float64 if st.get("dtype") == "float64" else torch.float32
        self._net = DeepVARNet(len(self._meta["series"]), self._meta["n_cov"], int(hp.get("hidden_size", 32)), int(hp.get("num_layers", 1)),
                               float(hp.get("dropout", 0.0)), int(hp.get("rank", 2)), dtype,
                               diagonal=str(hp.get("covariance", "low_rank")) == "diagonal")
        self._net.load_state_dict(st["state_dict"])
        self._net.eval()
        self._fitted_model = st.get("fitted")
