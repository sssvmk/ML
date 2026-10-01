"""
DeepState (PRD §3.2 #26; gap G-26; decisions D-4 / D-11) -- native PyTorch, written from the paper's formulation (Rangapuram et al.
2018) with no code ported from the MXNet reference. It is a CUSTOM model, so it must pass the layered admission validation of G-42
(admission.py) before it competes; until then config.json keeps it disabled and `requires_admission` keeps it out of the candidates.

Model (per series i, scaled by its own scale v_i -- no FX conversion, decision on G-03):
    latent state   l_t = F l_{t-1} + g_t * eps_t,     eps_t ~ N(0, 1)
    observation    z_t = a^T l_{t-1} + b_t + sigma_t * nu_t,   nu_t ~ N(0, 1)
  * F and a are FIXED by the component structure: level + trend + one dummy-form seasonal block per period (default weekly, 7).
  * ONE recurrent network shared by all pooled series maps each series' covariates (calendar features, optional future-known
    covariates, a static-attribute embedding) to the per-step parameters g_t (innovation strengths), b_t (offset), sigma_t (noise).
  * Training maximises the marginal likelihood computed by a DIFFERENTIABLE KALMAN FILTER over windows of the pooled series.
  * Forecast = Kalman prediction from the filtered state: exact Gaussian mean/variance per step (uncertainty grows with the
    horizon) and sample paths that carry the state uncertainty.

Deviation from the matrix text, stated: the spec lists F_t and a_t among the network's outputs; here they are structural (as in the
paper's level/trend/seasonality models), so only g_t, b_t and sigma_t are learned.

Eligibility (inherited from DeepAR, PRD): >= 2 series in the pool and pooled N >= 1,000; a single segment is not a pool.
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
from ._frames import DailyFrameMixin, calendar_features, CALENDAR_COLS
from .utils import future_dates, future_known_datasets, exogenous_frame

LOG2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------ structure + Kalman filter
def build_structure(seasonal_periods=(7,), dtype=torch.float64):
    """F (n x n) and a (n) for level + trend + a dummy-form seasonal block per period. Returns (F, a, state_names)."""
    periods = [int(p) for p in seasonal_periods]
    n = 2 + sum(p - 1 for p in periods)
    F = torch.zeros(n, n, dtype=dtype)
    a = torch.zeros(n, dtype=dtype)
    F[0, 0] = F[0, 1] = F[1, 1] = 1.0                  # level_t = level + trend ; trend_t = trend
    a[0] = 1.0
    names = ["level", "trend"]
    i = 2
    for p in periods:
        k = p - 1
        F[i, i:i + k] = -1.0                            # new seasonal effect = -(sum of the previous p-1), so a full cycle sums to 0
        for j in range(1, k):
            F[i + j, i + j - 1] = 1.0
        a[i] = 1.0
        names += [f"season{p}_{j}" for j in range(k)]
        i += k
    return F, a, names


def kalman_loglik(z, F, a, g, sigma, b, m0, P0, mask=None, return_last=False):
    """
    Marginal log-likelihood of `z` (B, T) under the model above, by the Kalman filter, differentiable end to end.
      F (n, n), a (n,)   structural;   g (B, T, n), sigma (B, T), b (B, T)   per-step parameters;
      m0 (B, n), P0 (B, n, n)   distribution of l_0 before z_1 is seen;   mask (B, T) True where z_t is observed.
    Returns (loglik (B,), m_filtered, P_filtered) -- the last two only when `return_last` (state of l_{T-1} given z_1..z_T).
    """
    B, T = z.shape
    m, P = m0, P0
    ll = z.new_zeros(B)
    m_f, P_f = m, P
    for t in range(T):
        Pa = P @ a                                                    # (B, n)
        S = (Pa @ a) + sigma[:, t] ** 2                               # innovation variance
        v = z[:, t] - (m @ a + b[:, t])
        contrib = -0.5 * (LOG2PI + torch.log(S) + v ** 2 / S)
        K = Pa / S[:, None]
        if mask is not None:
            obs = mask[:, t]
            contrib = torch.where(obs, contrib, torch.zeros_like(contrib))
            K = torch.where(obs[:, None], K, torch.zeros_like(K))
            v = torch.where(obs, v, torch.zeros_like(v))
        ll = ll + contrib
        m_f = m + K * v[:, None]
        P_f = P - K[:, :, None] * (Pa[:, None, :] * (1.0 if mask is None else 1.0))
        if mask is not None:
            P_f = torch.where(mask[:, t][:, None, None], P_f, P)
        P_f = 0.5 * (P_f + P_f.transpose(1, 2))
        m = m_f @ F.T
        P = F @ P_f @ F.T + g[:, t, :, None] * g[:, t, None, :]
        P = 0.5 * (P + P.transpose(1, 2))
    if return_last:
        return ll, m_f, P_f
    return ll


def kalman_predict(m, P, F, a, g, sigma, b):
    """
    Predictive distribution of the next H observations from the prior (m, P) of l for the first of them.
    g (B, H, n), sigma (B, H), b (B, H). Returns mean (B, H) and variance (B, H).
    """
    B, H = sigma.shape
    means, varis = [], []
    for k in range(H):
        means.append(m @ a + b[:, k])
        varis.append(((P @ a) @ a) + sigma[:, k] ** 2)
        m = m @ F.T
        P = F @ P @ F.T + g[:, k, :, None] * g[:, k, None, :]
        P = 0.5 * (P + P.transpose(1, 2))
    return torch.stack(means, 1), torch.stack(varis, 1)


# ------------------------------------------------------------------------------------------------ network
class DeepStateNet(nn.Module):
    def __init__(self, n_cov: int, n_static: int, hidden: int, layers: int, dropout: float, emb: int, seasonal_periods, dtype=torch.float32):
        super().__init__()
        F, a, self.state_names = build_structure(seasonal_periods, dtype)
        self.register_buffer("F", F)
        self.register_buffer("a", a)
        self.n = F.shape[0]
        self.n_cov, self.n_static, self.emb_dim = n_cov, n_static, (emb if n_static else 0)
        self.static_emb = nn.Linear(n_static, emb) if n_static else None
        self.rnn = nn.LSTM(n_cov + self.emb_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, self.n + 2)                      # g (n), sigma, b
        with torch.no_grad():                                          # start with small innovations, moderate noise
            self.head.bias[: self.n].fill_(-3.0)
            self.head.bias[self.n].fill_(-1.0)
            self.head.bias[self.n + 1].fill_(0.0)
        self.log_p0 = nn.Parameter(torch.zeros(()))                    # log std of the diffuse initial-state prior
        self.to(dtype)

    def ss_params(self, cov, static):
        x = cov
        if self.static_emb is not None:
            e = self.static_emb(static)[:, None, :].expand(-1, cov.shape[1], -1)
            x = torch.cat([cov, e], dim=-1)
        h, _ = self.rnn(x)
        o = self.head(h)
        g = Fn.softplus(o[..., : self.n]) + 1e-4
        sigma = Fn.softplus(o[..., self.n]) + 1e-3
        b = torch.tanh(o[..., self.n + 1])
        return g, sigma, b

    def init_state(self, z0):
        B = z0.shape[0]
        m0 = torch.zeros(B, self.n, dtype=z0.dtype, device=z0.device)
        m0[:, 0] = z0                                                  # level starts at the window's first observation
        P0 = torch.eye(self.n, dtype=z0.dtype, device=z0.device)[None].repeat(B, 1, 1) * torch.exp(2 * self.log_p0)
        return m0, P0

    def nll(self, z, cov, static):
        """Negative log-likelihood per observation, averaged over the batch."""
        g, sigma, b = self.ss_params(cov, static)
        m0, P0 = self.init_state(z[:, 0])
        ll = kalman_loglik(z, self.F, self.a, g, sigma, b, m0, P0)
        return -(ll / z.shape[1]).mean()


# ------------------------------------------------------------------------------------------------ module
class DeepStateModule(DailyFrameMixin, AlgorithmModule):
    name = "deepstate"
    algorithm_version = "deepstate-native-1"
    has_eligibility_condition = True
    pooling_capable = True
    pooled_n_basis = "pooled_total"
    requires_admission = True          # custom model: must pass the G-42 layered validation (admission.py) to compete
    needs_horizon = True
    supports_hist_exog = False
    default_max_epochs = 30

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._net = None
        self._meta: dict = {}
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.horizon = int(hp.get("horizon", 4))
        self.context = int(hp.get("context_length", 56))
        self.periods = [int(p) for p in hp.get("seasonal_periods", [7])]
        self._use_calendar = True

    # ---- contract / eligibility ----------------------------------------------------------------------------------------
    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return 1000     # PRD v10 §3.2 row #26: Pooled N >= 1,000

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        return EligibilityResult(False, "pooling-capable algorithm: needs a pool of >= 2 series "
                                        "(train it through Orchestrator.full_train_pooled); one segment is not a pool")

    def hyperparameter_search_space(self) -> dict:
        return {
            "hidden_size": {"type": "choice", "choices": [16, 32, 64]},
            "num_layers": {"type": "int", "low": 1, "high": 2},
            "context_length": {"type": "int", "low": 28, "high": 112},
            "embedding_dim": {"type": "choice", "choices": [2, 4, 8]},
            "seasonal_periods": {"type": "choice", "choices": [[7], [7, 30]]},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 3e-2, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }

    # ---- covariates ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _fk_scaled(df: pd.DataFrame, names: list[str], scale_from: pd.Timestamp):
        """Date x k frame of future-known covariates, each divided by (1 + mean|value|) over the history (feature scaling)."""
        wide = exogenous_frame(df)[list(names)].copy()
        wide.columns = [f"fk_{i}" for i in range(len(names))]
        hist = wide[wide.index <= scale_from]
        return wide / (1.0 + hist.abs().mean())

    def _cov_matrix(self, dates: pd.DatetimeIndex, fk: pd.DataFrame | None, n_fk: int):
        cov = calendar_features(dates).to_numpy(dtype=np.float64)
        if n_fk:
            vals = fk.reindex(dates)
            if vals.isna().any().any():
                raise ValueError(f"future-known covariates are missing for some of the {len(dates)} periods "
                                 f"({dates.min().date()}..{dates.max().date()}); the model cannot forecast without them")
            cov = np.concatenate([cov, vals.to_numpy(dtype=np.float64)], axis=1)
        return cov

    def _static_vec(self, sid: str, seg_df: pd.DataFrame) -> np.ndarray:
        now = {"company_code": str(seg_df["company_code"].iloc[0]),
               "currency":     str(seg_df["currency"].iloc[0]),
               "dataset":      str(seg_df.loc[seg_df["series_role"] == "endogenous", "dataset"].iloc[0]
                                    if (seg_df["series_role"] == "endogenous").any() else "")}
        known = self._meta["attrs"].get(sid)
        if known is not None and known != now:
            raise ValueError(f"attributes of series {sid!r} changed since pooled training: trained {known}, supplied {now}")
        v = []
        for key, values in self._meta["cats"].items():
            if now[key] not in values:
                raise ValueError(f"{key} {now[key]!r} of series {sid!r} was not seen in pooled training (unseen category; known: {sorted(values)})")
            v += [1.0 if val == now[key] else 0.0 for val in values]
        return np.asarray(v, dtype=np.float64)

    # ---- train ---------------------------------------------------------------------------------------------------------------
    def train(self, segment_df) -> None:
        raise RuntimeError("deepstate is a pooled model: train it with train_pooled(batch) / Orchestrator.full_train_pooled")

    def _series_arrays(self, batch):
        out = {}
        for sid in batch.segment_ids:
            df = batch.segments[sid]
            fr = self._frame(df)
            names = future_known_datasets(df, min_future=self.horizon) if self.hyperparameters.get("use_future_known", True) else []
            out[sid] = (fr, names)
        return out

    def train_pooled(self, batch) -> None:
        hp = self.hyperparameters
        seed = int(hp.get("random_seed", 1))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dtype = torch.float64 if str(hp.get("dtype", "float32")) == "float64" else torch.float32
        arr = self._series_arrays(batch)
        counts = {len(v[1]) for v in arr.values()}
        n_fk = next(iter(counts)) if len(counts) == 1 else 0
        note = None if len(counts) == 1 or counts == {0} else f"future-known series ignored: series carry different numbers of them ({sorted(counts)})"
        if n_fk == 0:
            arr = {sid: (fr, []) for sid, (fr, _) in arr.items()}
        cats = batch.categories()
        attrs = {sid: dict(a) for sid, a in batch.attributes.items()}
        static_cols = [(k, v) for k, vals in cats.items() for v in vals]
        W = self.context + self.horizon
        data = {}
        for sid, (fr, names) in arr.items():
            z = fr["y"].to_numpy(dtype=np.float64) / batch.scales[sid]
            dates = pd.DatetimeIndex(fr["ds"])
            fk = self._fk_scaled(batch.segments[sid], names, dates.max()) if n_fk else None
            cov = self._cov_matrix(dates, fk, n_fk)
            st = np.array([1.0 if attrs[sid][k] == v else 0.0 for k, v in static_cols])
            if len(z) < 2 * W + 20:
                raise ValueError(f"series {sid} has {len(z)} observations; need >= {2 * W + 20} for windows of {W} plus a validation tail")
            data[sid] = (z, cov, st)
        ids = list(data)
        n_cov = data[ids[0]][1].shape[1]
        self._net = DeepStateNet(n_cov, len(static_cols), int(hp.get("hidden_size", 32)), int(hp.get("num_layers", 1)),
                                 float(hp.get("dropout", 0.0)), int(hp.get("embedding_dim", 4)), self.periods, dtype)
        patience = int(hp.get("early_stop_patience", 5))
        vf = float(hp.get("validation_fraction", 0.15))
        bounds = {}
        for sid in ids:
            n = len(data[sid][0])
            n_val = int(max(2 * W, vf * n)) if patience > 0 else 0
            bounds[sid] = (n - n_val, n)          # windows for training lie fully in [0, n - n_val); validation windows in [n - n_val, n)
        opt = torch.optim.Adam(self._net.parameters(), lr=float(hp.get("learning_rate", 1e-2)))
        clip = float(hp.get("gradient_clip_val", 1.0))
        bs, steps = int(hp.get("batch_size", 32)), int(hp.get("steps_per_epoch", 30))

        def make_batch(region: str, size: int, local_rng):
            zs, cs, ss = [], [], []
            for _ in range(size):
                sid = ids[local_rng.integers(len(ids))]
                z, cov, st = data[sid]
                cut, n = bounds[sid]
                lo, hi = (0, cut) if region == "train" else (cut, n)
                start = int(local_rng.integers(lo, hi - W + 1))
                zs.append(z[start:start + W]); cs.append(cov[start:start + W]); ss.append(st)
            return (torch.tensor(np.stack(zs), dtype=dtype), torch.tensor(np.stack(cs), dtype=dtype), torch.tensor(np.stack(ss), dtype=dtype))

        val_rng = np.random.default_rng(seed + 12345)
        val_batch = make_batch("val", 4 * bs, val_rng) if patience > 0 else None
        best, bad, history = None, 0, []
        for epoch in range(int(hp.get("max_epochs", self.default_max_epochs))):
            self._net.train()
            losses = []
            for _ in range(steps):
                zb, cb, sb = make_batch("train", bs, rng)
                opt.zero_grad()
                loss = self._net.nll(zb, cb, sb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), clip)
                opt.step()
                losses.append(float(loss))
            tr = float(np.mean(losses))
            rec = {"epoch": epoch, "train_nll": tr}
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
        self._meta = {"cats": cats, "attrs": attrs, "scales": dict(batch.scales), "static_cols": static_cols, "n_fk": n_fk,
                      "n_cov": n_cov, "series": list(ids)}
        self._fitted_model = {
            "hyperparameters": self.hyperparameters, "n_parameters": self.n_parameters(),
            "pooled": {"series": list(ids), "attributes": attrs, "n_series": len(ids), "future_known_channels": n_fk, "future_known_note": note},
            "early_stopping": {"used": val_batch is not None, "epochs_run": len(history), "max_epochs": int(hp.get("max_epochs", self.default_max_epochs)),
                               "stopped_early": len(history) < int(hp.get("max_epochs", self.default_max_epochs)),
                               "best_epoch": best[2] if best else None, "restored_best_checkpoint": restored,
                               "loss_units": "negative log-likelihood per observation, scaled series (train and validation alike)",
                               "history": history}}
        self._resid = None

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters())) if self._net is not None else 0

    def network(self):
        return self._net

    # ---- filtering / forecasting ---------------------------------------------------------------------------------------
    def _prepare(self, segment_df: pd.DataFrame, horizon: int):
        if self._net is None:
            raise RuntimeError("load() or train_pooled() must run before infer()")
        fr = self._frame(segment_df)
        sid = str(fr["unique_id"].iloc[0])
        y = fr["y"].to_numpy(dtype=np.float64)
        dates = pd.DatetimeIndex(fr["ds"])
        if len(y) < self.context:
            raise ValueError(f"history of {len(y)} periods is shorter than the context length {self.context}")
        from pooling import series_scale
        scale = self._meta["scales"].get(sid) or series_scale(pd.Series(y))
        n_fk = self._meta["n_fk"]
        fk = None
        if n_fk:
            names = future_known_datasets(segment_df, min_future=horizon if horizon <= 0 else min(horizon, self.horizon))
            if len(names) != n_fk:
                raise ValueError(f"the pooled model was trained with {n_fk} future-known series per series; these rows supply {len(names)}")
            fk = self._fk_scaled(segment_df, names, dates.max())
        ctx_dates = dates[-self.context:]
        fut_dates = pd.DatetimeIndex(future_dates(dates.max(), horizon, freq=self.frequency))
        all_dates = ctx_dates.append(fut_dates)
        cov = self._cov_matrix(all_dates, fk, n_fk)
        dt = next(self._net.parameters()).dtype
        z = torch.tensor(y[-self.context:] / scale, dtype=dt)[None]
        cov_t = torch.tensor(cov, dtype=dt)[None]
        st = torch.tensor(self._static_vec(sid, segment_df), dtype=dt)[None]
        return sid, scale, z, cov_t, st, fut_dates

    def _filter_and_predict(self, segment_df: pd.DataFrame, horizon: int):
        sid, scale, z, cov, st, fut_dates = self._prepare(segment_df, horizon)
        net = self._net
        with torch.no_grad():
            g, sigma, b = net.ss_params(cov, st)
            C = z.shape[1]
            m0, P0 = net.init_state(z[:, 0])
            _, m_f, P_f = kalman_loglik(z, net.F, net.a, g[:, :C], sigma[:, :C], b[:, :C], m0, P0, return_last=True)
            m1 = m_f @ net.F.T
            P1 = net.F @ P_f @ net.F.T + g[:, C - 1, :, None] * g[:, C - 1, None, :]
            mean, var = kalman_predict(m1, P1, net.F, net.a, g[:, C:], sigma[:, C:], b[:, C:])
        return scale, mean[0].numpy(), var[0].numpy(), (m_f[0].numpy(), P_f[0].numpy()), (m1, P1, g[:, C:], sigma[:, C:], b[:, C:]), fut_dates

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        scale, mean, var, _, _, fut_dates = self._filter_and_predict(segment_df, horizon)
        return pd.DataFrame({"date": fut_dates, "forecast": mean * scale})

    def infer_quantiles(self, segment_df: pd.DataFrame, horizon: int | None = None, levels=(0.8, 0.9)) -> pd.DataFrame:
        """Exact Gaussian predictive intervals: mean and lo/hi bounds per level; the width grows with the horizon."""
        from statistics import NormalDist
        h = int(horizon or self.horizon)
        scale, mean, var, _, _, fut_dates = self._filter_and_predict(segment_df, h)
        out = pd.DataFrame({"date": fut_dates, "mean": mean * scale, "std": np.sqrt(var) * scale})
        for lv in levels:
            zq = NormalDist().inv_cdf(0.5 + lv / 2)
            out[f"lo-{int(lv * 100)}"] = out["mean"] - zq * out["std"]
            out[f"hi-{int(lv * 100)}"] = out["mean"] + zq * out["std"]
        return out

    def filtered_state(self, segment_df: pd.DataFrame) -> dict:
        """Filtered level / trend / seasonal state at the forecast origin (state tensors of the Kalman filter)."""
        _, _, _, (m, P), _, _ = self._filter_and_predict(segment_df, 1)
        return {"names": list(self._net.state_names), "mean": m, "variance": np.diag(P)}

    def sample_paths(self, segment_df: pd.DataFrame, horizon: int, n: int = 100, seed: int = 0) -> np.ndarray:
        """(n, horizon) sample paths in original units that carry the state uncertainty (level/trend/seasonal draws + noise)."""
        scale, _, _, _, (m1, P1, g, sigma, b), _ = self._filter_and_predict(segment_df, horizon)
        net = self._net
        gen = torch.Generator().manual_seed(seed)
        dt = m1.dtype
        L = torch.linalg.cholesky(P1[0] + 1e-9 * torch.eye(net.n, dtype=dt))
        alpha = m1[0][None] + torch.randn(n, net.n, generator=gen, dtype=dt) @ L.T
        paths = []
        for k in range(horizon):
            paths.append(alpha @ net.a + b[0, k] + sigma[0, k] * torch.randn(n, generator=gen, dtype=dt))
            alpha = alpha @ net.F.T + g[0, k][None] * torch.randn(n, 1, generator=gen, dtype=dt)
        return torch.stack(paths, 1).numpy() * scale

    # ---- save / load ----------------------------------------------------------------------------------------------------------
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
        meta = st["meta"]
        meta["static_cols"] = [tuple(x) for x in meta["static_cols"]]
        self._meta = meta
        hp = self.hyperparameters
        dtype = torch.float64 if st.get("dtype") == "float64" else torch.float32
        self._net = DeepStateNet(meta["n_cov"], len(meta["static_cols"]), int(hp.get("hidden_size", 32)), int(hp.get("num_layers", 1)),
                                 float(hp.get("dropout", 0.0)), int(hp.get("embedding_dim", 4)), self.periods, dtype)
        self._net.load_state_dict(st["state_dict"])
        self._net.eval()
        self._fitted_model = st.get("fitted")
