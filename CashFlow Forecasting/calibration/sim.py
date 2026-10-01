"""
Controlled synthetic data with KNOWN ground truth for the Layer 2 calibration run (G-42) and the true-parameter oracles.

  DeepState data: local level + weekly seasonality, observed with noise:
        y_t = level_t + s_{t mod 7} + e_t,   level_t = level_{t-1} + u_t,   u ~ N(0, sigma_level^2),  e ~ N(0, sigma_obs^2)
  DeepVAR data: 3-dimensional VAR(1), x_t = A x_{t-1} + w_t, spectral radius of A = 0.8, w ~ N(0, Sigma) with innovation
        correlation 0 or 0.6; each dimension is shifted / scaled to its own level (no FX conversion is involved anywhere).
The oracles use the TRUE parameters: statsmodels' Kalman smoother for DeepState (an implementation independent of ours), the exact
conditional Gaussian of the VAR for DeepVAR.
"""
from __future__ import annotations
import numpy as np

START = "2024-01-01"


# ---------------------------------------------------------------------------------------------------- DeepState data + oracle
def sim_deepstate(seed: int, n_series: int = 6, T: int = 350, sigma_level: float = 0.6, sigma_obs: float = 2.0, season_amp: float = 6.0,
                  base: float = 100.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    Y = np.empty((n_series, T))
    seasons = []
    for i in range(n_series):
        pattern = season_amp * rng.normal(0, 1, 7)
        pattern -= pattern.mean()                                  # a cycle sums to zero
        level = base * (0.6 + 0.8 * rng.random()) + np.cumsum(rng.normal(0, sigma_level, T))
        Y[i] = level + pattern[t % 7] + rng.normal(0, sigma_obs, T)
        seasons.append(pattern)
    truth = {"sigma_level": sigma_level, "sigma_obs": sigma_obs, "seasons": np.array(seasons)}
    return Y, truth


def oracle_deepstate(y_hist: np.ndarray, H: int, sigma_level: float, sigma_obs: float):
    """True-parameter forecast (statsmodels UnobservedComponents, local level + fixed 7-period seasonal). Returns mean, var (H,)."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    mod = UnobservedComponents(y_hist, level="local level", seasonal=7, stochastic_seasonal=False)
    names = list(mod.param_names)
    params = [{"sigma2.irregular": sigma_obs ** 2, "sigma2.level": sigma_level ** 2}[n] for n in names]
    res = mod.smooth(params)
    fc = res.get_forecast(H)
    return np.asarray(fc.predicted_mean), np.asarray(fc.var_pred_mean)


# ---------------------------------------------------------------------------------------------------- DeepVAR data + oracle
def var_system(seed: int, dim: int = 3, radius: float = 0.8, corr: float = 0.0, sigma: float = 1.0):
    rng = np.random.default_rng(10_000 + seed)
    M = rng.normal(0, 1, (dim, dim))
    A = M * (radius / max(abs(np.linalg.eigvals(M))))              # spectral radius exactly `radius`
    Sigma = sigma ** 2 * ((1 - corr) * np.eye(dim) + corr * np.ones((dim, dim)))
    return A, Sigma


def sim_var1(seed: int, T: int = 400, dim: int = 3, radius: float = 0.8, corr: float = 0.0, levels=(100.0, 1000.0, 20.0), scales=(5.0, 50.0, 1.0)):
    A, Sigma = var_system(seed, dim, radius, corr)
    rng = np.random.default_rng(20_000 + seed)
    x = np.zeros((T + 100, dim))
    L = np.linalg.cholesky(Sigma)
    for t in range(1, T + 100):
        x[t] = A @ x[t - 1] + L @ rng.normal(0, 1, dim)
    x = x[100:]                                                    # burn-in
    Y = (np.asarray(levels)[None] + x * np.asarray(scales)[None]).T   # (dim, T)
    return Y, {"A": A, "Sigma": Sigma, "levels": np.asarray(levels), "scales": np.asarray(scales), "corr": corr}


def oracle_var1(Y_hist: np.ndarray, H: int, truth: dict, n_samples: int = 200, seed: int = 0):
    """Exact conditional Gaussian of the VAR given the last observation. Returns samples (n, H, dim) in original units."""
    A, Sigma, lv, sc = truth["A"], truth["Sigma"], truth["levels"], truth["scales"]
    x0 = (Y_hist[:, -1] - lv) / sc
    rng = np.random.default_rng(seed)
    dim = len(x0)
    L = np.linalg.cholesky(Sigma)
    x = np.tile(x0, (n_samples, 1))
    out = []
    for _ in range(H):
        x = x @ A.T + rng.normal(0, 1, (n_samples, dim)) @ L.T
        out.append(x * sc + lv)
    return np.stack(out, 1)
