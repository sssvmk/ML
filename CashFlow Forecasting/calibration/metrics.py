"""Proper scores and tests used by the calibration run (numpy / scipy only)."""
from __future__ import annotations
import numpy as np


def crps_samples(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of an empirical predictive distribution. samples (n, ...) and obs (...) -> CRPS (...). E|X-y| - 0.5 E|X-X'| (sorted form)."""
    x = np.sort(samples, axis=0)
    n = x.shape[0]
    term1 = np.abs(x - obs[None]).mean(axis=0)
    i = np.arange(1, n + 1).reshape((n,) + (1,) * (x.ndim - 1))
    term2 = (2 * i - n - 1) * x
    return term1 - term2.sum(axis=0) / (n * n)


def energy_score(samples: np.ndarray, obs: np.ndarray, n_pairs: int = 400, seed: int = 0) -> float:
    """Multivariate energy score of samples (n, dim) versus obs (dim,)  (Monte-Carlo over sample pairs)."""
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    t1 = np.linalg.norm(samples - obs[None], axis=1).mean()
    a, b = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    t2 = np.linalg.norm(samples[a] - samples[b], axis=1).mean()
    return float(t1 - 0.5 * t2)


def central_interval_hits(samples: np.ndarray, obs: np.ndarray, level: float = 0.8) -> np.ndarray:
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return ((obs >= lo) & (obs <= hi)).astype(float)


def kupiec_pvalue(hits: int, n: int, p: float) -> float:
    """Kupiec unconditional-coverage likelihood-ratio test: H0 = the interval's true coverage is p. Returns the p-value."""
    from scipy.stats import chi2
    x = n - hits                                                   # misses
    q = 1 - p
    def ll(prob_miss):
        prob_miss = min(max(prob_miss, 1e-12), 1 - 1e-12)
        return x * np.log(prob_miss) + (n - x) * np.log(1 - prob_miss)
    lr = -2 * (ll(q) - ll(x / n))
    return float(chi2.sf(max(lr, 0.0), 1))
