"""
TimesFM-3 (roster #39; gap G-39; decisions D-8, D-13) -- ZERO-SHOT foundation model behind the algorithm interface.

  * The model code comes from Google's `timesfm` package (Apache-2.0, `timesfm3` PyTorch backend), a declared pip dependency, not vendored.
  * The WEIGHTS are read ONLY from `config.json -> prebuilt_models.timesfm3.path` (a directory in the Hugging Face layout, or a single
    .safetensors file); `local_files_only=True`, so nothing is ever downloaded at run time. The entry is disabled and has no path by
    default: the algorithm then reports itself unavailable (visible in the elimination log), it never falls back silently.
  * Licence guard (D-8): TimesFM 3.0 weights are distributed under `timesfm-non-commercial-license-v1.0` -- non-commercial, non-production
    use only. The entry must acknowledge that (`non_commercial_use_acknowledged`) and the run's `deployment.environment` must be listed in
    `allowed_environments`, otherwise the model refuses to load.
  * `train()` FITS NOTHING: it resolves and loads the weights (shared across modules through a process-level cache, since a backtest
    builds a fresh module per fold) and records the weights' fingerprint on the artifact. Fine-tuning is not part of this adapter.
  * `infer()` feeds the last `context_length` observations of the series (univariate) and returns the median as the point forecast; the
    nine quantiles are available through `infer_quantiles`. Horizons longer than the model's output patch are handled by the library.

Eligibility / required observations (decision D-13): only what the model needs to run -- `context_length` observations plus the horizon.
There is no 1,000-observation floor and no eligibility condition beyond history length. `context_length` is a hyperparameter (default
512; searchable) -- the default is a starting point chosen here, not a PRD value.

Not wired: the library's native covariate inputs (past-only / past-and-future covariates). This adapter forecasts the target alone.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .base import AlgorithmModule, EligibilityResult
from ._frames import DailyFrameMixin
from .prebuilt import resolve_prebuilt, PrebuiltSpec
from .utils import endogenous_series, future_dates

_CACHE: dict = {}
_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _forecaster(spec: PrebuiltSpec, device: str):
    key = (spec.fingerprint, device)
    with _LOCK:
        if key not in _CACHE:
            from timesfm3 import TimesFM3Forecaster
            _CACHE[key] = TimesFM3Forecaster.from_pretrained(str(spec.path), device=device, local_files_only=True)
        return _CACHE[key]


class TimesFM3Module(DailyFrameMixin, AlgorithmModule):
    name = "timesfm3"
    algorithm_version = "timesfm3-adapter-1"
    has_eligibility_condition = True
    prebuilt_key = "timesfm3"
    needs_horizon = True
    supports_hist_exog = False
    _use_calendar = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._spec: PrebuiltSpec | None = None
        self._model = None
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.horizon = int(hp.get("horizon", 4))
        self.context = int(hp.get("context_length", 512))
        self.device = str(hp.get("device", "cpu"))

    # ---- eligibility (D-13) -------------------------------------------------------------------------------------------
    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return self.context + int(horizon if horizon is not None else self.horizon)

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        n = len(endogenous_series(segment_df))
        need = self.required_observations()
        if n < need:
            return EligibilityResult(False, f"history {n} < context length {self.context} + horizon {self.horizon} = {need}")
        return EligibilityResult(True, f"history {n} covers context {self.context} + horizon {self.horizon}")

    def hyperparameter_search_space(self) -> dict:
        return {"context_length": {"type": "choice", "choices": [128, 256, 512, 1024]}}

    # ---- "training" = resolve + load, no fitting ---------------------------------------------------------------------------
    def _load_weights(self) -> None:
        hp = self.hyperparameters
        self._spec = resolve_prebuilt(hp.get("prebuilt"), self.prebuilt_key, default_fine_tune=False, environment=hp.get("environment"))
        self._model = _forecaster(self._spec, self.device)

    def train(self, segment_df) -> None:
        self._load_weights()
        self._fitted_model = {
            "zero_shot": True, "context_length": self.context, "horizon": self.horizon, "hyperparameters": self.hyperparameters,
            "prebuilt": {"key": self.prebuilt_key, "path": str(self._spec.path), "fingerprint": self._spec.fingerprint,
                         "fingerprint_verified": self._spec.fingerprint_verified, "licence_id": self._spec.entry.get("licence_id"),
                         "use_scope": self._spec.entry.get("use_scope")},
            "early_stopping": {"used": False, "note": "zero-shot: nothing is fitted, so there is no training loss or validation tail"},
        }
        self._resid = None

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._model.model.parameters())) if self._model is not None else 0

    # ---- inference ---------------------------------------------------------------------------------------------------------------
    def _context_values(self, segment_df) -> tuple[np.ndarray, pd.Timestamp]:
        if self._model is None:
            raise RuntimeError("load() or train() must run before infer()")
        fr = self._frame(segment_df)
        y = fr["y"].to_numpy(dtype=np.float32)
        if len(y) < self.context:
            raise ValueError(f"history of {len(y)} periods is shorter than the context length {self.context}")
        return y[-self.context:], pd.Timestamp(fr["ds"].max())

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ctx, last = self._context_values(segment_df)
        with _INFER_LOCK:
            out = self._model.predict(ctx, int(horizon), return_quantiles=False)
        return pd.DataFrame({"date": future_dates(last, horizon, freq=self.frequency), "forecast": np.asarray(out.forecast, dtype=float)[:horizon]})

    def infer_quantiles(self, segment_df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
        """Date + one column per model quantile (q10..q90), sorted per row by the library."""
        h = int(horizon or self.horizon)
        ctx, last = self._context_values(segment_df)
        with _INFER_LOCK:
            out = self._model.predict(ctx, h, return_quantiles=True, sort_quantiles=True)
        qs = list(self._model.config.quantiles)
        q = np.asarray(out.quantiles, dtype=float)[:h]
        res = pd.DataFrame({f"q{int(round(x * 100))}": q[:, i] for i, x in enumerate(qs)})
        res.insert(0, "date", future_dates(last, h, freq=self.frequency))
        return res

    # ---- save / load (the weights stay where the config says; only the reference is stored) ---------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"hyperparameters": self.hyperparameters, "fitted": self._fitted_model}, f)

    def load(self, path: Path) -> None:
        with open(Path(path), "rb") as f:
            st = pickle.load(f)
        self.hyperparameters, self._fitted_model = st["hyperparameters"], st.get("fitted")
        self._reconfigure()
        self._load_weights()      # re-resolves the configured path: a moved / missing / changed checkpoint fails loudly, naming the key
