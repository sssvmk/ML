from __future__ import annotations
import pickle
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from .base import AlgorithmModule, EligibilityResult
from ._frames import DailyFrameMixin
from .utils import endogenous_series, future_dates


def receptive_field(dilation_depth: int, num_stacks: int) -> int:
    """GluonTS WaveNet (kernel size 2): dilations 1,2,4,...,2**(depth-1) repeated `num_stacks` times; RF = sum(dilations) + 1."""
    return 1 + int(num_stacks) * (2 ** int(dilation_depth) - 1)


class WaveNetModule(DailyFrameMixin, AlgorithmModule):
    """
    WaveNet (PRD §3.2 #24; gap G-24). Route (D-4): GluonTS PyTorch `WaveNetEstimator` behind the adapter. Architecture (GluonTS
    `gluonts.torch.model.wavenet`): causal 1x1 input projection; residual blocks of dilated causal Conv1d with the gated
    activation sigmoid(conv) * tanh(conv), dilations 1,2,4,..,2**(depth-1) repeated for `num_stacks` cycles; per-block residual
    and skip 1x1 convs; summed skip outputs -> ReLU -> 1x1 -> ReLU -> 1x1 head.

    DEVIATIONS from the matrix text (both follow from the approved library route, not from a choice made here):
      * the output head is a CATEGORICAL distribution over `num_bins` quantised values (ancestral sampling of
        `num_parallel_samples` paths; the median is the point forecast), not the continuous regression head the spec describes;
      * time features (day-of-week / day-of-month / ...) come from GluonTS's own frequency-based features; there are no
        historical exogenous inputs, so exogenous datasets are not used.

    Eligibility (inherited rule from TCN): history must cover the receptive field of the real network,
    RF = 1 + num_stacks*(2**depth - 1). Required observations: N >= max(1000, RF + horizon + 500).
    """

    name = "wavenet"
    algorithm_version = "gluonts0.17-1"
    has_eligibility_condition = True
    needs_horizon = True
    supports_hist_exog = False
    context_family_floor = 1000
    _use_calendar = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        self._predictor = None
        self._reconfigure()

    def _reconfigure(self) -> None:
        hp = self.hyperparameters
        self.horizon = int(hp.get("horizon", 4))
        self.dilation_depth = int(hp.get("dilation_depth", 6))
        self.num_stacks = int(hp.get("num_stacks", 1))
        self.receptive_field = receptive_field(self.dilation_depth, self.num_stacks)

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        return max(self.context_family_floor, self.receptive_field + int(horizon if horizon is not None else self.horizon) + 500)

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        n = len(endogenous_series(segment_df))
        if n < self.receptive_field + self.horizon:
            return EligibilityResult(False, f"history {n} < receptive field {self.receptive_field} + horizon {self.horizon} "
                                            f"(dilation_depth={self.dilation_depth}, stacks={self.num_stacks})")
        return EligibilityResult(True, f"history {n} covers receptive field {self.receptive_field}")

    def hyperparameter_search_space(self) -> dict:
        return {
            "dilation_depth": {"type": "int", "low": 4, "high": 9},
            "num_stacks": {"type": "int", "low": 1, "high": 3},
            "num_residual_channels": {"type": "choice", "choices": [8, 16, 32]},
            "num_skip_channels": {"type": "choice", "choices": [16, 32, 64]},
            "num_bins": {"type": "choice", "choices": [128, 256, 512]},
            "lr": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }

    # ---- data / estimator ---------------------------------------------------------------------------------------
    @staticmethod
    def _dataset(y: np.ndarray, start: pd.Timestamp, freq: str = "D"):
        from gluonts.dataset.common import ListDataset
        return ListDataset([{"start": pd.Period(start, freq), "target": np.asarray(y, dtype=np.float32)}], freq=freq)

    def _estimator(self, tmp: str, val: bool):
        from gluonts.torch.model.wavenet import WaveNetEstimator
        import lightning.pytorch as pl
        hp = self.hyperparameters
        cbs = [pl.callbacks.EarlyStopping(monitor="val_loss", patience=int(hp.get("early_stop_patience", 3)), mode="min")] if val else []
        tk = {"max_epochs": int(hp.get("max_epochs", 5)), "accelerator": "cpu", "devices": 1, "enable_progress_bar": False,
              "enable_model_summary": False, "logger": False, "default_root_dir": tmp, "callbacks": cbs}
        return WaveNetEstimator(
            freq=self.frequency, prediction_length=self.horizon, num_bins=int(hp.get("num_bins", 256)),
            num_residual_channels=int(hp.get("num_residual_channels", 16)), num_skip_channels=int(hp.get("num_skip_channels", 16)),
            dilation_depth=self.dilation_depth, num_stacks=self.num_stacks, temperature=float(hp.get("temperature", 1.0)),
            lr=float(hp.get("lr", 1e-3)), batch_size=int(hp.get("batch_size", 32)),
            num_batches_per_epoch=int(hp.get("num_batches_per_epoch", 50)),
            num_parallel_samples=int(hp.get("num_parallel_samples", 100)), trainer_kwargs=tk)

    def train(self, segment_df: pd.DataFrame) -> None:
        import lightning.pytorch as pl
        hp = self.hyperparameters
        frame = self._frame(segment_df)
        y, start = frame["y"].to_numpy(dtype=np.float32), pd.Timestamp(frame["ds"].min())
        patience = int(hp.get("early_stop_patience", 3))
        n_val = int(max(2 * self.horizon, float(hp.get("validation_fraction", 0.15)) * len(y))) if patience > 0 else 0
        if len(y) - n_val < self.receptive_field + self.horizon + 100:
            n_val = 0
        tmp = tempfile.mkdtemp(prefix="stc_wavenet_")
        try:
            pl.seed_everything(int(hp.get("random_seed", 1)), workers=False)
            est = self._estimator(tmp, val=bool(n_val))
            train_ds = self._dataset(y[: len(y) - n_val], start, self.frequency)
            val_ds = self._dataset(y, start, self.frequency) if n_val else None       # validation windows are drawn from the chronological tail
            self._predictor = est.train(train_ds, validation_data=val_ds, num_workers=0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._fitted_model = {"receptive_field": self.receptive_field, "horizon": self.horizon, "hyperparameters": self.hyperparameters,
                              "n_parameters": self.n_parameters(),
                              "early_stopping": {"used": bool(n_val), "validation_rows": int(n_val),
                                                 "max_epochs": int(hp.get("max_epochs", 5)), "restored_best_checkpoint": bool(n_val),
                                                 "note": "GluonTS reloads the best-validation checkpoint after training"}}

    def network(self):
        return self._predictor.prediction_net.model if self._predictor is not None else None

    def n_parameters(self) -> int:
        net = self.network()
        return int(sum(p.numel() for p in net.parameters())) if net is not None else 0

    # ---- infer -----------------------------------------------------------------------------------------------------
    def _samples(self, y: np.ndarray, start: pd.Timestamp) -> np.ndarray:
        import torch
        torch.manual_seed(int(self.hyperparameters.get("random_seed", 1)))
        fc = next(iter(self._predictor.predict(self._dataset(y, start, self.frequency))))
        return np.asarray(fc.samples)                                   # (num_samples, horizon)

    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._predictor is None:
            raise RuntimeError("load() or train() must run before infer()")
        frame = self._frame(segment_df)
        y, start = frame["y"].to_numpy(dtype=np.float32), pd.Timestamp(frame["ds"].min())
        preds: list[float] = []
        while len(preds) < horizon:
            med = np.median(self._samples(y, start), axis=0)
            preds.extend(float(v) for v in med)
            y = np.concatenate([y, med.astype(np.float32)])            # roll forward in chunks of the trained horizon
        return pd.DataFrame({"date": future_dates(frame["ds"].max(), horizon, freq=self.frequency), "forecast": preds[:horizon]})

    def infer_quantiles(self, segment_df: pd.DataFrame, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Sample-based quantiles of the first `horizon` steps (the categorical output's distribution)."""
        frame = self._frame(segment_df)
        s = self._samples(frame["y"].to_numpy(dtype=np.float32), pd.Timestamp(frame["ds"].min()))
        out = {f"q{int(q * 100)}": np.quantile(s, q, axis=0) for q in quantiles}
        return pd.DataFrame({"date": future_dates(frame["ds"].max(), self.horizon, freq=self.frequency), **out})

    # ---- save / load ------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = path.parent / "gluonts_predictor"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        self._predictor.serialize(d)
        with open(path, "wb") as f:
            pickle.dump({"hyperparameters": self.hyperparameters, "fitted": self._fitted_model, "dir": "gluonts_predictor"}, f)

    def load(self, path: Path) -> None:
        import torch
        from gluonts.model.predictor import Predictor
        path = Path(path)
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.hyperparameters, self._fitted_model = st["hyperparameters"], st.get("fitted")
        self._reconfigure()
        self._predictor = Predictor.deserialize(path.parent / st["dir"], device=torch.device("cpu"))
