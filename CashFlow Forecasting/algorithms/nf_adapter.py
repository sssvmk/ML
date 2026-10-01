"""
Adapter layer: NeuralForecast models behind the ten AlgorithmModule interfaces (PRD §5.1; gaps G-41 and
G-20..G-38 for the library-routed models, decisions D-4/D-6/D-9).

What the adapter does, and deliberately does not do:
  * The architecture is the LIBRARY's own nn.Module (no shared MLP/sklearn backbone stands in for it).
  * The system's own machinery stays in charge: hyperparameters come from this module's search space
    (search.py + six-metric rank-sum), NOT from the library's Auto* single-loss search; rolling backtest,
    elimination, ranking and registry are unchanged.
  * Contract rows -> NeuralForecast long format (unique_id, ds, y, exogenous columns).
  * Chronological validation tail inside every training window with early stopping and best-checkpoint restore
    (C-2); the train-vs-validation loss gap is recorded in `_fitted_model["early_stopping"]`.
  * NeuralForecast models have a fixed forecast length h chosen at construction. Any requested horizon is served:
    a shorter one is truncated, a longer one is produced by rolling forward in chunks of h (C-3).
  * Missing calendar days are never silently invented: `gap_policy` = "error" (default) | "zero" | "ffill".
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from .base import AlgorithmModule
from .utils import future_dates, future_known_datasets, exogenous_frame, next_period_after
from ._vendor import ensure_vendored_neuralforecast
from ._frames import DailyFrameMixin, CALENDAR_COLS, calendar_features


def _best_state_callback():
    """
    Lightning callback keeping the best-validation weights ON THE MODEL OBJECT (`pl_module._stc_best`). NeuralForecast
    deep-copies trainer callbacks, so state held on the callback instance itself is lost; state on the module survives.
    """
    from pytorch_lightning.callbacks import Callback

    class BestState(Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            v = trainer.callback_metrics.get("ptl/val_loss")
            if v is None:
                return
            v = float(v)
            best = getattr(pl_module, "_stc_best", None)
            if best is None or v < best["loss"]:
                import copy
                pl_module._stc_best = {"loss": v, "step": int(trainer.global_step),
                                       "state": copy.deepcopy({k: t.detach().cpu() for k, t in pl_module.state_dict().items()})}

    return BestState()


def mlp_units(width: int, layers: int, n_stacks: int) -> list[list[int]]:
    """NeuralForecast's per-stack MLP spec: `layers` hidden layers of `width` units in each of `n_stacks` stacks."""
    return [[int(width)] * int(layers) for _ in range(int(n_stacks))]


class NeuralForecastModule(DailyFrameMixin, AlgorithmModule):
    """Base for the NeuralForecast-backed algorithms. A subclass sets `name`, `nf_model_name`, its
    model-specific search space and (optionally) eligibility / required-observation rules."""

    name = "nf_base"
    algorithm_version = "nf3.2.2-1"
    nf_model_name: str = ""
    default_n_lags = 14
    default_max_steps = 200
    #: PRD §3.2: N >= lookback + horizon + 500 (RNN/LSTM/GRU/N-BEATS/N-HiTS); the context family additionally
    #: has an absolute floor of 1000 (set by those subclasses).
    context_family_floor = 0
    supports_hist_exog = True
    supports_futr_exog = True
    default_use_calendar = False
    default_gradient_clip = None
    #: model-specific hyperparameters forwarded verbatim to the NeuralForecast model constructor
    forwarded_hp: tuple = ()
    #: model-specific search space (merged over the common space below)
    model_search_space: dict = {}
    is_simplified_stand_in = False

    def __init__(self, hyperparameters: dict | None = None):
        super().__init__(hyperparameters)
        hp = self.hyperparameters
        self.n_lags = int(hp.get("n_lags", self.default_n_lags))
        self.horizon = int(hp.get("horizon", 4))  # h the network is built with; infer() serves any horizon
        self._nf = None
        self._fk_names: list[str] = []      # future-known datasets used (structure-inferred, sorted); positional columns fk_0..
        self._fk_cols: list[str] = []
        self._exog_cols: list[str] = []
        self._use_calendar = bool(hp.get("use_calendar", self.default_use_calendar))

    # ---- PRD contract / eligibility ---------------------------------------------------------------------------
    def describe_contract(self) -> dict:
        return {"requires_endogenous": True, "requires_exogenous": False, "extra_columns": [],
                "regular_daily_index": True, "gap_policy": self.hyperparameters.get("gap_policy", "error"),
                "future_known_covariates": CALENDAR_COLS if self._use_calendar else []}

    def required_observations(self, n_exog: int = 0, horizon: int | None = None) -> int:
        formula = self.n_lags + int(horizon if horizon is not None else self.horizon) + 500
        return max(self.context_family_floor, formula) if self.context_family_floor else formula

    def hyperparameter_search_space(self) -> dict:
        space = {
            "n_lags": {"type": "int", "low": 7, "high": 60},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "batch_size": {"type": "choice", "choices": [16, 32, 64]},
        }
        space.update(self.model_search_space)
        return space

    def _with_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._use_calendar:
            return df
        f = calendar_features(df["ds"]).reset_index(drop=True)
        return pd.concat([df.reset_index(drop=True), f], axis=1)

    # ---- future-known covariates (inferred from structure, decision on G-28/G-29) -----------------------------------------
    fk_required = False   # TFT / TiDE: eligibility depends on them

    def _fk_names_for(self, segment_df: pd.DataFrame) -> list[str]:
        if not (self.supports_futr_exog and bool(self.hyperparameters.get("use_future_known", True))):
            return []
        # only series with enough future values for one forecast chunk are usable (a shorter one could not be fed at inference)
        return future_known_datasets(segment_df, min_future=self.horizon)

    @staticmethod
    def _fk_wide(segment_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        """Date x fk_i frame of the future-known series, including the dates AFTER the last observation."""
        ex = exogenous_frame(segment_df)
        missing = [n for n in names if n not in ex.columns]
        if missing:
            raise ValueError(f"future-known series {missing} used in training are absent from the supplied rows")
        wide = ex[list(names)].copy()
        wide.columns = [f"fk_{i}" for i in range(len(names))]
        return wide

    def _frame(self, segment_df: pd.DataFrame, series_id: str | None = None, exog_cols: list | None = None,
               fk_names: list | None = None) -> pd.DataFrame:
        names = self._fk_names if fk_names is None else fk_names
        sub = segment_df
        if names:   # a future-known series is a covariate over the horizon, never also a historical exogenous column
            sub = segment_df[~((segment_df["series_role"] == "exogenous") & segment_df["dataset"].isin(names))]
        df = super()._frame(sub, series_id=series_id, exog_cols=exog_cols)
        if names:
            wide = self._fk_wide(segment_df, names)
            idx = pd.DatetimeIndex(df["ds"])
            for c in wide.columns:
                df[c] = self._fill(wide[c].astype(float), idx, f"future-known series {c}").to_numpy()
            if df[list(wide.columns)].isna().any().any():
                raise ValueError("future-known series contain gaps within the observed range (see gap_policy)")
        return df

    def _futr(self, series_id: str, last_ds, h: int, fk_wide: pd.DataFrame | None = None):
        if not (self._use_calendar or self._fk_cols):
            return None
        ds = future_dates(pd.Timestamp(last_ds), h, freq=self.frequency)
        out = pd.DataFrame({"unique_id": series_id, "ds": ds})
        if self._use_calendar:
            out = pd.concat([out, calendar_features(ds).reset_index(drop=True)], axis=1)
        if self._fk_cols:
            vals = fk_wide.reindex(ds) if fk_wide is not None else None
            if vals is None or vals[self._fk_cols].isna().any().any():
                have = 0 if vals is None else int(vals[self._fk_cols].notna().all(axis=1).sum())
                raise ValueError(f"future-known covariates are available for {have} of the {h} periods after "
                                 f"{pd.Timestamp(last_ds).date()}; this model cannot forecast further without them")
            for c in self._fk_cols:
                out[c] = vals[c].to_numpy(dtype=float)
        return out

    def _infer_fk_names(self, segment_df: pd.DataFrame) -> list[str]:
        if not self._fk_cols:
            return []
        missing = [n for n in self._fk_names if n not in set(segment_df.loc[segment_df["series_role"] == "exogenous", "dataset"])]
        if missing:
            raise ValueError(f"future-known series {missing} used in training are absent from the supplied rows")
        return list(self._fk_names)

    # ---- model construction ---------------------------------------------------------------------------------
    def _model_kwargs(self) -> dict:
        """Constructor arguments for the NeuralForecast model. Subclasses extend/override."""
        hp = self.hyperparameters
        kw = {name: hp[name] for name in self.forwarded_hp if name in hp}
        if kw.get("encoder_n_layers") == 1 and kw.get("encoder_dropout"):
            kw["encoder_dropout"] = 0.0  # inter-layer dropout is inert (and warns) with a single layer
        return kw

    def _common_kwargs(self, early_stopping: bool, callbacks: list) -> dict:
        hp = self.hyperparameters
        kw = dict(
            h=self.horizon, input_size=self.n_lags,
            max_steps=int(hp.get("max_steps", self.default_max_steps)),
            learning_rate=float(hp.get("learning_rate", 1e-3)),
            batch_size=int(hp.get("batch_size", 32)),
            scaler_type=hp.get("scaler_type", "robust"),  # per-window scaling, fit on the input window only (C-5)
            random_seed=int(hp.get("random_seed", 1)),
            early_stop_patience_steps=int(hp.get("early_stop_patience_steps", 8)) if early_stopping else -1,
            val_check_steps=int(hp.get("val_check_steps", 20)),
            accelerator="cpu", devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
            enable_checkpointing=False, callbacks=callbacks or None,  # best weights are kept by _best_state_callback, no files
        )
        clip = hp.get("gradient_clip_val", self.default_gradient_clip)
        if clip:
            kw["gradient_clip_val"] = float(clip)
        if callbacks == []:
            kw.pop("callbacks")
        return {k: v for k, v in kw.items() if v is not None}

    def _build_model(self, exog_cols: list[str], early_stopping: bool, callbacks: list):
        nfm = ensure_vendored_neuralforecast().models
        cls = getattr(nfm, self.nf_model_name)
        kw = self._common_kwargs(early_stopping, callbacks)
        kw.update(self._model_kwargs())
        if exog_cols and self.supports_hist_exog:
            kw["hist_exog_list"] = list(exog_cols)
        futr = list(self._fk_cols) + (list(CALENDAR_COLS) if self._use_calendar else [])
        if futr and self.supports_futr_exog:
            kw["futr_exog_list"] = futr
        if self._static_cols:
            kw["stat_exog_list"] = list(self._static_cols)
        return cls(**kw)

    # ---- train ----------------------------------------------------------------------------------------------
    def _split_val(self, n: int) -> int:
        hp = self.hyperparameters
        if int(hp.get("early_stop_patience_steps", 8)) <= 0:
            return 0
        val = int(max(2 * self.horizon, float(hp.get("validation_fraction", 0.15)) * n))
        return val if n - val >= self.n_lags + self.horizon + 100 else 0

    def _training_frame(self, segment_df: pd.DataFrame):
        """(frame handed to NeuralForecast, exogenous column names, number of time steps). Multivariate modules override."""
        self._fk_names = self._fk_names_for(segment_df)
        self._fk_cols = [f"fk_{i}" for i in range(len(self._fk_names))]
        frame = self._frame(segment_df, fk_names=self._fk_names)
        cols = [c for c in frame.columns if c not in ("unique_id", "ds", "y") and not c.startswith("fk_")]
        return self._with_calendar(frame), cols, len(frame)

    def train(self, segment_df: pd.DataFrame) -> None:
        frame_m, self._exog_cols, n_time = self._training_frame(segment_df)
        self._fit(frame_m, self._exog_cols, n_time)

    _static_cols: list = []          # static (per-series) covariate columns; set by the pooled subclass
    valid_loss_label = "MAE, original target units"

    def _fit(self, frame_m: pd.DataFrame, exog_cols: list, n_time: int, static_df: pd.DataFrame | None = None) -> None:
        nf_mod = ensure_vendored_neuralforecast()
        val_size = self._split_val(n_time)
        callbacks = [_best_state_callback()] if val_size else []
        model = self._build_model(exog_cols, early_stopping=bool(val_size), callbacks=callbacks)
        self._nf = nf_mod.NeuralForecast(models=[model], freq=self.frequency)
        if static_df is not None:
            self._nf.fit(df=frame_m, static_df=static_df, val_size=val_size)
        else:
            self._nf.fit(df=frame_m, val_size=val_size)
        m = self._nf.models[0]
        best = getattr(m, "_stc_best", None)
        restored = False
        if val_size and best is not None:
            m.load_state_dict(best["state"])  # best-checkpoint restore (C-2)
            restored = True
        vt, tt = list(getattr(m, "valid_trajectories", [])), list(getattr(m, "train_trajectories", []))
        es = {"used": bool(val_size), "validation_rows": int(val_size),
              "max_steps": int(self.hyperparameters.get("max_steps", self.default_max_steps)),
              "stopped_step": int(max([s_ for s_, _ in vt] + [s_ for s_, _ in tt] + [0])),
              "restored_best_checkpoint": restored}
        if best is not None:
            es.update({"best_step": best["step"], "best_validation_loss": best["loss"]})
        # NOTE: NeuralForecast logs the training loss on SCALED targets and the validation loss in original units, so
        # their difference is not an overfitting measure. The comparable train-vs-validation gap (G-10) is computed
        # by the backtest from the six metrics on the training tail vs the held-out fold.
        if tt:
            es["final_train_loss_scaled"] = float(tt[-1][1])
        es["validation_loss_units"] = self.valid_loss_label
        self._fitted_model = {"n_lags": self.n_lags, "horizon": self.horizon, "hyperparameters": self.hyperparameters,
                              "n_parameters": self.n_parameters(), "early_stopping": es,
                              "exogenous_columns": self._exog_cols, "calendar_covariates": self._use_calendar,
                              "future_known_series": list(self._fk_names)}
        self._store_insample_residuals(frame_m)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._nf.models[0].parameters())) if self._nf is not None else 0

    def network(self):
        """The underlying torch module (for architecture checks in tests)."""
        return self._nf.models[0] if self._nf is not None else None

    def _store_insample_residuals(self, frame_m: pd.DataFrame) -> None:
        """One-step-ahead in-sample residuals (first step of every window), chronological, for diagnose()."""
        try:
            ins = self._nf.predict_insample(step_size=1)
            col = self._point_column(ins)
            first = ins[ins["ds"] == ins["cutoff"].map(lambda c: next_period_after(c, self.frequency))].sort_values("ds")
            self._resid = (first["y"] - first[col]).to_numpy(dtype=float)
        except Exception:
            self._resid = None

    @staticmethod
    def _point_column(out: pd.DataFrame) -> str:
        cols = [c for c in out.columns if c not in ("unique_id", "ds", "cutoff", "y")]
        med = [c for c in cols if c.endswith("-median")]
        return med[0] if med else cols[0]

    def _predict_extra(self, sid: str, segment_df: pd.DataFrame) -> dict:
        """Extra keyword arguments for NeuralForecast.predict (static covariates in the pooled subclass)."""
        return {}

    def _state_extra(self) -> dict:
        return {"fk_names": list(self._fk_names), "fk_cols": list(self._fk_cols)}

    def _load_extra(self, st: dict) -> None:
        self._fk_names, self._fk_cols = list(st.get("fk_names", [])), list(st.get("fk_cols", []))

    # ---- infer ----------------------------------------------------------------------------------------------
    def infer(self, segment_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("load() or train() must run before infer()")
        fk_names = self._infer_fk_names(segment_df)
        frame = self._frame(segment_df, exog_cols=self._exog_cols, fk_names=fk_names)
        fkw = self._fk_wide(segment_df, fk_names) if fk_names else None
        sid = frame["unique_id"].iloc[0]
        target_dates = future_dates(frame["ds"].max(), horizon, freq=self.frequency)
        preds: list[float] = []
        hist = frame
        while len(preds) < horizon:
            last = hist["ds"].max()
            out = self._nf.predict(df=self._with_calendar(hist), futr_df=self._futr(sid, last, self.horizon, fkw),
                                   **self._predict_extra(sid, segment_df))
            yhat = out[self._point_column(out)].to_numpy(dtype=float)[: self.horizon]
            preds.extend(float(v) for v in yhat)
            if len(preds) >= horizon:
                break
            nds = future_dates(last, len(yhat), freq=self.frequency)
            nxt = pd.DataFrame({"unique_id": sid, "ds": nds, "y": yhat})
            for c in self._exog_cols:
                nxt[c] = hist[c].iloc[-1]  # hist exogenous carried forward when rolling beyond h
            if self._fk_cols:
                vals = fkw.reindex(nds)
                if vals[self._fk_cols].isna().any().any():
                    raise ValueError(f"requested horizon {horizon} exceeds the future-known covariates supplied "
                                     f"(through {fkw.dropna().index.max().date()}); this model cannot forecast further without them")
                for c in self._fk_cols:
                    nxt[c] = vals[c].to_numpy(dtype=float)
            hist = pd.concat([hist, nxt], ignore_index=True)
        return pd.DataFrame({"date": target_dates, "forecast": preds[:horizon]})

    # ---- save / load ----------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = path.parent / "nf_model"
        self._nf.save(str(d), save_dataset=False, overwrite=True)
        with open(path, "wb") as f:
            pickle.dump({"hyperparameters": self.hyperparameters, "exog_cols": self._exog_cols, "horizon": self.horizon,
                         "n_lags": self.n_lags, "use_calendar": self._use_calendar, "nf_dir": "nf_model",
                         "nf_model_name": self.nf_model_name, "fitted": self._fitted_model, "extra": self._state_extra()}, f)

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.hyperparameters = st["hyperparameters"]
        self._exog_cols, self.horizon, self.n_lags = st["exog_cols"], st["horizon"], st["n_lags"]
        self._use_calendar = st["use_calendar"]
        self._fitted_model = st.get("fitted")
        self._load_extra(st.get("extra") or {})
        self._nf = ensure_vendored_neuralforecast().NeuralForecast.load(str(path.parent / st["nf_dir"]))
        for m in self._nf.models:
            m.trainer_kwargs["enable_checkpointing"] = False
            m.trainer_kwargs["logger"] = False
