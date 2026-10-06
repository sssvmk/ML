"""Template for an algorithm module.

An algorithm module (one file per algorithm, named after the algorithm) only declares WHAT is specific to the
algorithm; the shared `run()` protocol below guarantees every algorithm is evaluated identically, which is what
makes the final comparison fair:

    3  encoding / normalisation        -> build_preprocessor()
    5  algorithm choice                -> the module itself (family, complexity_rank, description)
    7  assumption validation           -> check_assumptions()
    8  loss, metrics, hyper-parameters -> loss_function, search_space(), Optuna TPE search, trial table
    9  metrics + plots                 -> run(): ROC/PR, calibration, confusion, curves, importance, slices
    10 inference scaffolding           -> a ModelBundle (raw rows in -> probability/label out) is saved
    11 MLOps                           -> tracker logs params / metrics / artifacts, config, data hash, seed
"""
from __future__ import annotations

import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..assumptions import AssumptionContext, AssumptionResult
from ..bundle import ModelBundle, compute_reference_stats
from ..features import FeatureEngineer, FeatureSet
from ..metrics import choose_threshold, compute_metrics, primary_value, slice_metrics
from ..plots import (plot_bar, plot_calibration, plot_confusion, plot_curve, plot_optuna_history, plot_roc_pr)
from ..preprocessing import build_preprocessor, feature_names
from ..schema import Schema
from ..tracking import Tracker
from ..utils import dump_json


# ----------------------------------------------------------------------------------------------------------
@dataclass
class ExperimentData:
    X_train: pd.DataFrame            # engineered features (+ meta columns), TRAIN
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    schema: Schema
    feature_engineer: FeatureEngineer
    cfg: dict
    run_meta: dict = field(default_factory=dict)   # data hash, git commit, seeds ... (lineage)

    @property
    def feature_set(self) -> FeatureSet:
        return self.feature_engineer.feature_set


@dataclass
class AlgorithmResult:
    name: str
    display_name: str
    family: str
    status: str = "ok"                      # ok | skipped | failed
    message: str = ""
    description: str = ""
    loss_function: str = ""
    optimisation_notes: str = ""
    complexity_rank: int = 0
    primary_metric: str = ""
    best_params: dict = field(default_factory=dict)
    hyperparameter_docs: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    train_primary: float = float("nan")
    val_primary: float = float("nan")
    overfit_gap: float = float("nan")
    cv: dict = field(default_factory=dict)
    diagnosis: dict = field(default_factory=dict)
    importance: dict = field(default_factory=dict)
    slices: dict = field(default_factory=dict)
    latency_ms_per_1000_rows: float = float("nan")
    fit_seconds: float = float("nan")
    total_seconds: float = float("nan")
    n_features_encoded: int = 0
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------------------------
class AlgorithmModule(ABC):
    # ---- identity / documentation (used by README generation and by the LLM agent) -------------------------
    name: str = ""
    display_name: str = ""
    family: str = ""                      # baseline | linear | bagging | boosting | deep
    description: str = ""
    complexity_rank: int = 0              # lower = simpler; used only as a tie-breaker
    loss_function: str = ""
    optimisation_notes: str = ""
    # ---- behaviour flags ----------------------------------------------------------------------------------------
    searchable: bool = True
    uses_early_stopping: bool = False
    iterative: bool = False               # exposes a per-iteration/epoch training curve
    nominal_encoding: str = "onehot"
    scale_inputs: bool = True

    # ---- to implement ---------------------------------------------------------------------------------------------
    @abstractmethod
    def build_estimator(self, params: dict, seed: int):
        """Return an unfitted object with fit(X, y) and predict_proba(X)."""

    @abstractmethod
    def default_params(self) -> dict: ...

    def search_space(self, trial) -> dict:  # noqa: D401 - optional for non-searchable modules
        return self.default_params()

    def hyperparameter_docs(self) -> dict:
        return {}

    def check_assumptions(self, ctx: AssumptionContext) -> list[AssumptionResult]:
        return []

    # ---- optional hooks -------------------------------------------------------------------------------------------
    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def build_preprocessor(self, fs: FeatureSet):
        return build_preprocessor(fs, nominal=self.nominal_encoding, scale=self.scale_inputs)

    def fit_estimator(self, est, X, y, X_es=None, y_es=None, trial=None):
        est.fit(X, y)
        return est

    def training_curve(self, est) -> pd.DataFrame | None:
        return None

    def native_importance(self, est, names: list[str]) -> pd.Series | None:
        return None

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _score(self, cfg: dict, y, p) -> float:
        metric = cfg["metric"]["primary"]
        th = 0.5
        if metric in {"f1", "accuracy"}:
            th = choose_threshold(y, p, cfg["metric"]["threshold_rule"], cfg["metric"]["fixed_threshold"])
        return primary_value(metric, y, p, th)

    def _fit_with_optional_es(self, params, X, y, seed, trial=None):
        est = self.build_estimator(params, seed)
        if self.uses_early_stopping:
            X1, Xes, y1, yes = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
            return self.fit_estimator(est, X1, y1, Xes, yes, trial=trial)
        return self.fit_estimator(est, X, y, trial=trial)

    def design_summary(self) -> dict:
        return {"name": self.name, "display_name": self.display_name, "family": self.family,
                "description": self.description, "loss_function": self.loss_function,
                "optimisation_notes": self.optimisation_notes, "default_params": self.default_params(),
                "hyperparameter_docs": self.hyperparameter_docs(), "complexity_rank": self.complexity_rank,
                "nominal_encoding": self.nominal_encoding, "inputs_scaled": self.scale_inputs}

    # ---- the shared protocol ---------------------------------------------------------------------------------------
    def run(self, data: ExperimentData, tracker: Tracker, out_dir: Path) -> AlgorithmResult:
        cfg = data.cfg
        seed = cfg["project"]["seed"]
        res = AlgorithmResult(name=self.name, display_name=self.display_name, family=self.family,
                              description=self.description, loss_function=self.loss_function,
                              optimisation_notes=self.optimisation_notes, complexity_rank=self.complexity_rank,
                              primary_metric=cfg["metric"]["primary"], hyperparameter_docs=self.hyperparameter_docs())
        ok, why = self.is_available()
        if not ok:
            res.status, res.message = "skipped", why
            return res
        algo_dir = Path(out_dir) / "algorithms" / self.name
        plots = algo_dir / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        t_total = time.perf_counter()
        try:
            with tracker.run(self.name, nested=True, tags={"phase": "candidate", "family": self.family}):
                self._run_inner(data, tracker, algo_dir, plots, res)
        except Exception as exc:  # one failing algorithm must not kill the whole experiment
            res.status, res.message = "failed", f"{type(exc).__name__}: {exc}"
            (algo_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        res.total_seconds = time.perf_counter() - t_total
        dump_json(res.to_dict(), algo_dir / "result.json")
        return res

    # ................................................................................................................
    def _run_inner(self, data: ExperimentData, tracker: Tracker, algo_dir: Path, plots: Path, res: AlgorithmResult):
        cfg, seed = data.cfg, data.cfg["project"]["seed"]
        fs, metric = data.feature_set, cfg["metric"]["primary"]
        cols = fs.model_columns
        tracker.log_params({"algorithm": self.name, "family": self.family, "loss_function": self.loss_function,
                            "primary_metric": metric, "seed": seed, "nominal_encoding": self.nominal_encoding,
                            "inputs_scaled": self.scale_inputs, **{f"meta.{k}": v for k, v in data.run_meta.items()}})

        # step 3 -- encode / normalise (fit on TRAIN only) -------------------------------------------------------
        pre = self.build_preprocessor(fs)
        Xtr = pre.fit_transform(data.X_train[cols])
        Xva = pre.transform(data.X_val[cols])
        names = feature_names(pre)
        res.n_features_encoded = len(names)

        # step 7 -- assumption validation --------------------------------------------------------------------------
        ctx = AssumptionContext(X_df=data.X_train, y=data.y_train, Xt=Xtr, names=names, feature_set=fs,
                                X_val_df=data.X_val, seed=seed)
        checks = [c for c in self.check_assumptions(ctx) if c is not None]
        res.assumptions = [c.to_dict() for c in checks]
        dump_json(res.assumptions, algo_dir / "assumption_checks.json")
        for c in checks:
            tracker.log_metrics({f"assumption.{c.name}.passed": float(c.passed)})
        blocking = [c for c in checks if c.blocking]
        if blocking:
            raise RuntimeError("Blocking assumption violated: " + "; ".join(f"{c.name} ({c.detail})" for c in blocking))

        # step 8 -- hyper-parameter search ---------------------------------------------------------------------------
        best_params, search_info, trials_df = self._search(Xtr, data.y_train, Xva, data.y_val, cfg, seed)
        res.best_params, res.search = best_params, search_info
        tracker.log_params({f"best.{k}": v for k, v in best_params.items()})
        if trials_df is not None and len(trials_df):
            trials_df.to_csv(algo_dir / "search_trials.csv", index=False)
            plots_hist = plot_optuna_history(trials_df, plots / "search_history.png", self.display_name)
            res.artifacts["search_history_plot"] = plots_hist
            res.artifacts["search_trials_csv"] = str(algo_dir / "search_trials.csv")
            for _, row in trials_df.iterrows():
                with tracker.run(f"trial_{int(row['number'])}", nested=True, tags={"phase": "search"}):
                    tracker.log_params({k[7:]: row[k] for k in trials_df.columns if k.startswith("params_")})
                    tracker.log_metrics({"val_primary": row["value"]})

        # final fit on the FULL training set ------------------------------------------------------------------------
        t0 = time.perf_counter()
        est = self._fit_with_optional_es(best_params, Xtr, data.y_train, seed)
        res.fit_seconds = time.perf_counter() - t0

        p_val = est.predict_proba(Xva)[:, 1]
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(data.y_train), size=min(50_000, len(data.y_train)), replace=False)
        p_tr = est.predict_proba(Xtr[tr_idx])[:, 1]
        thr = choose_threshold(data.y_val, p_val, cfg["metric"]["threshold_rule"], cfg["metric"]["fixed_threshold"])
        res.val_metrics = compute_metrics(data.y_val, p_val, thr)
        res.train_metrics = compute_metrics(data.y_train[tr_idx], p_tr, thr)
        key = "log_loss" if metric == "neg_log_loss" else metric
        sign = -1.0 if metric == "neg_log_loss" else 1.0
        res.val_primary, res.train_primary = sign * res.val_metrics[key], sign * res.train_metrics[key]
        res.overfit_gap = res.train_primary - res.val_primary
        tracker.log_metrics({f"val.{k}": v for k, v in res.val_metrics.items()})
        tracker.log_metrics({f"train.{k}": v for k, v in res.train_metrics.items()})
        tracker.log_metrics({"overfit_gap": res.overfit_gap, "fit_seconds": res.fit_seconds})

        # step 9 -- curves / diagnostics -------------------------------------------------------------------------------
        curve = self.training_curve(est)
        if curve is None and self.searchable:
            curve = self._size_curve(best_params, Xtr, data.y_train, Xva, data.y_val, cfg, seed, est, res)
            curve_x, curve_cols, xlog, ylabel = "train_rows", ["train_primary", "val_primary"], True, metric
        else:
            curve_x, curve_cols, xlog, ylabel = "iteration", ["train_loss", "val_loss"], False, "log loss"
        if curve is not None and len(curve):
            curve.to_csv(algo_dir / "training_curve.csv", index=False)
            res.artifacts["training_curve_csv"] = str(algo_dir / "training_curve.csv")
            res.artifacts["training_curve_plot"] = plot_curve(curve, curve_x, curve_cols, plots / "training_curve.png",
                                                             f"{self.display_name}: learning curve", ylabel, xlog,
                                                             hline=cfg["metric"].get("target_value") if curve_x == "train_rows" else None)
        res.diagnosis = self._diagnose(res, curve, cfg)

        pred = (p_val >= thr).astype(int)
        res.artifacts["roc_pr_plot"] = plot_roc_pr(data.y_val, p_val, plots / "roc_pr.png", f"({self.display_name})")
        res.artifacts["calibration_plot"] = plot_calibration(data.y_val, p_val, plots / "calibration.png", f"({self.display_name})")
        res.artifacts["confusion_plot"] = plot_confusion(data.y_val, pred, plots / "confusion.png", f"(thr={thr:.2f})")
        np.savez_compressed(algo_dir / "val_predictions.npz", p=p_val, y=data.y_val)
        res.artifacts["val_predictions"] = str(algo_dir / "val_predictions.npz")

        res.importance = self._importance(pre, est, data, names, cfg, seed, plots, res)
        res.slices = self._slices(data, p_val, thr, cfg)
        if cfg["cv_check"]["enabled"] and self.searchable:
            res.cv = self._cv_check(best_params, data, pre, cfg, seed)
            tracker.log_metrics({"cv.mean": res.cv.get("mean", float("nan")), "cv.std": res.cv.get("std", float("nan"))})

        # latency (engineered rows -> encoded -> predict; excludes feature engineering, measured at serving) --------
        n_lat = min(5000, len(data.X_val))
        t0 = time.perf_counter()
        est.predict_proba(pre.transform(data.X_val[cols].iloc[:n_lat]))
        res.latency_ms_per_1000_rows = (time.perf_counter() - t0) / n_lat * 1000 * 1000

        # step 10 -- inference bundle --------------------------------------------------------------------------------------
        metadata = {"algorithm": self.name, "best_params": best_params, "val_metrics": res.val_metrics,
                    "primary_metric": metric, "seed": seed, **data.run_meta}
        bundle = ModelBundle(self.name, data.schema, data.feature_engineer, pre, est, thr, metadata,
                             compute_reference_stats(data.X_train, fs))
        bundle.save(algo_dir / "bundle")
        res.artifacts["bundle_dir"] = str(algo_dir / "bundle")

        # step 11 -- track ---------------------------------------------------------------------------------------------------
        for k in ("roc_pr_plot", "calibration_plot", "confusion_plot", "training_curve_plot", "search_history_plot"):
            if k in res.artifacts:
                tracker.log_artifact(res.artifacts[k], "plots")
        dump_json(res.to_dict(), algo_dir / "result.json")
        tracker.log_artifact(algo_dir / "result.json")
        tracker.log_artifact(algo_dir / "assumption_checks.json")

    # ................................................................................................................
    def _search(self, Xtr, ytr, Xva, yva, cfg, seed):
        scfg = cfg["search"]
        if not self.searchable or scfg["n_trials"] <= 0:
            return self.default_params(), {"method": "none (fixed defaults)", "n_trials": 0}, None
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n = min(scfg["train_subsample"], len(ytr))
        if n < len(ytr):
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, stratify=ytr, random_state=seed)
        else:
            Xs, ys = Xtr, ytr
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True) if scfg["sampler"] == "tpe" else optuna.samplers.RandomSampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2) if scfg["pruner"] == "median" else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        # enqueue the defaults so the search can never end up worse than the documented starting point
        study.enqueue_trial(self.default_params())

        def objective(trial):
            params = self.search_space(trial)
            est = self._fit_with_optional_es(params, Xs, ys, seed, trial=trial)
            return self._score(cfg, yva, est.predict_proba(Xva)[:, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=scfg["n_trials"], timeout=scfg["timeout_seconds"], catch=(Exception,))
        done = [t for t in study.trials if t.value is not None]
        if not done:
            raise RuntimeError("hyper-parameter search produced no successful trial")
        trials_df = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
        trials_df = trials_df.rename(columns={c: c.replace("params_", "params_") for c in trials_df.columns})
        try:
            importances = optuna.importance.get_param_importances(study) if len(done) >= 4 else {}
        except Exception:
            importances = {}
        info = {"method": f"Optuna {scfg['sampler'].upper()} (Bayesian) + {scfg['pruner']} pruner",
                "n_trials_requested": scfg["n_trials"], "n_trials_completed": len(done),
                "n_trials_failed_or_pruned": len(study.trials) - len(done), "best_value": study.best_value,
                "search_rows": int(len(ys)), "hyperparameter_importance": importances,
                "worst_value": min(t.value for t in done)}
        best = {**self.default_params(), **study.best_params}
        return best, info, trials_df

    def _size_curve(self, params, Xtr, ytr, Xva, yva, cfg, seed, final_est, res):
        rows = []
        for f in cfg["diagnostics"]["learning_curve_fractions"]:
            n = int(len(ytr) * f)
            if n < 200:
                continue
            if f >= 1.0:
                est, Xs, ys = final_est, Xtr, ytr
            else:
                Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, stratify=ytr, random_state=seed)
                est = self._fit_with_optional_es(params, Xs, ys, seed)
            sub = np.random.default_rng(seed).choice(len(ys), size=min(20000, len(ys)), replace=False)
            rows.append({"train_rows": n, "train_primary": self._score(cfg, ys[sub], est.predict_proba(Xs[sub])[:, 1]),
                         "val_primary": self._score(cfg, yva, est.predict_proba(Xva)[:, 1])})
        return pd.DataFrame(rows)

    def _diagnose(self, res: AlgorithmResult, curve, cfg) -> dict:
        gap = res.overfit_gap
        max_gap = cfg["selection"]["max_overfit_gap"]
        target = cfg["metric"].get("target_value")
        notes, verdict = [], "healthy"
        if gap > max_gap:
            verdict = "overfitting"
            notes.append(f"train-validation gap {gap:.4f} exceeds {max_gap}: add regularisation, reduce capacity or add data.")
        elif target is not None and res.val_primary < target and gap < max_gap / 3:
            verdict = "underfitting_suspected"
            notes.append(f"validation {res.val_primary:.4f} is below the target {target} while train≈validation: increase capacity / features.")
        elif target is None:
            notes.append("no business target configured: cannot judge under-fitting against a goal (see TODO).")
        if curve is not None and len(curve) > 5 and {"val_primary"}.issubset(curve.columns):
            last, prev = curve["val_primary"].iloc[-1], curve["val_primary"].iloc[-2]
            slope = last - prev
            notes.append(f"validation metric changes by {slope:+.4f} over the last data doubling -> "
                         + ("more data likely still helps." if slope > 0.002 else "more data will help little."))
        if curve is not None and "val_loss" in curve.columns and len(curve) > 10:
            best_it = int(curve["val_loss"].idxmin())
            if best_it < 0.8 * (len(curve) - 1):
                notes.append(f"validation loss bottomed at iteration {best_it} of {len(curve) - 1}: early stopping was necessary.")
        return {"verdict": verdict, "gap": gap, "notes": notes}

    def _importance(self, pre, est, data, names, cfg, seed, plots, res):
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score

        cols = data.feature_set.model_columns
        n = min(cfg["diagnostics"]["permutation_importance_rows"], len(data.y_val))
        idx = np.random.default_rng(seed).choice(len(data.y_val), size=n, replace=False)
        Xs, ys = data.X_val[cols].iloc[idx].reset_index(drop=True), data.y_val[idx]

        class _Raw:
            """engineered rows -> probabilities, so permutation happens on interpretable (pre-encoding) columns"""

            def fit(self_inner, X=None, y=None):  # sklearn requires the attribute; the model is already fitted
                return self_inner

            def predict_proba(self_inner, X):
                return est.predict_proba(pre.transform(X))

        scorer = lambda m, X, y: roc_auc_score(y, m.predict_proba(X)[:, 1])  # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = permutation_importance(_Raw(), Xs, ys, scoring=scorer, n_repeats=3, random_state=seed, n_jobs=1)
        s = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
        res.artifacts["importance_plot"] = plot_bar(s.head(20), plots / "permutation_importance.png",
                                                    f"{self.display_name}: permutation importance (AUC drop)", "AUC drop")
        out = {"permutation_auc_drop": {k: float(v) for k, v in s.head(15).items()}}
        native = self.native_importance(est, names)
        if native is not None:
            out["native"] = {k: float(v) for k, v in native.sort_values(ascending=False).head(15).items()}
        return out

    def _slices(self, data, p_val, thr, cfg) -> dict:
        return slice_metrics(data.X_val, data.y_val, p_val, thr, cfg["diagnostics"]["slice_columns"])

    def _cv_check(self, params, data, pre, cfg, seed) -> dict:
        ccfg = cfg["cv_check"]
        cols = data.feature_set.model_columns
        X, y = data.X_train[cols], data.y_train
        if len(y) > ccfg["subsample"]:
            X, _, y, _ = train_test_split(X, y, train_size=ccfg["subsample"], stratify=y, random_state=seed)
        scores = []
        skf = StratifiedKFold(n_splits=ccfg["folds"], shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            p_k = clone(pre)                                  # preprocessing re-fitted INSIDE each fold
            Xtr_k = p_k.fit_transform(X.iloc[tr])
            est = self._fit_with_optional_es(params, Xtr_k, y[tr], seed)
            scores.append(self._score(cfg, y[te], est.predict_proba(p_k.transform(X.iloc[te]))[:, 1]))
        return {"folds": ccfg["folds"], "rows": int(len(y)), "scores": scores, "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0}
