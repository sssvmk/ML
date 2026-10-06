"""Step 7: statistical / data assumption checks.

Each algorithm module composes the checks that matter for IT (e.g. linearity of the logit for logistic
regression, correct input scaling for a neural net). Every check returns an AssumptionResult so the findings
are logged, shown in the report and explained by the LLM agent.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


@dataclass
class AssumptionResult:
    name: str
    passed: bool
    severity: str            # "pass" | "warn" | "fail"
    value: float | str | None
    detail: str
    why_it_matters: str = ""
    mitigation: str = ""
    blocking: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class AssumptionContext:
    X_df: pd.DataFrame            # engineered TRAIN rows (pre-encoding)
    y: np.ndarray
    Xt: np.ndarray                # encoded TRAIN matrix
    names: list[str]              # encoded feature names
    feature_set: object
    X_val_df: pd.DataFrame | None = None
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def sample_idx(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.choice(len(self.y), size=min(n, len(self.y)), replace=False)


def _r(name, ok, value, detail, why="", mitigation="", warn_only=True, blocking=False):
    sev = "pass" if ok else ("warn" if warn_only else "fail")
    return AssumptionResult(name, bool(ok), sev, value if isinstance(value, str) or value is None else float(value),
                            detail, why, mitigation, blocking and not ok)


# ------------------------------------------------------------------------------------------------ shared
def check_finite_inputs(ctx: AssumptionContext, blocking: bool = True) -> AssumptionResult:
    bad = int((~np.isfinite(ctx.Xt)).sum())
    return _r("encoded_inputs_finite", bad == 0, bad, f"{bad} NaN/inf values in encoded training matrix",
              "NaN/inf silently break gradient methods and many solvers.", "Imputation inside the preprocessor.",
              warn_only=False, blocking=blocking)


def check_class_balance(ctx: AssumptionContext) -> AssumptionResult:
    rate = float(np.mean(ctx.y))
    ratio = max(rate, 1 - rate) / max(min(rate, 1 - rate), 1e-9)
    return _r("class_balance", ratio <= 3, ratio, f"positive rate {rate:.1%}, majority:minority = {ratio:.2f}:1",
              "Severe imbalance makes accuracy misleading and shifts calibrated probabilities.",
              "Use PR-AUC/F1, class weights or a tuned threshold.")


def check_duplicates(ctx: AssumptionContext) -> AssumptionResult:
    share = float(pd.util.hash_pandas_object(ctx.X_df[ctx.feature_set.model_columns], index=False).duplicated().mean())
    return _r("duplicate_rows", share < 0.01, share, f"{share:.2%} of training rows have a duplicate feature vector",
              "Duplicates make memorising models look better than they are and break CV independence.",
              "Duplicates are kept in the same split (group split).")


def check_single_feature_leakage(ctx: AssumptionContext) -> AssumptionResult:
    """A single raw feature that almost perfectly predicts the label usually means target leakage."""
    idx = ctx.sample_idx(30000)
    X, y = ctx.X_df.iloc[idx], ctx.y[idx]
    best, best_col = 0.5, ""
    for c in ctx.feature_set.model_columns:
        s = X[c]
        if not pd.api.types.is_numeric_dtype(s):
            enc = pd.Series(y).groupby(s.astype(str).to_numpy()).transform("mean").to_numpy()
        else:
            enc = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy()
        if np.nanstd(enc) == 0:
            continue
        auc = roc_auc_score(y, enc)
        auc = max(auc, 1 - auc)
        if auc > best:
            best, best_col = auc, c
    return _r("no_single_feature_leakage", best < 0.97, best, f"strongest single feature: '{best_col}' with AUC {best:.3f}",
              "A near-perfect single predictor suggests the label leaks into the features.",
              "Investigate/remove the feature if it would not exist at prediction time.", warn_only=False)


def check_train_validation_shift(ctx: AssumptionContext) -> AssumptionResult | None:
    if ctx.X_val_df is None:
        return None
    from .splitting import adversarial_validation

    cols = ctx.feature_set.model_columns
    auc = adversarial_validation(ctx.X_df[cols], ctx.X_val_df[cols], seed=ctx.seed)
    return _r("train_validation_same_distribution", auc < 0.6, auc, f"adversarial-validation AUC {auc:.3f} (0.5 = indistinguishable)",
              "Model selection on a shifted validation set would not predict test/production behaviour.",
              "Re-split / investigate drifted features.")


def check_sample_size(ctx: AssumptionContext, min_rows: int = 5000) -> AssumptionResult:
    n = len(ctx.y)
    return _r("sufficient_training_rows", n >= min_rows, n, f"{n} training rows", "Tiny samples give unstable hyper-parameter search.",
              "Collect more data or reduce model capacity.")


# ------------------------------------------------------------------------------------------------ logistic regression
def check_events_per_variable(ctx: AssumptionContext, threshold: float = 10) -> AssumptionResult:
    events = min(ctx.y.sum(), len(ctx.y) - ctx.y.sum())
    epv = events / max(ctx.Xt.shape[1], 1)
    return _r("events_per_variable>=10", epv >= threshold, epv, f"{int(events)} minority events / {ctx.Xt.shape[1]} predictors = {epv:.1f}",
              "Logistic regression coefficients overfit when events per predictor are low.", "Regularisation / fewer predictors.")


def check_multicollinearity(ctx: AssumptionContext, vif_warn: float = 10.0) -> AssumptionResult:
    keep = [i for i, n in enumerate(ctx.names) if not n.startswith("nom__")]
    idx = ctx.sample_idx(50000)
    Z = ctx.Xt[np.ix_(idx, keep)].astype(float)
    sd = Z.std(axis=0)
    Z = Z[:, sd > 0]
    names = [ctx.names[keep[i]] for i in np.where(sd > 0)[0]]
    corr = np.corrcoef(Z, rowvar=False)
    cond = float(np.linalg.cond(corr))
    vif = np.diag(np.linalg.pinv(corr))
    order = np.argsort(-vif)[:3]
    worst = ", ".join(f"{names[i]} (VIF {vif[i]:.0f})" for i in order)
    ok = float(vif.max()) <= vif_warn
    return _r("low_multicollinearity", ok, float(vif.max()), f"max VIF {vif.max():.1f}; condition number {cond:.1e}; worst: {worst}",
              "Collinear predictors make unpenalised coefficients unstable and uninterpretable.",
              "Elastic-net (L1+L2) penalty stabilises/zeroes redundant coefficients; predictions are unaffected.")


def _fit_lr(X, y, C=1e4, max_iter=300):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs").fit(X, y)


def check_linearity_of_logit(ctx: AssumptionContext, practical_gain: float = 0.002) -> AssumptionResult:
    """Box-Tidwell style: add x*ln(x) for each continuous predictor; if it improves fit materially the logit is not linear in x."""
    idx = ctx.sample_idx(30000)
    cont_cols = [i for i, n in enumerate(ctx.names) if n.startswith("cont__")]
    X, y = ctx.Xt[idx].astype(float), ctx.y[idx]
    base = _fit_lr(X, y)
    base_ll = log_loss(y, base.predict_proba(X)[:, 1])
    gains = {}
    for i in cont_cols:
        v = X[:, i]
        shifted = v - v.min() + 1.0
        aug = np.column_stack([X, shifted * np.log(shifted)])
        ll = log_loss(y, _fit_lr(aug, y).predict_proba(aug)[:, 1])
        gains[ctx.names[i]] = base_ll - ll
    if not gains:
        return _r("linearity_of_logit", True, 0.0, "no continuous predictors")
    worst = max(gains, key=gains.get)
    lr_stat = 2 * gains[worst] * len(y)
    p = float(stats.chi2.sf(max(lr_stat, 0), 1))
    nonlin = [k for k, v in gains.items() if v > practical_gain]
    return _r("linearity_of_logit", not nonlin, gains[worst],
              f"largest log-loss gain from a Box-Tidwell term: {gains[worst]:.4f} on '{worst}' (LR p={p:.2g}); non-linear: {nonlin or 'none'}",
              "Logistic regression assumes the log-odds are linear in each continuous predictor.",
              "log1p / log-distance features and age/distance bands are supplied as engineered features.")


def check_no_separation(ctx: AssumptionContext) -> AssumptionResult:
    idx = ctx.sample_idx(30000)
    m = _fit_lr(ctx.Xt[idx].astype(float), ctx.y[idx], C=1e6, max_iter=500)
    max_coef = float(np.abs(m.coef_).max())
    acc = float((m.predict(ctx.Xt[idx].astype(float)) == ctx.y[idx]).mean())
    ok = max_coef < 25 and acc < 0.9999
    return _r("no_complete_separation", ok, max_coef, f"max |coef| {max_coef:.1f}, train accuracy {acc:.4f}",
              "Perfect separation makes maximum-likelihood coefficients diverge.", "Penalised likelihood (always used here).")


def check_residual_independence(ctx: AssumptionContext) -> AssumptionResult:
    """Durbin-Watson on the logistic residuals in row order (~2 = no serial correlation)."""
    idx = np.sort(ctx.sample_idx(30000))
    X, y = ctx.Xt[idx].astype(float), ctx.y[idx]
    res = y - _fit_lr(X, y, C=1.0).predict_proba(X)[:, 1]
    dw = float(np.sum(np.diff(res) ** 2) / np.sum(res ** 2))
    return _r("independent_observations", 1.8 <= dw <= 2.2, dw, f"Durbin-Watson {dw:.3f} (2 = independent)",
              "Dependent rows (e.g. repeated passengers/flights) understate uncertainty and inflate CV scores.",
              "Group-aware splitting if a passenger/flight id becomes available.")


# ------------------------------------------------------------------------------------------------ neural nets
def check_scaled_inputs(ctx: AssumptionContext, tol: float = 0.25) -> AssumptionResult:
    cols = [i for i, n in enumerate(ctx.names) if n.startswith(("cont__", "ord__"))]
    if not cols:
        return _r("inputs_standardised", True, 0.0, "no scaled columns")
    Z = ctx.Xt[:, cols].astype(float)
    dev = float(max(np.abs(Z.mean(axis=0)).max(), np.abs(Z.std(axis=0) - 1).max()))
    return _r("inputs_standardised", dev < tol, dev, f"max |mean| or |std-1| deviation over scaled columns = {dev:.3f}",
              "Unscaled inputs make optimisation ill-conditioned for gradient-trained nets.", "StandardScaler fitted on train.")


def check_params_vs_samples(ctx: AssumptionContext, n_params: int, ratio_warn: float = 0.1) -> AssumptionResult:
    ratio = n_params / len(ctx.y)
    return _r("parameters_per_sample", ratio <= ratio_warn, ratio, f"{n_params:,} parameters for {len(ctx.y):,} samples (ratio {ratio:.3f})",
              "Over-parameterised nets memorise small datasets.", "Weight decay, dropout, early stopping.")


# ------------------------------------------------------------------------------------------------ trees
def check_both_classes_present(ctx: AssumptionContext) -> AssumptionResult:
    ok = len(np.unique(ctx.y)) == 2
    return _r("both_classes_present", ok, int(len(np.unique(ctx.y))), "labels contain both classes", "Boosting/forests need both labels.",
              warn_only=False, blocking=True)
