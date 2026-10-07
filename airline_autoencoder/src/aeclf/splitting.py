"""Step 2b: train / validation / test split that reflects the population, plus evidence that it does.

Design decisions
  * Stratified on a composite key (target x Class x Type of Travel) so that the target AND the two strongest
    segments keep their proportions in every split (rare strata are merged to avoid empty cells).
  * Rows with identical feature vectors are kept together (group split). Otherwise a duplicate in train and test
    would inflate test performance for memorising models (trees, kNN, deep nets).
  * Splitting happens BEFORE any learned preprocessing, so no statistic leaks from validation/test into training.
  * With n ~ 700k, classical p-values are always 'significant'. Representativeness is therefore judged by
    effect sizes (KS statistic, Cramer's V, prevalence difference) and by adversarial validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 20  # granularity of the assignment: 70/15/15 -> 14/3/3 folds


def build_strata(X: pd.DataFrame, y: pd.Series, cfg: dict) -> np.ndarray:
    parts = []
    for name in cfg["split"]["stratify_by"]:
        if name == cfg["data"]["target"]:
            parts.append(y.astype(str).to_numpy())
        elif name in X.columns:
            parts.append(X[name].astype(str).to_numpy())
    if not parts:
        parts = [y.astype(str).to_numpy()]
    key = pd.Series(["|".join(t) for t in zip(*parts)])
    counts = key.value_counts()
    rare = counts[counts < cfg["split"].get("min_stratum_size", 20)].index
    key = key.where(~key.isin(rare), other="__rare__")
    return pd.factorize(key)[0]


def make_splits(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, np.ndarray]:
    seed = cfg["project"]["seed"]
    s = cfg["split"]
    handling = cfg["data"].get("duplicate_handling", "group_split")
    n = len(X)
    if n < N_FOLDS * 10:
        raise ValueError(f"Need at least {N_FOLDS * 10} rows to split, got {n}")
    if handling == "group_split":
        row_hash = pd.util.hash_pandas_object(X, index=False).to_numpy()
        groups = pd.factorize(row_hash)[0]
    else:
        groups = np.arange(n)
    strata = build_strata(X, y, cfg)
    n_test = max(1, round(s["test"] * N_FOLDS))
    n_val = max(1, round(s["validation"] * N_FOLDS))
    fold = np.empty(n, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(sgkf.split(X, strata, groups)):
        fold[idx] = k
    test_idx = np.where(fold < n_test)[0]
    val_idx = np.where((fold >= n_test) & (fold < n_test + n_val))[0]
    train_idx = np.where(fold >= n_test + n_val)[0]
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def population_report(X: pd.DataFrame, y: pd.Series, splits: dict[str, np.ndarray], schema, cfg: dict) -> dict:
    """Compare each split against the full population. Returns a JSON-able dict with a pass/fail verdict.

    Limits are max(base effect-size threshold, sampling-noise allowance): a 1,200-row split legitimately wobbles
    more than a 100,000-row one, so the allowance shrinks as ~1/sqrt(n).
    """
    base = {"ks": 0.03, "cramers_v": 0.03, "prevalence_diff": 0.01}
    rep: dict = {"thresholds": base, "sizes": {k: int(len(v)) for k, v in splits.items()}, "splits": {}}
    overall_rate = float(y.mean())
    worst = {"ks": 0.0, "cramers_v": 0.0, "prevalence_diff": 0.0}
    all_ok = True
    for name, idx in splits.items():
        n_s = len(idx)
        limits = {"ks": max(base["ks"], 1.95 / np.sqrt(n_s)), "cramers_v": max(base["cramers_v"], 3.0 / np.sqrt(n_s)),
                  "prevalence_diff": max(base["prevalence_diff"], 3.0 * np.sqrt(0.25 / n_s))}
        sub, ysub = X.iloc[idx], y.iloc[idx]
        entry = {"n": int(n_s), "limits": limits, "target_rate": float(ysub.mean()),
                 "prevalence_diff": float(abs(ysub.mean() - overall_rate)), "continuous_ks": {}, "categorical_v": {}}
        worst["prevalence_diff"] = max(worst["prevalence_diff"], entry["prevalence_diff"])
        all_ok &= entry["prevalence_diff"] <= limits["prevalence_diff"]
        for col in schema.continuous + schema.ordinal:
            a = pd.to_numeric(sub[col], errors="coerce").dropna()
            b = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(a) and len(b):
                ks = float(stats.ks_2samp(a.sample(min(len(a), 50000), random_state=0),
                                          b.sample(min(len(b), 50000), random_state=0)).statistic)
                entry["continuous_ks"][col] = ks
                worst["ks"] = max(worst["ks"], ks)
                all_ok &= ks <= limits["ks"]
        member = np.zeros(len(X), dtype=int)
        member[idx] = 1
        for col in schema.nominal:
            v = _cramers_v(X[col].astype(str), pd.Series(member))
            entry["categorical_v"][col] = v
            worst["cramers_v"] = max(worst["cramers_v"], v)
            all_ok &= v <= limits["cramers_v"]
        rep["splits"][name] = entry
    rep["worst"] = worst
    rep["representative"] = bool(all_ok)
    return rep


def adversarial_validation(Xa: pd.DataFrame, Xb: pd.DataFrame, max_rows: int = 40000, seed: int = 0) -> float:
    """AUC of a classifier trying to tell set A from set B. ~0.5 means the sets are indistinguishable."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    a = Xa.sample(min(len(Xa), max_rows), random_state=seed)
    b = Xb.sample(min(len(Xb), max_rows), random_state=seed)
    data = pd.concat([a, b], ignore_index=True)
    label = np.r_[np.zeros(len(a)), np.ones(len(b))]
    data = pd.get_dummies(data, dtype=float)
    clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    proba = cross_val_predict(clf, data, label, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(label, proba))
