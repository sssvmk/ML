"""Step 1: exploratory data analysis + statistical tests that drive later modelling decisions.

Every test below exists to answer a concrete modelling question (written next to it), not for decoration.
With ~700k rows every p-value is tiny, so we always report an EFFECT SIZE next to the p-value.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from .schema import Schema
from .utils import dump_json

PLOT_ROWS = 100_000


def _cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))), float(p)


def run_eda(X: pd.DataFrame, y: pd.Series, ids: pd.Series | None, schema: Schema, cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["seed"]
    n = len(X)
    findings: list[str] = []
    summary: dict = {"n_rows": n, "n_features": X.shape[1], "schema": schema.to_dict()}

    # ---- 1. target ---------------------------------------------------------------------------------------
    rate = float(y.mean())
    se = np.sqrt(rate * (1 - rate) / n)
    summary["target"] = {"positive_rate": rate, "ci95": [rate - 1.96 * se, rate + 1.96 * se],
                         "imbalance_ratio": float(max(rate, 1 - rate) / max(min(rate, 1 - rate), 1e-9))}
    if summary["target"]["imbalance_ratio"] > 3:
        findings.append(f"Target is imbalanced (ratio {summary['target']['imbalance_ratio']:.1f}:1) -> prefer PR-AUC/F1, consider class weights.")
    else:
        findings.append(f"Target is roughly balanced ({rate:.1%} positive) -> accuracy, ROC-AUC and F1 are all meaningful.")

    # ---- 2. quality: missingness, duplicates, constants, ranges ----------------------------------------------
    miss = X.isna().sum()
    summary["missing"] = {c: {"count": int(v), "share": float(v / n)} for c, v in miss.items() if v > 0}
    for c, d in summary["missing"].items():
        findings.append(f"'{c}' has {d['count']} missing values ({d['share']:.3%}) -> imputation + missing indicator.")
    dup_rows = int(pd.util.hash_pandas_object(X, index=False).duplicated().sum())
    summary["duplicate_feature_rows"] = dup_rows
    if dup_rows:
        findings.append(f"{dup_rows} rows share identical feature vectors -> kept together when splitting (group split).")
    for c in schema.ordinal:
        lo, hi = schema.ordinal_ranges[c]
        zero_share = float((pd.to_numeric(X[c], errors="coerce") == 0).mean())
        if zero_share > 0:
            summary.setdefault("zero_rating_share", {})[c] = zero_share
    if summary.get("zero_rating_share"):
        top = max(summary["zero_rating_share"], key=summary["zero_rating_share"].get)
        findings.append(f"Ratings contain 0 ('not applicable'); largest share in '{top}' ({summary['zero_rating_share'][top]:.1%}) -> 0 must not be averaged as a real rating.")

    # ---- 3. identifier leakage: does row order / id carry signal? ------------------------------------------
    if ids is not None:
        try:
            rho, p = stats.spearmanr(pd.to_numeric(ids, errors="coerce").fillna(0), y)
            summary["id_vs_target_spearman"] = {"rho": float(rho), "p": float(p)}
            findings.append("Identifier is " + ("**correlated** with the target (possible leak) - excluded from features anyway."
                                               if abs(rho) > 0.02 else "uncorrelated with the target (|rho|<0.02); dropped from features."))
        except Exception:
            pass

    # ---- 4. continuous variables ------------------------------------------------------------------------
    cont = {}
    sample = X.sample(min(n, 50_000), random_state=seed)
    ysample = y.loc[sample.index]
    for c in schema.continuous:
        v = pd.to_numeric(X[c], errors="coerce")
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_share = float(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean()) if iqr > 0 else 0.0
        a, b = v[y == 1].dropna(), v[y == 0].dropna()
        u = stats.mannwhitneyu(a.sample(min(len(a), 50_000), random_state=seed), b.sample(min(len(b), 50_000), random_state=seed))
        auc_like = float(u.statistic / (min(len(a), 50_000) * min(len(b), 50_000)))
        vs = pd.to_numeric(sample[c], errors="coerce").dropna()
        normal_p = float(stats.normaltest(vs.sample(min(len(vs), 5000), random_state=seed)).pvalue) if len(vs) > 20 else float("nan")
        cont[c] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "median": float(v.median()),
                   "max": float(v.max()), "skew": float(v.skew()), "kurtosis": float(v.kurt()),
                   "zero_share": float((v == 0).mean()), "extreme_outlier_share_3iqr": outlier_share,
                   "normality_p_dagostino": normal_p,
                   "mannwhitney_p": float(u.pvalue), "single_feature_auc": max(auc_like, 1 - auc_like)}
        if abs(cont[c]["skew"]) > 2:
            findings.append(f"'{c}' is heavily skewed (skew {cont[c]['skew']:.1f}) -> log1p transform for linear/NN models.")
        if cont[c]["single_feature_auc"] > 0.95:
            findings.append(f"LEAK WARNING: '{c}' alone gives AUC {cont[c]['single_feature_auc']:.3f}.")
    summary["continuous"] = cont

    # ---- 5. ordinal + nominal variables: association with target ------------------------------------------
    cat = {}
    for c in schema.ordinal + schema.nominal:
        v, p = _cramers_v(X[c].astype(str), y)
        rates = y.groupby(X[c].astype(str)).mean()
        counts = X[c].astype(str).value_counts()
        cat[c] = {"cramers_v": v, "chi2_p": p, "levels": int(len(rates)),
                  "target_rate_by_level": {k: float(r) for k, r in rates.items()},
                  "level_counts": {k: int(r) for k, r in counts.items()}}
    summary["categorical_and_ordinal"] = cat

    # ---- 6. mutual information (captures non-linear dependence missed by correlation) ---------------------
    mi_frame = pd.DataFrame(index=sample.index)
    for c in schema.continuous + schema.ordinal:
        mi_frame[c] = pd.to_numeric(sample[c], errors="coerce").fillna(pd.to_numeric(sample[c], errors="coerce").median())
    for c in schema.nominal:
        mi_frame[c] = pd.factorize(sample[c].astype(str))[0]
    mi = mutual_info_classif(mi_frame, ysample, discrete_features=[c in schema.ordinal + schema.nominal for c in mi_frame.columns],
                             random_state=seed)
    summary["mutual_information"] = dict(sorted(zip(mi_frame.columns, map(float, mi)), key=lambda kv: -kv[1]))

    # ---- 7. multicollinearity among numeric/ordinal (matters for logistic regression) -----------------------
    num_cols = schema.continuous + schema.ordinal
    corr = sample[num_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.8]
    summary["high_correlation_pairs"] = pairs
    for a, b, r in pairs:
        findings.append(f"'{a}' and '{b}' are strongly correlated (rho {r:.2f}) -> collinearity for linear models; trees unaffected.")

    # ---- 8. which are the strongest drivers (for the report) ------------------------------------------------
    top_mi = list(summary["mutual_information"].items())[:5]
    findings.append("Strongest dependence on the target (mutual information): " + ", ".join(f"{k} ({v:.3f})" for k, v in top_mi) + ".")
    summary["findings"] = findings

    # ---- plots ------------------------------------------------------------------------------------------------
    _plots(X, y, schema, corr, out_dir / "plots", seed)
    dump_json(summary, out_dir / "eda_summary.json")
    (out_dir / "data_profile.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _plots(X, y, schema, corr, d: Path, seed: int):
    take = X.sample(min(len(X), PLOT_ROWS), random_state=seed)
    ty = y.loc[take.index]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    y.value_counts(normalize=True).sort_index().plot.bar(ax=ax)
    ax.set(title="Target balance", xlabel="satisfied (1) / not (0)")
    fig.tight_layout(); fig.savefig(d / "target_balance.png", dpi=100); plt.close(fig)

    if schema.continuous:
        k = len(schema.continuous)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.5), squeeze=False)
        for ax, c in zip(axes[0], schema.continuous):
            v = pd.to_numeric(take[c], errors="coerce")
            v = np.log1p(v.clip(lower=0)) if v.skew() > 2 else v
            for lab in (0, 1):
                ax.hist(v[ty == lab].dropna(), bins=40, alpha=0.55, label=f"y={lab}", density=True)
            ax.set_title(c + (" (log1p)" if pd.to_numeric(take[c], errors="coerce").skew() > 2 else ""))
            ax.legend()
        fig.tight_layout(); fig.savefig(d / "continuous_by_target.png", dpi=100); plt.close(fig)

    levels_cols = schema.ordinal + schema.nominal
    cols = 4
    rows = int(np.ceil(len(levels_cols) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), levels_cols):
        ax.axis("on")
        y.groupby(X[c].astype(str)).mean().plot.bar(ax=ax, rot=30)
        ax.set(title=c[:28], ylabel="P(satisfied)", ylim=(0, 1), xlabel="")
        ax.axhline(y.mean(), color="r", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(d / "satisfaction_rate_by_level.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im); ax.set_title("Spearman correlation")
    fig.tight_layout(); fig.savefig(d / "correlation_heatmap.png", dpi=100); plt.close(fig)


def _markdown(s: dict) -> str:
    lines = ["# Data profile (auto-generated)", "", f"* rows: **{s['n_rows']:,}**, feature columns: **{s['n_features']}**",
             f"* positive rate: **{s['target']['positive_rate']:.2%}** (95% CI {s['target']['ci95'][0]:.2%} - {s['target']['ci95'][1]:.2%})",
             "", "## Variable roles", ""]
    for col, why in s["schema"]["reasons"].items():
        lines.append(f"* `{col}` -> {why}")
    lines += ["", "## Findings that influence modelling", ""] + [f"* {f}" for f in s["findings"]]
    lines += ["", "## Mutual information with target (top 10)", ""]
    for k, v in list(s["mutual_information"].items())[:10]:
        lines.append(f"* {k}: {v:.4f}")
    return "\n".join(lines) + "\n"
