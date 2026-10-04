"""Human-readable reporting: the headline metric block and an auto-generated model card."""


def headline_lines(task, m, reference=None, info=None):
    """Lines for the console / run report. Headline = AUC (classification) or MSE +- SE (regression).

    `info` supplies the target's label and, for money-valued targets, a conversion to dollars.
    """
    info = info or {}
    label, money = info.get("target_label", "target"), info.get("money_scale")
    if task == "classification":
        lines = [
            f"ROC-AUC (macro one-vs-rest): {m['auc_macro_ovr']:.4f}   "
            f"[95% bootstrap CI {m['auc_ci95_low']:.4f} - {m['auc_ci95_high']:.4f}, SE {m['auc_se']:.4f}]",
            f"accuracy: {m['accuracy']:.4f} +- {m['accuracy_se']:.4f} (SE)",
            f"log-loss: {m['log_loss']:.4f} +- {m['log_loss_se']:.4f} (SE)",
            f"macro-F1: {m['macro_f1']:.4f}",
        ]
        if reference and "logistic_regression_test_auc" in reference:
            lines.append(f"reference (not tuned): logistic-regression AUC {reference['logistic_regression_test_auc']:.4f}, "
                         f"majority-class accuracy {reference['majority_class_test_accuracy']:.4f}")
        return lines
    unit_sq = "($100k)^2" if money else f"({label})^2"
    rmse_note = f"(= ${m['rmse'] * money:,.0f} typical error)" if money else f"(same units as {label})"
    lines = [
        f"MSE: {m['mse']:.4f} +- {m['mse_se']:.4f} (SE)   [units: {unit_sq}; 95% CI "
        f"{m['mse_ci95_low']:.4f} - {m['mse_ci95_high']:.4f}]",
        f"RMSE: {m['rmse']:.4f} +- {m['rmse_se']:.4f} (SE)   {rmse_note}",
        f"MAE: {m['mae']:.4f} +- {m['mae_se']:.4f} (SE)   R2: {m['r2']:.4f}",
    ]
    if "mse_diff_vs_linear" in m:
        d, se = m["mse_diff_vs_linear"], m["mse_diff_vs_linear_se"]
        verdict = "model better" if d + 1.96 * se < 0 else ("model worse" if d - 1.96 * se > 0
                                                            else "difference not significant")
        lines.append(f"MSE difference vs linear regression (paired): {d:+.4f} +- {se:.4f} (SE) -> {verdict}")
    if reference and "linear_regression_test_mse" in reference:
        lines.append(f"linear-regression reference MSE: {reference['linear_regression_test_mse']:.4f}")
    return lines


def model_card(*, dataset, task, cfg, n_params, info, audit, val_metrics, test_metrics,
               diagnosis, search, promotion, ids, versions, git_commit, target_msg, kind=None):
    split = info.get("split_info", {})
    head = "\n".join(f"- {l}" for l in headline_lines(task, test_metrics, info.get("baseline"), info))
    notes = "\n".join(f"  - {n}" for n in info.get("notes", [])) or "  - none"
    search_txt = ("none (config was fixed)" if not search else
                  f"Optuna TPE + MedianPruner, {search['n_trials']} trials ({search['n_pruned']} pruned); "
                  f"best validation loss {search['best_value']:.4f}"
                  + (f" vs {search['baseline_value']:.4f} for the default config"
                     if search.get("baseline_value") is not None else ""))
    warn = "\n".join(f"- {w}" for w in audit["warnings"]) or "- none"
    limits = {
        "mnist": "Trained on centred 28x28 handwritten digits (white on black). Expect lower accuracy on "
                 "photos, other scripts, off-centre or noisy digits. Not evaluated for robustness or fairness.",
        "housing": "Trained on 1990 California census block-group aggregates; the target is capped at "
                   "$500k, so very high prices are unreliable. Predictions describe a block group, not a "
                   "single house, and the data are decades old. Not evaluated for fairness across areas.",
    }.get(kind or dataset, f"Trained on {info.get('source', 'a user-supplied CSV')} (target: "
                   f"{info.get('target_label', 'target')}). Valid only for data that looks like the training "
                   f"file. Inputs far outside the training range are flagged but still scored. Not "
                   f"evaluated for robustness or fairness.")
    return f"""# Model card: MLP-{dataset}

**Task:** {task}   **Model:** fully connected network (ReLU hidden units, {n_params:,} parameters)
**Hidden layers:** {cfg['hidden']}   **Optimizer:** {cfg['optimizer']} (lr {cfg['lr']:.5g}, weight decay {cfg['weight_decay']:.2g}, dropout {cfg['dropout']})

## Data and splits
- Source: {info.get('source', dataset)}
- {split.get('strategy', 'split not described')}
- Sizes: {audit['sizes']}
- Data notes (joins, dropped rows or columns, auto-detected task):
{notes}
- Data audit: {'passed' if audit['ok'] else 'FAILED'}; warnings:
{warn}
- Split/data fingerprint: `{ids['fingerprint']}`

## Test performance (test set used once, after model selection)
{head}

Standard errors and intervals reflect the finite size of the test set only; they do not include
variation between training runs with different seeds.

Validation (selection) metrics at the chosen epoch: { {k: round(v, 4) for k, v in val_metrics.items()} }

## Training
- Hyperparameter search: {search_txt}
- Selection criterion: lowest validation loss; early stopping on validation loss (best epoch {diagnosis['best_epoch']} of {diagnosis['epochs_run']}).
- Diagnosis: {diagnosis['verdict']} (val/train loss ratio {diagnosis['val_over_train_loss']:.2f}; heuristic).

## Intended use and limitations
{limits}

## Governance and lineage
- Acceptance target: {target_msg}
- Registry decision: {promotion}
- MLflow: experiment `{ids['experiment']}`, parent run `{ids['parent_run_id']}`, final run `{ids['final_run_id']}`
- Code version (git): {git_commit}; torch {versions['torch']}, mlflow {versions['mlflow']}, python {versions['python']}
- Human approval: **not recorded** (see TODO.md)
"""
