"""
hd_runner.py - the generic train / validate / test routine every method file calls. Per method it:
 1. tunes by REPEATED (stratified) K-fold CV on the training rows only (GridSearchCV; every preprocessing / screening step is inside the pipeline, so it is refitted inside each fold),
    choosing the least complex candidate within one Nadeau-Bengio-corrected SE of the best (one-SE rule);
 2. validation (regression, 12 rows): the CV-chosen model fitted on the training rows is scored on the validation rows;
 3. final fit: regression on train + validation, classification on the 63 training samples, at the CV-chosen hyper-parameters; the TEST rows are used once;
 4. writes cv_results.csv, predictions_test.csv (per-sample losses for the comparison tests), summary.json, plots and the method's extra outputs, and a log file.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone

import hd_lib as hl
from hd_data import load_bundle


def run_method(name, task, bundle_dir, out_dir, cfg, prog, make_est, grid, complexity, n_features, extra=None, baseline=False, notes="", scoring=None):
    t0 = time.time()
    d = load_bundle(bundle_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = hl.get_logger(out / f"{name}.log", name)
    reg = task == "regression"
    X_tr, y_tr, X_va, y_va, X_te, y_te = (d[k] for k in ("X_tr", "y_tr", "X_va", "y_va", "X_te", "y_te"))
    p = X_tr.shape[1]
    prog.set_total(5)
    est = make_est(d["winsorize"], out)
    g = grid(p) if callable(grid) else grid
    scoring = scoring or ("neg_mean_squared_error" if reg else "accuracy")
    log.info("%s | task=%s | train=%d val=%d test=%d | genes=%d | winsorise=%s | CV %dx%d | scoring=%s", name, task, len(y_tr), len(y_va), len(y_te), p, d["winsorize"], cfg.cv_repeats, cfg.cv_folds, scoring)
    # ---- 1. tuning --------------------------------------------------------------------------------------------------------------------------
    cv, best, cv_min = hl.tune(est, g, X_tr, y_tr, task, cfg, complexity, scoring)
    cv.assign(params=cv["params"].astype(str)).to_csv(out / "cv_results.csv", index=False)
    log.info("CV minimum %s | one-SE choice %s", cv_min, best)
    prog.tick("CV tuning")
    # ---- 2. validation ----------------------------------------------------------------------------------------------------------------------
    val = None
    model_tr = clone(est).set_params(**best)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        model_tr.fit(X_tr, y_tr)
    if reg and len(y_va):
        val = hl.reg_metrics(y_va, model_tr.predict(X_va), float(y_tr.mean()))
        log.info("validation: %s", val)
    prog.tick("validation")
    # ---- 3. final fit and test ------------------------------------------------------------------------------------------------------------
    Xf, yf = (np.vstack([X_tr, X_va]), np.r_[y_tr, y_va]) if (reg and len(y_va)) else (X_tr, y_tr)
    final = clone(est).set_params(**best)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        final.fit(Xf, yf)
    pred = final.predict(X_te)
    pf = pd.DataFrame({"y_true": y_te, "y_pred": pred})
    if reg:
        test = hl.reg_metrics(y_te, pred, float(yf.mean()))
        pf["sq_err"] = (y_te - pred) ** 2
    else:
        classes = np.unique(np.r_[y_tr, y_te])
        proba = final.predict_proba(X_te) if hasattr(final, "predict_proba") else None
        score = final.decision_function(X_te) if hasattr(final, "decision_function") else None
        test = hl.clf_metrics(y_te, pred, proba, score, classes)
        pf["correct"] = (pred == y_te).astype(int)
        if proba is not None:
            for k, c in enumerate(classes):
                pf[f"p_class{c}"] = proba[:, k]
            pc = np.clip(proba, 1e-6, 1 - 1e-6); pc = pc / pc.sum(1, keepdims=True)
            pf["log_loss"] = -np.log(pc[np.arange(len(y_te)), np.searchsorted(classes, y_te)])
    pf.to_csv(out / "predictions_test.csv", index=False)
    nfeat = int(n_features(final, p))
    log.info("TEST: %s | genes used: %d", test, nfeat)
    prog.tick("test evaluation")
    # ---- 4. outputs ---------------------------------------------------------------------------------------------------------------------------
    summary = {"method": name, "task": task, "baseline": bool(baseline), "selected": best, "cv_minimum": cv_min, "cv": {"selected_loss": float(cv.loc[cv["selected_one_se"], "cv_loss"].iloc[0]), "selected_se_nb": float(cv.loc[cv["selected_one_se"], "cv_se"].iloc[0]),
               "minimum_loss": float(cv["cv_loss"].min()), "n_configurations": int(len(cv)), "folds": f"{cfg.cv_repeats}x{cfg.cv_folds}"},
               "validation": val, "test": test, "n_features_used": nfeat, "n_genes_total": int(p), "n_train": int(len(yf)), "n_test": int(len(y_te)), "notes": notes}
    for fn, path, ttl in ((hl.plot_cv, "cv_curve.png", name), (hl.plot_heatmap, "cv_heatmap.png", f"{name}: CV loss")):
        try:
            fn(cv, out / path, ttl)
        except Exception as e:                                           # a plotting problem must never destroy the run's results
            log.error("plot %s failed: %s: %s", path, type(e).__name__, e)
    if reg:
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        lo, hi = float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))
        ax.scatter(y_te, pred, s=30, label="test"); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        if val is not None:
            ax.scatter(y_va, model_tr.predict(X_va), s=20, alpha=0.6, label="validation")
        ax.set_xlabel("actual y"); ax.set_ylabel("predicted y"); ax.legend(); ax.set_title(f"{name}: test MSE {test['mse']:.3f}")
        fig.tight_layout(); fig.savefig(out / "pred_vs_actual.png", dpi=120); plt.close(fig)
    else:
        cm = hl.plot_confusion(y_te, pred, classes, out / "confusion_test.png", f"{name}: test error {test['error']:.3f}")
        summary["confusion_matrix_test"] = cm.tolist()
    if extra is not None:
        ctx = dict(model=final, est=est, best=best, cv=cv, d=d, out=out, cfg=cfg, task=task, log=log, summary=summary, genes=d["genes"], Xf=Xf, yf=yf, p=p, make=lambda: clone(est).set_params(**best))
        try:
            extra(ctx)
        except Exception as e:                                           # extras never destroy the run's results
            import traceback
            log.error("extra outputs FAILED (%s: %s)\n%s", type(e).__name__, e, traceback.format_exc())
            summary["extra_error"] = f"{type(e).__name__}: {e}"
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(hl.jsonable(summary), indent=2))
    shutil.rmtree(out / "cache", ignore_errors=True)
    prog.tick("done")
    log.info("done in %.1fs", summary["runtime_sec"])
    return summary
