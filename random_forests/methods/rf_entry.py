"""
Factory for the four forest entries (scikit-learn estimators; one file per entry calls make()):
  rf_tuned     RandomForestRegressor / Classifier, hyper-parameters tuned by CV (m, minimum node size, [class weights], fraction of variables kept)
  extratrees   ExtraTreesRegressor / Classifier (random cut-points, no bootstrap -> no OOB), the same search space
  rf_defaults  the book's defaults, NOT tuned: regression m = floor(p/3), minimum node size 5; classification m = floor(sqrt(p)), minimum node size 1; all p variables
  bagging      m = p (no random variable subset), minimum node size tuned - the Chapter 15 comparison point
Loss: regression - squared error (RSS splitting), prediction = tree average (15.2); classification - Gini splitting, class vote fraction (soft vote of the trees' leaf proportions; with fully grown trees it is the majority-vote
fraction of Algorithm 15.1).  Tuning: time-blocked (Zillow) / stratified (Santander) CV on the tuning subsample; selection on CV MSE / CV AUC with the one-SE rule (paired), complexity: larger minimum node size and smaller m = simpler.
Final forest: ALL training rows. Outputs per entry: test metrics (MSE +- SE and MAE; error +- SE, AUC +- DeLong SE, log-loss +- SE and Brier from clipped votes + Platt-calibrated votes), OOB metrics per configuration
(oob_mse / oob_auc, averaged over the CV folds), validation and OOB curves vs the number of trees, CV over (m, minimum node size), split-based and permutation importance, proximity plot (apply() leaf co-occurrence + MDS).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
import rflib as rl


def make(name, kind, mode):
    """mode: 'tuned' | 'defaults' | 'bagging'."""
    def make_grid(cfg, prep, Xs, ys):
        reg = prep.task == "regression"
        p = prep.p
        if mode == "defaults":
            m = max(1, p // 3) if reg else max(1, int(np.sqrt(p)))
            return [{"frac": 1.0, "mf": m, "leaf": 5 if reg else 1, "k": p, **({} if reg else {"cw": None})}]
        if mode == "bagging":
            space = {"frac": [1.0], "mf": [1.0], "leaf": [1, 5, 20, 50]}
        elif reg:
            space = {"frac": rl.FRACS, "mf": [0.1, 0.2, 0.333, 0.5, 0.75, 1.0], "leaf": [1, 3, 5, 10, 20, 50, 100]}
        else:
            space = {"frac": rl.FRACS, "mf": ["sqrt", 0.1, 0.2, 0.333, 0.5], "leaf": [1, 3, 5, 10, 20, 50]}
        if not reg:
            space["cw"] = [None, "balanced_subsample"]
        return rl.with_k(rl.sample_configs(space, cfg.n_configs_rf, cfg.seed), p)

    def complexity(hp):
        return rl.drop_penalty(hp) + 100.0 / hp["leaf"] + 10 * min(1.0, rl.mf_value(hp))

    def extra(ctx):
        rl.standard_extra(ctx, name)

    def run(prep_dir, out_dir, cfg, progress=None, console=False):
        task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
        return common.run_method(name, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=rl.make_predict_path(kind), complexity=complexity, extra=extra,
                                 select_by="mse" if task == "regression" else "auc", notes=f"scikit-learn {'ExtraTrees' if kind == 'et' else 'RandomForest'} ({mode}).", console=console)
    return make_grid, complexity, run
