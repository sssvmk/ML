# ESLII Ch. 15 - random forests: regression (Zillow) and classification (Santander)

    pip install -r requirements.txt
    python run_all.py --task regression     --data zillow.csv          --results results_reg
    python run_all.py --task classification --data santander_train.csv --results results_clf

Per task: EDA -> split + FULL feature engineering (Zillow ~170-240 candidate variables, Santander 600) -> variable ranking inside every CV fold -> the four forest entries IN PARALLEL (live progress) -> comparison + winner.
Entries (all scikit-learn): rf_tuned (RandomForest*, tuned), extratrees (ExtraTrees*, tuned, no bootstrap -> no OOB), rf_defaults (the book's defaults, not tuned: regression m = floor(p/3), node size 5;
classification m = floor(sqrt(p)), node size 1), bagging (m = p, node size tuned) + the mean / base-rate baseline.
Tuning: random search (--n-configs-rf) over m, minimum node size, [class weights] and the fraction of the fold-ranked variables kept (every variable is kept unless dropping some is clearly better, one-SE rule on
PAIRED fold differences); time-blocked CV (Zillow) / stratified CV (Santander) on a tuning subsample (--tune-rows, default 20000); final forest on ALL training rows (--rf-trees-final, default 300).
Winner: regression = lowest test MSE +- SE (MAE alongside; Diebold-Mariano / paired t, Holm); classification = highest test AUC +- DeLong SE (error +- SE at the validation-tuned threshold, log-loss +- SE and Brier;
DeLong / McNemar / paired t, Holm); report: the winner, every entry not significantly worse, the baseline.
Per-entry outputs: variable_importance.csv/.png (split-based + scikit-learn permutation importance on validation rows), curves_vs_trees (validation + OOB vs number of trees), cv_over_m_and_node_size.png, proximity_plot.png
(apply() leaf co-occurrence over --prox-trees trees on --prox-rows rows, scikit-learn MDS), forest_extra_metrics.json (classification: clipped AND Platt-calibrated log-loss / Brier), OOB metrics per configuration in cv_curve.csv.
Compute / memory: fully grown trees on all rows are large (several GB for 300 trees on 160k Santander rows); bagging (m = p) on 600 variables is ~25x slower than a default forest. Use --workers 2, --rf-trees-final 200,
--n-configs-rf 6 or --tune-rows 10000 to reduce. OOB error is optimistic on time-ordered Zillow: tuning there uses time-blocked CV, OOB is reported alongside.
Santander leakage guard: identifier / row-index columns and any single feature with univariate train AUC >= --leak-auc (default 0.9) are excluded (prepared/excluded_columns.csv).
Verify: python tests/test_correctness.py

Recovery / reruns: if one entry fails or you want to retrain only some entries, keep the results folder and run
    python run_all.py --task regression --data zillow.csv --results results_reg --methods rf_defaults --reuse-prepared --compare-existing
(use the SAME --cv-folds / --tune-rows as the first run). --compare-existing compares every entry that already has results in the folder. Extra outputs (importances, curves, proximity) can no longer destroy an
entry's results: a failure there is logged, recorded as "extra_error" in summary.json, and the model, predictions and metrics are still saved. Permutation importance runs on threads (no pickling of the forest).
