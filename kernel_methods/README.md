# ESLII Ch. 6 kernel smoothing methods - regression (Zillow) and classification (Santander)

    pip install -r requirements.txt
    python run_all.py --task regression     --data zillow.csv          --results results_reg
    python run_all.py --task classification --data santander_train.csv --results results_clf

Per task: EDA -> split + features -> 5 methods IN PARALLEL (live progress) -> comparison + winner (winner.txt / winner.json / comparison*.png).

regression     (temporal split; fit target = clipped logerror; metrics on raw logerror): nw_regression, local_polynomial (1-D, chosen predictor),
               structured_local_regression (varying-coefficient, top-6 predictors), local_likelihood_regression (Gaussian), rbf_network (top-6 predictors)
               winner = lowest test MSE +- SE; paired Diebold-Mariano / t tests, Holm-adjusted
classification (stratified 80/10/10): nw_classification, local_logistic, kernel_density_classifier (top-8 features),
               naive_bayes, gaussian_mixture_classifier (all 200 features)
               winner = lowest test log-loss +- SE; error +- SE and AUC +- SE reported; paired t / McNemar / DeLong, Holm-adjusted

Compute: kernel methods are memory-based. Smoothing parameters are tuned by 10-fold CV on a 30k-row subsample (--tune-rows); the final model uses ALL training rows and is
evaluated on the full validation and test sets. Spans are fractions of the reference set, so they transfer. Slowest: local_logistic and kernel_density_classifier.
Other flags: --methods ..., --workers N, --cv-folds, --one-se paired|plain, --predictor, --n-top-features, --nb-compare-rows (0 = skip), --skip-eda, --reuse-prepared.
One method alone: python methods/local_logistic.py --prepared results_clf/prepared --results results_clf
Verify numerics: python tests/test_correctness.py     Synthetic data: tests/make_synthetic_zillow.py, tests/make_synthetic_santander.py
