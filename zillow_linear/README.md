# Zillow logerror - 12 linear regression methods (ESLII Ch. 3)

    pip install -r requirements.txt
    python run_all.py --data /path/to/zillow.csv --results results

Stages: EDA -> temporal split + features -> cached CV statistics -> 12 methods in parallel -> comparison + winner.
Split: train <= 2017-02-28, validation 2017-03-01..07-31, test >= 2017-08-01.

Useful flags: --cv-scheme blocked|random (default blocked), --refit train|train_val (default train_val),
--subset-pool 20 (exact up to 22), --dantzig-pool 30, --workers N, --methods ols ridge ..., --skip-eda, --reuse-prepared.

Run one method alone:  python methods/ridge.py --prepared results/prepared   (add --data zillow.csv if not prepared yet)
Verify solvers:        python tests/test_correctness.py
Synthetic smoke data:  python tests/make_synthetic.py 15000 synthetic.csv

Results per method in results/<method>/ ; comparison.csv, comparison_test_mse.png, winner.txt in results/.
