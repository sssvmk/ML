#!/usr/bin/env bash
# End-to-end demo on SYNTHETIC data (results say nothing about the real dataset).
set -euo pipefail
python -m aeclf.synthetic --rows 20000 --seed 1 --out /tmp/airline_train.csv
python -m aeclf.synthetic --rows 3000 --seed 2 --no-target --out /tmp/airline_unseen.csv
python -m aeclf.run_pipeline --data /tmp/airline_train.csv --run-id demo --set autoencoder.search.n_trials=4 --set classifier.search.n_trials=4 --explain offline
python -m aeclf.predict --input /tmp/airline_unseen.csv --model runs/demo/champion --output runs/demo/predictions.csv --drift-report
