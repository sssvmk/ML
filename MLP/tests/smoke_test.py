#!/usr/bin/env python3
"""
End-to-end smoke test of the MLP pipeline on SYNTHETIC data shaped like MNIST and
California Housing (no downloads needed), against a throwaway MLflow store.

    python tests/smoke_test.py

It checks: tuning + nested runs + final run + logged metrics/artifacts, model registration,
the champion/challenger gate, loading the registered model in a FRESH process from another
directory, and inference from the registry. It does NOT measure real accuracy/RMSE.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

tmp = Path(tempfile.mkdtemp(prefix="mlp_smoke_"))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp}/mlflow.db"
os.chdir(tmp)

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

import mlp_core  # noqa: E402

def fake_mnist(root, seed):
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(10, 784)).astype(np.float32)

    def make(n):
        y = rng.integers(0, 10, n)
        x = protos[y] + 2.0 * rng.normal(size=(n, 784)).astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

    splits = {k: TensorDataset(*make(n)) for k, n in [("train", 3000), ("val", 500), ("test", 500)]}
    info = dict(task="classification", in_dim=784, out_dim=10, y_mean=0.0, y_std=1.0, baseline={},
                class_names=[str(i) for i in range(10)], source="synthetic-mnist",
                split_info={"presplit": True, "strategy": "synthetic pre-split (smoke test)"},
                preproc={"type": "mnist", "mean": mlp_core.MNIST_MEAN, "std": mlp_core.MNIST_STD})
    return splits, info


def fake_housing(seed):
    rng = np.random.default_rng(seed)
    names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    mu, sd = rng.normal(size=8) * 5, rng.uniform(1, 4, size=8)
    ym, ys = 2.0, 1.2

    def tens(n):
        z = rng.normal(size=(n, 8))
        y = 2 + z[:, 0] + 0.5 * np.maximum(z[:, 1], 0) + 0.3 * rng.normal(size=n)
        return TensorDataset(torch.tensor(z, dtype=torch.float32),
                             torch.tensor(((y - ym) / ys).reshape(-1, 1), dtype=torch.float32))

    splits = {k: tens(n) for k, n in [("train", 3000), ("val", 600), ("test", 600)]}
    t_orig = splits["test"].tensors[1].numpy().ravel() * ys + ym
    base_pred = t_orig + 0.6 * rng.normal(size=len(t_orig))  # a deliberately weaker "linear" reference
    info = dict(task="regression", in_dim=8, out_dim=1, y_mean=ym, y_std=ys,
                baseline={"linear_regression_test_mse": 0.36, "linear_regression_test_rmse": 0.6,
                          "predict_mean_test_rmse": 1.2},
                class_names=None, source="synthetic-housing", baseline_test_pred=base_pred,
                split_info={"presplit": False, "strategy": "synthetic single table split 3000/600/600"},
                preproc={"type": "housing", "feature_names": names, "x_mean": mu.tolist(),
                         "x_scale": sd.tolist(), "y_mean": ym, "y_std": ys})
    return splits, info


mlp_core._load_mnist = fake_mnist
mlp_core._load_housing = fake_housing
import run  # noqa: E402

client = MlflowClient()


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        sys.exit(1)


def pipeline(ds, trials, extra=()):
    return run.main(["--data", ds, "--trials", str(trials), "--trial-epochs", "2", "--final-epochs", "3",
                     "--startup-trials", "3", "--out", str(tmp / f"out_{ds}"),
                     "--study-dir", str(tmp / f"study_{ds}"), *extra])


# ---------------- housing (regression): 6 trials, run twice to exercise the gate ---------------- #
r1 = pipeline("housing", 6)
exp = client.get_experiment_by_name("MLP-housing")
runs = client.search_runs([exp.experiment_id], max_results=100)
phases = [r.data.tags.get("phase") for r in runs]
check(phases.count("pipeline") == 1 and phases.count("search_trial") == 6 and phases.count("final_candidate") == 1,
      f"runs logged: 1 pipeline + 6 trials + 1 final (got {phases.count('pipeline')}/{phases.count('search_trial')}/{phases.count('final_candidate')})")
final = [r for r in runs if r.data.tags.get("phase") == "final_candidate"][0]
for m in ["test_mse", "test_mse_se", "test_mse_ci95_low", "test_mse_ci95_high", "test_rmse", "test_rmse_se",
          "test_mae", "test_mae_se", "test_r2", "test_mse_diff_vs_linear", "test_mse_diff_vs_linear_se",
          "test_loss", "final_val_loss", "val_loss", "train_loss", "lr"]:
    check(m in final.data.metrics, f"final run has metric {m}")
arts = {a.path for a in client.list_artifacts(final.info.run_id)}
check({"curves.png", "diagnosis.json", "results.json", "evaluation", "MODEL_CARD.md"} <= arts,
      f"final run artifacts: {sorted(arts)}")
check(final.data.metrics["test_mse_se"] > 0 and abs(final.data.metrics["test_mse"] - final.data.metrics["test_rmse"] ** 2) < 1e-6,
      "test MSE has a positive SE and equals RMSE^2")
check(final.data.metrics["test_mse_diff_vs_linear"] < 0, "paired MSE difference vs the (weaker) linear reference is negative")
check(len(client.get_metric_history(final.info.run_id, "val_loss")) >= 1, "per-epoch metric history logged")
parent = [r for r in runs if r.data.tags.get("phase") == "pipeline"][0]
check({"data/audit.json"} <= {a.path for a in client.list_artifacts(parent.info.run_id, "data")}, "data audit logged to the parent run")
check(parent.data.tags.get("split_presplit") == "false" and "single table" in parent.data.tags.get("split_strategy", ""),
      "parent run records the split strategy (housing: not pre-split)")
check(parent.data.tags.get("data_audit_ok") == "true", "data audit passed")
check("search_improvement_vs_default" in parent.data.metrics, "parent logs improvement vs default config")
check(any(r.data.tags.get("is_default_baseline") == "true" for r in runs), "trial 0 tagged as default baseline")
states = [r.data.tags.get("optuna_state") for r in runs if r.data.tags.get("phase") == "search_trial"]
check(all(st in ("COMPLETE", "PRUNED", "DIVERGED") for st in states), f"every trial run has a final state tag: {states}")

mv = client.get_model_version_by_alias("MLP-housing", "champion")
check(str(mv.version) == "1", f"first version auto-promoted to champion (got version={mv.version!r})")
check(str(client.get_model_version_by_alias("MLP-housing", "challenger").version) == "1", "challenger alias set")
check(mv.tags.get("validated") == "true", "reload-and-predict verification passed")

r2 = pipeline("housing", 6)
ch = client.get_model_version_by_alias("MLP-housing", "challenger")
check(str(ch.version) == "2", "second run registered as version 2 / challenger")
decision = ch.tags.get("promotion_decision", "")
check("does not beat champion" in decision and decision.startswith("not promoted"),
      f"an identical retrain does NOT replace the champion (tie / below margin): {decision}")
champ_v = str(client.get_model_version_by_alias("MLP-housing", "champion").version)
check(champ_v == "1", f"champion alias still v1 (got v{champ_v})")

champ_before = champ_v
pipeline("housing", 2, ["--max-mse", "1e-9"])
ch3 = client.get_model_version_by_alias("MLP-housing", "challenger")
check("acceptance target not met" in ch3.tags.get("promotion_decision", ""),
      f"impossible --max-mse blocks promotion: {ch3.tags.get('promotion_decision')}")
check(str(client.get_model_version_by_alias("MLP-housing", "champion").version) == champ_before,
      "champion unchanged after a blocked promotion")
try:
    run.main(["--data", "housing", "--min-auc", "0.9"])
    check(False, "--min-auc on a regression dataset should be rejected")
except SystemExit:
    check(True, "--min-auc rejected for a regression dataset")

# ---------------- fresh-process load of the registered model from another directory ---------------- #
names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
code = (
    "import mlflow, pandas as pd, json;"
    "m = mlflow.pyfunc.load_model('models:/MLP-housing@champion');"
    f"df = pd.DataFrame([[0.0]*8], columns={names});"
    "print(json.dumps(m.predict(df).to_dict('records')))"
)
p = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True,
                   env={**os.environ, "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"]})
check(p.returncode == 0 and "prediction_usd" in p.stdout, "registered model loads and predicts in a fresh process")
if p.returncode != 0:
    print(p.stderr[-1500:])

import infer  # noqa: E402
pr = infer.Predictor.from_registry("models:/MLP-housing@champion")
out = pr.predict_housing([[0.0] * 8])
check("prediction_usd" in out[0], "infer.Predictor.from_registry works")

# ---------------- mnist (classification) ---------------- #
pipeline("mnist", 4, ["--min-auc", "0.5", "--bootstrap", "50"])
exp = client.get_experiment_by_name("MLP-mnist")
final = [r for r in client.search_runs([exp.experiment_id])
         if r.data.tags.get("phase") == "final_candidate"][0]
for m in ["test_auc_macro_ovr", "test_auc_se", "test_auc_ci95_low", "test_auc_ci95_high", "test_accuracy",
          "test_accuracy_se", "test_log_loss", "test_log_loss_se", "test_macro_f1", "test_auc_class_0",
          "test_acc_class_0"]:
    check(m in final.data.metrics, f"classification run has metric {m}")
check(final.data.metrics["test_auc_ci95_low"] <= final.data.metrics["test_auc_macro_ovr"] <= final.data.metrics["test_auc_ci95_high"],
      "AUC lies inside its bootstrap interval")
check(len(client.get_metric_history(final.info.run_id, "val_auc")) >= 1, "per-epoch validation AUC logged")
mn_parent = [r for r in client.search_runs([exp.experiment_id]) if r.data.tags.get("phase") == "pipeline"][0]
check(mn_parent.data.tags.get("split_presplit") == "true", "parent run records the split strategy (mnist: pre-split)")
check("required >= 0.5" in client.get_model_version_by_alias("MLP-mnist", "champion").tags.get("acceptance_target", ""),
      "AUC acceptance target recorded on the champion version")
ev = {a.path for a in client.list_artifacts(final.info.run_id, "evaluation")}
check({"evaluation/confusion_matrix.png", "evaluation/roc_curves.png", "evaluation/worst_errors.csv"} <= ev,
      f"evaluation artifacts: {sorted(ev)}")

code = (
    "import mlflow, numpy as np, json;"
    "m = mlflow.pyfunc.load_model('models:/MLP-mnist@champion');"
    "x = np.zeros((2,784), dtype=np.float32);"
    "o = m.predict(x); print(o.shape, list(o.columns)[:3])"
)
p = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True,
                   env={**os.environ})
check(p.returncode == 0 and "(2, 12)" in p.stdout, f"mnist model loads in fresh process: {p.stdout.strip()}")
if p.returncode != 0:
    print(p.stderr[-1500:])

pipeline("mnist", 2, ["--min-auc", "1.01", "--bootstrap", "20"])
ch = client.get_model_version_by_alias("MLP-mnist", "challenger")
check("acceptance target not met" in ch.tags.get("promotion_decision", "") and
      str(client.get_model_version_by_alias("MLP-mnist", "champion").version) == "1",
      "impossible --min-auc blocks promotion; champion stays v1")

# ---------------- determinism (CPU, same seed, no search) ---------------- #
det = [run.main(["--data", "housing", "--trials", "0", "--final-epochs", "3", "--no-register",
                 "--out", str(tmp / f"det{i}"), "--seed", "7"])["test_metrics"] for i in (1, 2)]
check(det[0]["mse"] == det[1]["mse"], f"same seed -> identical test MSE on CPU ({det[0]['mse']:.6f})")

# ---------------- no-tuning paths ---------------- #
run.main(["--data", "housing", "--trials", "0", "--final-epochs", "2", "--lr", "0.02",
          "--out", str(tmp / "out_notune"), "--no-register"])
check(True, "--trials 0 with overrides and --no-register runs")
try:
    run.main(["--data", "housing", "--lr", "0.02"])
    check(False, "overrides with a search should be rejected")
except SystemExit:
    check(True, "overrides combined with a search are rejected")

print(f"\nSMOKE TEST PASSED (artifacts in {tmp})")
