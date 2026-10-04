"""
End-to-end tests of CSV data: run.py --data file.csv ... through search, final training, test
evaluation, MLflow logging, registration, reload of the registered model, and infer.py.

The data are SYNTHETIC, shaped like two well-known Kaggle tables (no downloads):
  * santander-like : ID_code + binary `target` (~10% positives, imbalanced) + 200 numeric columns
  * zillow-like    : a small `train` file (parcelid, logerror, transactiondate) that must be joined
                     to a larger `properties` file (many missing values, code columns, text columns,
                     an all-missing column); regression on a noisy target
They test that the machinery works, not how good a model is on the real data.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import run  # noqa: E402


# ------------------------------- synthetic data ------------------------------- #
def make_santander_like(path, n=2500, n_feat=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    logit = -2.6 + 1.2 * X[:, 0] + 0.9 * X[:, 1] - 0.8 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=[f"var_{i}" for i in range(n_feat)])
    df.insert(0, "target", y)
    df.insert(0, "ID_code", [f"train_{i}" for i in range(n)])
    df.to_csv(path, index=False)
    return df


def make_zillow_like(train_path, props_path, n=3000, extra=1500, seed=1):
    rng = np.random.default_rng(seed)
    total = n + extra
    ids = np.arange(10_000_000, 10_000_000 + total)
    sqft = rng.lognormal(7.3, 0.4, total)
    props = pd.DataFrame({
        "parcelid": ids,
        "calculatedfinishedsquarefeet": np.where(rng.random(total) < 0.10, np.nan, sqft),
        "bedroomcnt": rng.integers(1, 7, total).astype(float),
        "yearbuilt": np.where(rng.random(total) < 0.05, np.nan, rng.integers(1900, 2016, total).astype(float)),
        "poolcnt": np.where(rng.random(total) < 0.80, np.nan, 1.0),          # mostly missing
        "garagecarcnt": np.where(rng.random(total) < 0.65, np.nan, rng.integers(1, 4, total).astype(float)),
        "propertylandusetypeid": rng.choice([261, 262, 266, 269], total),     # numeric CODES
        "propertycountylandusecode": rng.choice(["0100", "0101", "010C", "1111", "122"], total),
        "hashottuborspa": np.where(rng.random(total) < 0.9, None, "True"),
        "fireplaceflag": np.nan,                                              # 100% missing
        "latitude": rng.normal(34_000_000, 400_000, total),
        "longitude": rng.normal(-118_000_000, 400_000, total),
    })
    sel = rng.choice(total, n, replace=False)
    sub = props.iloc[sel]
    signal = 0.0002 * (np.nan_to_num(sub["calculatedfinishedsquarefeet"].to_numpy(), nan=1500) - 1500) / 1.0
    logerror = 0.02 + 0.5 * signal + 0.08 * rng.normal(size=n)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    train = pd.DataFrame({"parcelid": sub["parcelid"].to_numpy(), "logerror": logerror,
                          "transactiondate": dates.strftime("%Y-%m-%d")})
    train.to_csv(train_path, index=False)
    props.to_csv(props_path, index=False)
    return train, props


# --------------------------------- helpers ---------------------------------- #
@pytest.fixture()
def env(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    import mlflow
    mlflow.set_tracking_uri(uri)
    return tmp_path


def pipeline(args, trials=3, final_epochs=3):
    return run.main([*args, "--trials", str(trials), "--trial-epochs", "2", "--final-epochs", str(final_epochs),
                     "--startup-trials", "2", "--bootstrap", "40"])


def runs_of(name):
    from mlflow import MlflowClient
    c = MlflowClient()
    exp = c.get_experiment_by_name(f"MLP-{name}")
    assert exp is not None, f"experiment MLP-{name} not found"
    runs = c.search_runs([exp.experiment_id], max_results=200)
    by_phase = {}
    for r in runs:
        by_phase.setdefault(r.data.tags.get("phase"), []).append(r)
    return c, by_phase


def fresh_process_predict(model_uri, df, tracking_uri):
    """Load the REGISTERED model in a new process and score `df`.

    mlflow.pyfunc enforces the logged signature strictly: a whole-number column that pandas holds as int64
    (e.g. from pd.read_csv) is rejected for a `double` input, so numeric columns are cast to float64 here.
    The tolerant paths (infer.py, Predictor.tabular_frame, `mlflow models serve`) do not need the cast.
    """
    code = ("import sys, json, pandas as pd, mlflow;"
            "df = pd.read_json(sys.stdin, orient='split');"
            "num = df.select_dtypes('number').columns; df[num] = df[num].astype('float64');"
            f"m = mlflow.pyfunc.load_model({model_uri!r});"
            "print('RESULT' + m.predict(df).to_json(orient='split'))")
    p = subprocess.run([sys.executable, "-c", code], cwd="/", input=df.to_json(orient="split"),
                       capture_output=True, text=True,
                       env={**os.environ, "MLFLOW_TRACKING_URI": tracking_uri})
    assert p.returncode == 0, p.stderr[-2000:]
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][0]
    return pd.read_json(io.StringIO(line[len("RESULT"):]), orient="split")


# ------------------------------ santander-like ------------------------------ #
def test_santander_like_classification_end_to_end(env):
    csv = env / "train.csv"
    df = make_santander_like(csv)
    res = pipeline(["--data", str(csv), "--task", "classification", "--target", "target", "--name", "santander",
                    "--source-opt", "id_cols=ID_code", "--min-auc", "0.5"])

    assert res["task"] == "classification"                       # auto-detected from a 0/1 target
    assert res["split"]["presplit"] is False and "NOT pre-split" in res["split"]["strategy"]
    assert res["audit"]["ok"]
    sizes = res["audit"]["sizes"]
    assert sum(sizes.values()) == len(df) and sizes["test"] == 500 and sizes["val"] == 500

    client, phases = runs_of("santander")
    assert len(phases["pipeline"]) == 1 and len(phases["search_trial"]) == 3 and len(phases["final_candidate"]) == 1
    m = phases["final_candidate"][0].data.metrics
    for k in ["test_auc_macro_ovr", "test_auc_se", "test_auc_ci95_low", "test_auc_ci95_high", "test_accuracy",
              "test_accuracy_se", "test_log_loss", "test_macro_f1", "val_auc"]:
        assert k in m, k
    assert m["test_auc_ci95_low"] <= m["test_auc_macro_ovr"] <= m["test_auc_ci95_high"]
    # NOTE: no accuracy claim here. 200 noisy features with 1,500 training rows and a 3-feature signal is a
    # regime where a regularised linear model beats an MLP; the pipeline reports that reference next to the MLP.
    assert 0.0 < m["test_auc_macro_ovr"] <= 1.0
    parent = phases["pipeline"][0]
    assert "ref_logistic_regression_test_auc" in parent.data.metrics   # reference model logged
    assert parent.data.tags["split_presplit"] == "false"

    # all 200 numeric columns were used; the ID column was not
    prev = client.get_model_version_by_alias("MLP-santander", "champion")
    assert res["n_params"] > 200 * 8, "input layer should see the 200 feature columns"
    assert prev.tags["validated"] == "true"

    # registered model takes RAW rows (original column names), incl. NaN and a column order change
    raw = df.drop(columns=["target", "ID_code"]).head(6).copy()
    raw.iloc[0, 0] = np.nan
    raw = raw[list(reversed(raw.columns))]
    out = fresh_process_predict("models:/MLP-santander@champion", raw, os.environ["MLFLOW_TRACKING_URI"])
    assert list(out.columns)[:2] == ["label", "confidence"] and {"prob_0", "prob_1"} <= set(out.columns)
    assert len(out) == 6 and np.allclose(out["prob_0"] + out["prob_1"], 1.0, atol=1e-5)

    # the ID column is not required at inference time (it is not a feature)
    assert "ID_code" not in raw.columns


def test_training_learns_a_planted_signal_on_csv_data(env):
    """A well-posed problem (few features, strong signal, enough epochs): AUC must be clearly above chance."""
    csv = env / "easy.csv"
    make_santander_like(csv, n=4000, n_feat=10, seed=11)
    res = pipeline(["--data", str(csv), "--task", "classification", "--target", "target", "--name", "easy",
                    "--source-opt", "id_cols=ID_code", "--no-register"], trials=0, final_epochs=25)
    auc = res["test_metrics"]["auc_macro_ovr"]
    lo, hi = res["test_metrics"]["auc_ci95_low"], res["test_metrics"]["auc_ci95_high"]
    assert auc > 0.75 and lo > 0.6, f"AUC {auc:.3f} (CI {lo:.3f}-{hi:.3f}) is not clearly above chance"


def test_santander_like_infer_cli_scores_a_csv(env):
    csv = env / "train.csv"
    df = make_santander_like(csv)
    pipeline(["--data", str(csv), "--task", "classification", "--target", "target", "--name", "santander",
              "--source-opt", "id_cols=ID_code", "--no-register"], trials=0)
    new = df.drop(columns=["target"]).head(8)
    new.to_csv(env / "new_rows.csv", index=False)           # extra ID column is simply ignored
    sys.argv = ["infer.py", "--model", "runs/santander/model.pt", "--csv", str(env / "new_rows.csv"),
                "--out-csv", str(env / "scored.csv")]
    import infer
    infer.main()
    scored = pd.read_csv(env / "scored.csv")
    assert len(scored) == 8 and {"label", "confidence", "prob_0", "prob_1", "out_of_range_warning"} <= set(scored.columns)
    assert "ID_code" in scored.columns                        # the caller's columns are kept next to predictions


def test_unreachable_auc_target_blocks_promotion(env):
    csv = env / "train.csv"
    make_santander_like(csv)
    pipeline(["--data", str(csv), "--task", "classification", "--target", "target", "--name", "santander",
              "--source-opt", "id_cols=ID_code", "--min-auc", "1.01"], trials=0)
    from mlflow import MlflowClient
    c = MlflowClient()
    v = c.get_model_version_by_alias("MLP-santander", "challenger")
    assert "acceptance target not met" in v.tags["promotion_decision"]
    with pytest.raises(Exception):
        c.get_model_version_by_alias("MLP-santander", "champion")


def test_min_auc_rejected_for_regression_csv(env):
    csv = env / "r.csv"
    pd.DataFrame({"a": np.arange(300) % 7, "b": np.arange(300) * 0.1, "y": np.arange(300) * 0.3}).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        run.main(["--data", str(csv), "--target", "y", "--name", "r", "--task", "regression", "--min-auc", "0.9"])


# -------------------------------- zillow-like -------------------------------- #
def zillow_args(env):
    return ["--data", str(env / "train_2016_v2.csv"), "--target", "logerror", "--name", "zillow", "--task", "regression",
            "--source-opt", f"join_csv={env / 'properties_2016.csv'}", "--source-opt", "join_on=parcelid",
            "--source-opt", "date_cols=transactiondate", "--source-opt", "categorical_cols=propertylandusetypeid"]


def test_zillow_like_join_regression_end_to_end(env):
    make_zillow_like(env / "train_2016_v2.csv", env / "properties_2016.csv")
    res = pipeline([*zillow_args(env), "--max-mse", "10"])

    assert res["task"] == "regression"
    assert sum(res["audit"]["sizes"].values()) == 3000          # the join kept exactly the train rows
    assert res["audit"]["ok"]

    client, phases = runs_of("zillow")
    m = phases["final_candidate"][0].data.metrics
    for k in ["test_mse", "test_mse_se", "test_mse_ci95_low", "test_rmse", "test_rmse_se", "test_mae", "test_r2",
              "test_mse_diff_vs_linear", "test_mse_diff_vs_linear_se"]:
        assert k in m, k
    assert m["test_mse_se"] > 0 and m["test_mse"] == pytest.approx(m["test_rmse"] ** 2, rel=1e-6)

    feat = client.download_artifacts(phases["pipeline"][0].info.run_id, "data/audit.json")
    assert Path(feat).exists()

    # reload the registered model in a fresh process and score RAW rows from the properties table
    props = pd.read_csv(env / "properties_2016.csv")
    raw = props.drop(columns=["parcelid"]).head(5).copy()
    raw["transactiondate"] = "2016-06-15"
    raw.loc[raw.index[0], "propertycountylandusecode"] = "NEVER-SEEN-LEVEL"   # unseen level -> all zeros
    raw.loc[raw.index[1], "calculatedfinishedsquarefeet"] = np.nan             # missing -> imputed
    out = fresh_process_predict("models:/MLP-zillow@champion", raw, os.environ["MLFLOW_TRACKING_URI"])
    assert list(out.columns)[0] == "prediction" and len(out) == 5 and np.isfinite(out["prediction"]).all()


def test_tolerant_scoring_paths_accept_integer_columns(env):
    """Whole-number columns (int64) must score fine through Predictor / infer.py, with NaNs and unseen levels."""
    make_zillow_like(env / "train_2016_v2.csv", env / "properties_2016.csv")
    pipeline(zillow_args(env), trials=0)
    props = pd.read_csv(env / "properties_2016.csv").drop(columns=["parcelid"]).head(6)
    props["transactiondate"] = "2016-06-15"
    props["bedroomcnt"] = props["bedroomcnt"].astype("int64")
    props["propertylandusetypeid"] = props["propertylandusetypeid"].astype("int64")
    assert props["bedroomcnt"].dtype == np.int64
    import infer
    p = infer.Predictor.from_registry("models:/MLP-zillow@champion")
    out = p.tabular_frame(props)
    assert len(out) == 6 and np.isfinite(out["prediction"]).all()
    # same rows through the CLI, as a CSV that really contains integers
    props.to_csv(env / "ints.csv", index=False)
    sys.argv = ["infer.py", "--model", "runs/zillow/model.pt", "--csv", str(env / "ints.csv"),
                "--out-csv", str(env / "ints_scored.csv")]
    infer.main()
    scored = pd.read_csv(env / "ints_scored.csv")
    assert np.allclose(scored["prediction"], out["prediction"], atol=1e-6)


def test_zillow_like_dates_expand_and_all_missing_column_is_reported(env):
    make_zillow_like(env / "tr.csv", env / "pr.csv")
    from tabular import load_csv_dataset
    splits, info = load_csv_dataset(path=str(env / "tr.csv"), join_path=str(env / "pr.csv"), join_on="parcelid",
                                    target="logerror", name="z", task="regression", date_cols=["transactiondate"],
                                    categorical_cols=["propertylandusetypeid"])
    names = info["preproc"]["feature_names_out"]
    assert {"transactiondate__year", "transactiondate__month", "transactiondate__dow"} <= set(names)
    assert not any(n.startswith("fireplaceflag") for n in names)
    assert any("fireplaceflag" in n and "entirely missing" in n for n in info["notes"])
    assert "poolcnt__missing" in names and "calculatedfinishedsquarefeet__missing" in names
    assert "parcelid" not in names and "parcelid" not in info["preproc"]["raw_inputs"]
    assert any(n.startswith("propertylandusetypeid=") for n in names)       # numeric codes -> one-hot


# ----------------------------- pre-split test file ----------------------------- #
def test_presplit_test_csv_end_to_end(env):
    full = make_santander_like(env / "full.csv", n=2400)
    full.iloc[:1800].to_csv(env / "tr.csv", index=False)
    full.iloc[1800:].to_csv(env / "te.csv", index=False)
    res = pipeline(["--data", str(env / "tr.csv"), "--task", "classification", "--target", "target", "--name", "ps",
                    "--source-opt", f"test_csv={env / 'te.csv'}", "--source-opt", "id_cols=ID_code"], trials=0)
    assert res["split"]["presplit"] is True
    sizes = res["audit"]["sizes"]
    assert sizes["test"] == 600 and sizes["train"] + sizes["val"] == 1800
    _, phases = runs_of("ps")
    assert phases["pipeline"][0].data.tags["split_presplit"] == "true"


# ---------------------------- what the loader refuses ---------------------------- #
def test_own_data_needs_target_and_task(env):
    csv = env / "d.csv"
    pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        run.main(["--data", str(csv)])                                   # no --task, no --target
    with pytest.raises(SystemExit):
        run.main(["--data", str(csv), "--task", "classification"])       # no --target


def test_two_csv_datasets_do_not_share_experiments_or_models(env):
    a, b = env / "a.csv", env / "b.csv"
    make_santander_like(a, n=900, n_feat=10, seed=3)
    make_santander_like(b, n=900, n_feat=10, seed=4)
    for name, p in (("alpha", a), ("beta", b)):
        pipeline(["--data", str(p), "--task", "classification", "--target", "target", "--name", name,
                  "--source-opt", "id_cols=ID_code"], trials=0)
    from mlflow import MlflowClient
    c = MlflowClient()
    assert c.get_registered_model("MLP-alpha") and c.get_registered_model("MLP-beta")
    assert c.get_experiment_by_name("MLP-alpha") and c.get_experiment_by_name("MLP-beta")
