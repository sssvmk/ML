"""Unit tests for tabular.py: the CSV preprocessor and loader (no MLflow, no downloads)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tabular  # noqa: E402
from tabular import TabularPreprocessor  # noqa: E402


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(10, 2, n),
        "b": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
        "color": rng.choice(["red", "green", "blue"], n),
        "flag": rng.choice([True, False], n),
        "code": rng.choice([1, 2, 3], n),               # numeric codes
        "when": rng.choice(["2016-01-15", "2016-06-30", "2016-11-02"], n),
        "allnan": np.nan,
        "ident": [f"id{i}" for i in range(n)],
    })


# ------------------------------- preprocessor ------------------------------- #
def test_numeric_imputation_scaling_and_indicator():
    df = frame()
    p = TabularPreprocessor(date_cols=["when"]).fit(df)
    X = p.transform(df)
    assert "a" in p.num_cols and "b" in p.num_cols
    assert "b" in p.ind_cols and "a" not in p.ind_cols           # only b had missing values
    ja = p.feature_names_out.index("a")
    assert X[:, ja].mean() == pytest.approx(0.0, abs=1e-5) and X[:, ja].std() == pytest.approx(1.0, abs=1e-3)
    jb = p.feature_names_out.index("b__missing")
    assert X[:, jb].sum() == df["b"].isna().sum()                # indicator marks exactly the missing rows
    assert np.isfinite(X).all()


def test_drops_allnan_and_id_like_columns_with_notes():
    p = TabularPreprocessor().fit(frame())
    joined = " | ".join(p.notes)
    assert "allnan" in joined and "ident" in joined
    assert "allnan" not in p.raw_inputs and "ident" not in p.raw_inputs


def test_dates_expand_to_year_month_dow():
    p = TabularPreprocessor(date_cols=["when"]).fit(frame())
    assert {"when__year", "when__month", "when__dow"} <= set(p.num_cols)
    assert "when" in p.raw_inputs


def test_onehot_levels_missing_level_and_unseen_level():
    df = frame()
    df.loc[:9, "color"] = None
    p = TabularPreprocessor(date_cols=["when"]).fit(df)
    assert {"color=red", "color=green", "color=blue", f"color={tabular.MISSING}"} <= set(p.feature_names_out)
    new = df.head(3).copy()
    new["color"] = ["purple", "red", None]                       # unseen level, known, missing
    X = p.transform(new)
    cols = [i for i, n in enumerate(p.feature_names_out) if n.startswith("color=")]
    assert X[0, cols].sum() == 0                                 # unseen -> all zeros, no crash
    assert X[1, cols].sum() == 1
    assert X[2, p.feature_names_out.index(f"color={tabular.MISSING}")] == 1


def test_numeric_codes_stay_numeric_unless_forced_categorical():
    df = frame()
    assert "code" in TabularPreprocessor(date_cols=["when"]).fit(df).num_cols
    p = TabularPreprocessor(date_cols=["when"], categorical_cols=["code"]).fit(df)
    assert "code" in p.cat_cols and "code" not in p.num_cols
    assert {"code=1", "code=2", "code=3"} <= set(p.feature_names_out)


def test_code_levels_encode_identically_with_and_without_missing_values():
    """1, 1.0 must map to the same level, or a code column would silently change meaning at inference."""
    df = frame()
    df.loc[0, "code"] = np.nan                                   # makes the training column float64
    p = TabularPreprocessor(date_cols=["when"], categorical_cols=["code"]).fit(df)
    ints = df.head(5).copy()
    ints["code"] = [1, 2, 3, 1, 2]                               # arrives as int64 at inference
    floats = ints.copy()
    floats["code"] = floats["code"].astype(float)
    assert np.array_equal(p.transform(ints), p.transform(floats))
    assert p.transform(ints)[:, p.feature_names_out.index("code=1")].sum() == 2


def test_fit_uses_training_rows_only():
    train, other = frame(seed=1), frame(seed=2)
    other["a"] = other["a"] + 1000.0                             # wildly different distribution
    p = TabularPreprocessor(date_cols=["when"]).fit(train)
    X = p.transform(other)
    assert X[:, p.feature_names_out.index("a")].mean() > 100     # stats were NOT refit on `other`


def test_missing_raw_column_is_a_clear_error():
    p = TabularPreprocessor(date_cols=["when"]).fit(frame())
    with pytest.raises(ValueError, match="missing input columns"):
        p.transform(frame().drop(columns=["color"]))


def test_extra_columns_ignored_and_inf_treated_as_missing():
    df = frame()
    p = TabularPreprocessor(date_cols=["when"]).fit(df)
    new = df.head(3).copy()
    new["junk"] = 5
    new.loc[0, "a"] = np.inf
    X = p.transform(new)
    assert np.isfinite(X).all()
    assert X[0, p.feature_names_out.index("a")] == pytest.approx(
        (p.num_median[p.num_cols.index("a")] - p.num_mean[p.num_cols.index("a")]) / p.num_scale[p.num_cols.index("a")])


def test_state_roundtrips_through_torch_save_weights_only(tmp_path):
    df = frame()
    p = TabularPreprocessor(date_cols=["when"], categorical_cols=["code"]).fit(df)
    f = tmp_path / "p.pt"
    torch.save({"preproc": p.to_dict()}, f)
    back = TabularPreprocessor.from_dict(torch.load(f, weights_only=True)["preproc"])
    assert np.array_equal(p.transform(df), back.transform(df))
    assert back.feature_names_out == p.feature_names_out


def test_example_frame_has_stable_dtypes():
    df = frame()
    p = TabularPreprocessor(date_cols=["when"]).fit(df)
    ex = p.example_frame(df, 5)
    assert len(ex) == 5 and not ex.isna().any().any()
    assert ex["a"].dtype == "float64" and ex["b"].dtype == "float64"
    assert pd.api.types.is_string_dtype(ex["color"]) and pd.api.types.is_string_dtype(ex["when"])


# ---------------------------------- loader ---------------------------------- #
def write_clf(path, n=900, seed=0, with_id=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.2 * X[:, 0])))).astype(int)
    df = pd.DataFrame(X, columns=[f"var_{i}" for i in range(6)])
    df.insert(0, "target", y)
    if with_id:
        df.insert(0, "ID_code", [f"train_{i}" for i in range(n)])
    df.to_csv(path, index=False)
    return df


def test_single_csv_is_split_three_ways_stratified(tmp_path):
    write_clf(tmp_path / "d.csv")
    splits, info = tabular.load_csv_dataset(path=str(tmp_path / "d.csv"), target="target", name="t",
                                            id_cols=["ID_code"], seed=3)
    assert info["task"] == "classification" and info["class_names"] == ["0", "1"] and info["out_dim"] == 2
    assert {k: len(v) for k, v in splits.items()} == {"train": 540, "val": 180, "test": 180}
    assert info["split_info"]["presplit"] is False and "NOT pre-split" in info["split_info"]["strategy"]
    rates = [splits[k].tensors[1].float().mean().item() for k in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.02                         # stratified
    assert info["in_dim"] == 6 and "logistic_regression_test_auc" in info["baseline"]
    assert info["baseline"]["logistic_regression_test_auc"] > 0.6  # the reference model learned something
    assert info["lineage"]["main_csv"]["sha256"]


def test_id_like_column_is_dropped_automatically_even_without_id_cols(tmp_path):
    write_clf(tmp_path / "d.csv")
    _, info = tabular.load_csv_dataset(path=str(tmp_path / "d.csv"), target="target", name="t")
    assert info["in_dim"] == 6 and any("ID_code" in n for n in info["notes"])


def test_separate_test_csv_is_kept_untouched_and_validation_carved(tmp_path):
    write_clf(tmp_path / "tr.csv", n=800, seed=1)
    te = write_clf(tmp_path / "te.csv", n=300, seed=2)
    splits, info = tabular.load_csv_dataset(path=str(tmp_path / "tr.csv"), target="target", name="t",
                                            id_cols=["ID_code"], test_path=str(tmp_path / "te.csv"))
    assert len(splits["test"]) == 300 and len(splits["val"]) == 160 and len(splits["train"]) == 640
    assert info["split_info"]["presplit"] is True
    assert splits["test"].tensors[1].tolist() == te["target"].tolist()   # row order and labels unchanged


def test_three_files_used_as_given(tmp_path):
    write_clf(tmp_path / "tr.csv", n=500, seed=1)
    write_clf(tmp_path / "va.csv", n=100, seed=2)
    write_clf(tmp_path / "te.csv", n=120, seed=3)
    splits, info = tabular.load_csv_dataset(path=str(tmp_path / "tr.csv"), target="target", name="t",
                                            id_cols=["ID_code"], val_path=str(tmp_path / "va.csv"),
                                            test_path=str(tmp_path / "te.csv"))
    assert {k: len(v) for k, v in splits.items()} == {"train": 500, "val": 100, "test": 120}
    assert "nothing was re-split" in info["split_info"]["strategy"]


def test_regression_target_is_standardised_with_train_stats(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=500), "x2": rng.normal(size=500)})
    df["y"] = 50 + 10 * df["x1"] + rng.normal(size=500)
    df.to_csv(tmp_path / "r.csv", index=False)
    splits, info = tabular.load_csv_dataset(path=str(tmp_path / "r.csv"), target="y", name="r", task="regression")
    assert info["task"] == "regression" and info["out_dim"] == 1
    ytr = splits["train"].tensors[1].numpy()
    assert ytr.mean() == pytest.approx(0, abs=1e-5) and ytr.std() == pytest.approx(1, abs=1e-3)
    assert 45 < info["y_mean"] < 55 and info["y_std"] > 5
    assert len(info["baseline_test_pred"]) == len(splits["test"])        # aligned with the test rows
    assert info["baseline"]["linear_regression_test_mse"] < info["baseline"]["predict_mean_test_rmse"] ** 2


def test_join_filters_to_main_keys_and_validates(tmp_path):
    rng = np.random.default_rng(0)
    props = pd.DataFrame({"pid": np.arange(1000), "sqft": rng.normal(size=1000),
                          "kind": rng.choice(["a", "b", "c"], 1000)})
    main = pd.DataFrame({"pid": rng.choice(1000, 400, replace=False)})
    main["y"] = props.set_index("pid").loc[main.pid, "sqft"].to_numpy() * 2 + rng.normal(0, 0.1, 400)
    props.to_csv(tmp_path / "p.csv", index=False)
    main.to_csv(tmp_path / "m.csv", index=False)
    _, info = tabular.load_csv_dataset(path=str(tmp_path / "m.csv"), target="y", name="j", task="regression",
                                       join_path=str(tmp_path / "p.csv"), join_on="pid")
    assert any("400 after the inner join" in n for n in info["notes"])
    assert info["in_dim"] == 1 + 3                                      # sqft + one-hot kind; pid ignored
    dup = pd.concat([props, props.head(3)])
    dup.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        tabular.load_csv_dataset(path=str(tmp_path / "m.csv"), target="y", name="j",
                                 join_path=str(tmp_path / "dup.csv"), join_on="pid")


@pytest.mark.parametrize("kwargs,msg", [
    (dict(target="nope"), "target column 'nope'"),
    (dict(target="target", name="bad name!"), "must start with a letter"),
    (dict(target="target", task="regression"), "numeric target"),     # string labels
    (dict(target="target", id_cols=["ghost"]), "'ghost'"),
    (dict(target="target", val_path="x.csv"), "needs test_csv"),
])
def test_clear_errors(tmp_path, kwargs, msg):
    df = write_clf(tmp_path / "d.csv", with_id=False)
    if msg == "numeric target":
        df["target"] = df["target"].map({0: "no", 1: "yes"})
        df.to_csv(tmp_path / "d.csv", index=False)
    args = dict(path=str(tmp_path / "d.csv"), name="t")
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        tabular.load_csv_dataset(**args)


def test_missing_target_rows_are_dropped_and_reported(tmp_path):
    df = write_clf(tmp_path / "d.csv", n=500, with_id=False).astype({"target": "float"})
    df.loc[:19, "target"] = np.nan
    df.to_csv(tmp_path / "d.csv", index=False)
    splits, info = tabular.load_csv_dataset(path=str(tmp_path / "d.csv"), target="target", name="t")
    assert sum(len(v) for v in splits.values()) == 480
    assert any("dropped 20 rows with a missing target" in n for n in info["notes"])


def test_single_class_and_missing_file_errors(tmp_path):
    df = write_clf(tmp_path / "d.csv", with_id=False)
    df["target"] = 1
    df.to_csv(tmp_path / "one.csv", index=False)
    with pytest.raises(ValueError, match="at least 2 classes"):
        tabular.load_csv_dataset(path=str(tmp_path / "one.csv"), target="target", name="t")
    with pytest.raises(FileNotFoundError):
        tabular.load_csv_dataset(path=str(tmp_path / "missing.csv"), target="target", name="t")


def test_task_detection():
    assert tabular.detect_task(pd.Series([0, 1, 0, 1])) == "classification"
    assert tabular.detect_task(pd.Series(["a", "b"])) == "classification"
    assert tabular.detect_task(pd.Series(np.linspace(0, 1, 100))) == "regression"
    assert tabular.detect_task(pd.Series(np.arange(500) * 1000)) == "regression"   # integer prices


def test_numeric_code_columns_stay_numeric_in_the_example_and_signature():
    """A forced-categorical column that callers send as numbers must be a double in the example,
    otherwise MLflow's schema enforcement rejects the natural input (found in the end-to-end run)."""
    df = frame()
    df.loc[:3, "code"] = np.nan
    p = TabularPreprocessor(date_cols=["when"], categorical_cols=["code"]).fit(df)
    assert p.cat_numeric == ["code"]
    ex = p.example_frame(df, 6)
    assert ex["code"].dtype == "float64" and not ex["code"].isna().any()
    assert pd.api.types.is_string_dtype(ex["color"])             # real text stays text
    back = TabularPreprocessor.from_dict(p.to_dict())
    assert back.cat_numeric == ["code"]
