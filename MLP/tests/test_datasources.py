"""
Tests for datasources.py: resolving --data, the options mechanism, the TableSource / DataSource
contracts, validation of what a source returns, plug-ins (from a file and from a module), the Parquet
source, and the result-folder behaviour of run.py. Synthetic data only; nothing is downloaded.
"""
import os
import sqlite3
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import datasources as ds  # noqa: E402
import run  # noqa: E402


# --------------------------------- helpers ---------------------------------- #
def frame(n=400, seed=0, classification=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "grp": rng.choice(["a", "b", "c"], n), "row_id": np.arange(n)})
    if classification:
        df["y"] = (df["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    else:
        df["y"] = 2 * df["f1"] - df["f2"] + 0.1 * rng.normal(size=n)
    return df


class MemorySource(ds.TableSource):
    """Minimal plug-in: serves a DataFrame it was given through a class attribute."""
    FRAME = None

    def read_tables(self):
        return ds.TableData(main=type(self).FRAME, label="memory")


@pytest.fixture()
def clean_env(tmp_path, monkeypatch):
    """No MLFLOW_TRACKING_URI: run.py must put the MLflow store inside the result folder."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


FAST = ["--trials", "2", "--trial-epochs", "2", "--final-epochs", "2", "--startup-trials", "2", "--bootstrap", "20"]


# ------------------------------ resolving --data ------------------------------ #
@pytest.mark.parametrize("name", ["california", "calif", "California", "housing", "california_housing"])
def test_california_aliases(name):
    s = ds.resolve_source(name)
    assert isinstance(s, ds.CaliforniaHousingSource) and s.task == "regression" and s.profile == "housing"
    assert s.default_name() == "housing"


def test_mnist_builtin():
    s = ds.resolve_source("mnist")
    assert isinstance(s, ds.MnistSource) and s.task == "classification" and s.profile == "mnist"


def test_builtin_task_and_target_are_checked():
    ds.resolve_source("california", task="regression", target="MedHouseVal")      # consistent: fine
    ds.resolve_source("mnist", task="auto")                                       # auto defers to the source
    with pytest.raises(ds.DataSourceError, match="regression dataset"):
        ds.resolve_source("california", task="classification")
    with pytest.raises(ds.DataSourceError, match="fixed target"):
        ds.resolve_source("mnist", target="digit_class")


def test_files_and_directories_pick_the_source(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "b.CSV.GZ").write_bytes(b"")
    (tmp_path / "c.parquet").write_bytes(b"")
    (tmp_path / "pq_dir").mkdir()
    (tmp_path / "notes.txt").write_text("hi")
    assert isinstance(ds.resolve_source(str(tmp_path / "a.csv"), task="regression", target="y"), ds.CsvSource)
    assert isinstance(ds.resolve_source(str(tmp_path / "b.CSV.GZ")), ds.CsvSource)
    assert isinstance(ds.resolve_source(str(tmp_path / "c.parquet")), ds.ParquetSource)
    assert isinstance(ds.resolve_source(str(tmp_path / "pq_dir")), ds.ParquetSource)
    with pytest.raises(ds.DataSourceError, match="supported files are"):
        ds.resolve_source(str(tmp_path / "notes.txt"))
    with pytest.raises(ds.DataSourceError, match="file not found"):
        ds.resolve_source(str(tmp_path / "missing.csv"))
    with pytest.raises(ds.DataSourceError, match="cannot interpret --data"):
        ds.resolve_source("no-such-thing")


def test_default_names_are_safe_and_never_collide_with_builtins(tmp_path):
    for fname, expect in [("train.csv", "train"), ("my data (v2).csv", "my_data_v2"),
                          ("sales.csv.gz", "sales"), ("housing.csv", "housing_data"), ("mnist.csv", "mnist_data")]:
        p = tmp_path / fname
        p.write_text("x,y\n1,2\n")
        s = ds.resolve_source(str(p))
        assert s.default_name() == expect, (fname, s.default_name())
        from tabular import validate_name
        validate_name(s.default_name())


# ---------------------------------- options ---------------------------------- #
def test_option_pairs():
    assert ds.parse_option_pairs(["a=1", "b=x,y", "c=k=v"]) == {"a": "1", "b": "x,y", "c": "k=v"}
    for bad in (["novalue"], ["=1"], ["a=1", "a=2"]):
        with pytest.raises(ds.DataSourceError):
            ds.parse_option_pairs(bad)


def test_option_coercion_uses_the_declared_types(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n")
    s = ds.resolve_source(str(p), options={"id_cols": "a, b ,c", "val_fraction": "0.25", "max_categories": "7",
                                           "join_on": "key"})
    assert s.options == {"id_cols": ["a", "b", "c"], "val_fraction": 0.25, "max_categories": 7, "join_on": "key"}
    with pytest.raises(ds.DataSourceError, match="expected int"):
        ds.resolve_source(str(p), options={"max_categories": "many"})
    assert ds._coerce("flag", "Yes", bool) is True and ds._coerce("flag", "0", bool) is False
    with pytest.raises(ds.DataSourceError, match="expected bool"):
        ds._coerce("flag", "maybe", bool)


def test_unknown_option_names_the_valid_ones(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n")
    with pytest.raises(ds.DataSourceError, match=r"no option 'idcols'.*id_cols"):
        ds.resolve_source(str(p), options={"idcols": "x"})
    with pytest.raises(ds.DataSourceError, match="valid options: none"):
        ds.resolve_source("mnist", options={"anything": "1"})


# ------------------------------ the TableSource contract ----------------------------- #
def test_table_source_builds_validated_data_from_a_dataframe():
    MemorySource.FRAME = frame()
    s = MemorySource(target="y", task="classification", id_cols=["row_id"])
    splits, info = s.load(seed=3)
    ds.validate_loaded(splits, info)
    assert {k: len(v) for k, v in splits.items()} == {"train": 240, "val": 80, "test": 80}
    assert info["task"] == "classification" and info["class_names"] == ["0", "1"]
    assert info["source"] == "memory" and info["split_info"]["presplit"] is False
    assert "row_id" not in info["preproc"]["raw_inputs"] and "y" not in info["preproc"]["raw_inputs"]
    assert any(c.startswith("grp") for c in info["preproc"]["feature_names_out"])   # text column one-hot encoded


def test_table_source_regression_target_is_standardised():
    MemorySource.FRAME = frame(classification=False)
    splits, info = MemorySource(target="y", task="regression", id_cols=["row_id"]).load(seed=1)
    ds.validate_loaded(splits, info)
    y = splits["train"].tensors[1]
    assert tuple(y.shape) == (240, 1) and abs(float(y.mean())) < 1e-4 and abs(float(y.std(unbiased=False)) - 1) < 1e-4


def test_table_source_needs_target_task_and_the_right_return_type():
    MemorySource.FRAME = frame()
    with pytest.raises(ds.DataSourceError, match="needs a target"):
        MemorySource(task="classification").load(0)
    with pytest.raises(ds.DataSourceError, match="needs --task"):
        MemorySource(target="y").load(0)

    class Wrong(ds.TableSource):
        def read_tables(self):
            return frame()  # forgot to wrap in TableData
    with pytest.raises(ds.DataSourceError, match="must return TableData"):
        Wrong(target="y", task="classification").load(0)


def test_table_source_passes_test_and_ignore_columns_through():
    class Pre(ds.TableSource):
        def read_tables(self):
            full = frame()
            return ds.TableData(main=full.iloc[:300], test=full.iloc[300:], label="pre",
                                notes=["served in two parts"], ignore=("row_id",))
    splits, info = Pre(target="y", task="classification").load(seed=0)
    assert len(splits["test"]) == 100 and info["split_info"]["presplit"] is True
    assert "row_id" not in info["preproc"]["raw_inputs"] and "served in two parts" in info["notes"]


def test_registered_name_resolves_to_a_custom_source():
    @ds.register_source("my_memory")
    class Reg(MemorySource):
        pass
    assert isinstance(ds.resolve_source("MY_MEMORY", task="classification", target="y"), Reg)
    assert "my_memory" in ds.registered_names()
    with pytest.raises(TypeError):
        ds.register_source("bad")(dict)


# ------------------------------------ plug-ins ------------------------------------ #
PLUGIN = textwrap.dedent('''
    import pandas as pd
    from datasources import TableData, TableSource

    class FileSource(TableSource):
        OPTIONS = {**TableSource.OPTIONS, "scale": float}
        def read_tables(self):
            n = 300
            df = pd.DataFrame({"a": range(n), "b": [i % 7 for i in range(n)]})
            df["y"] = (df["a"] * self.options.get("scale", 1.0)) % 2
            return TableData(main=df, label="plugin-file")

    class NotASource:
        pass
''')


def test_plugin_from_a_python_file(tmp_path):
    f = tmp_path / "my_plugin.py"
    f.write_text(PLUGIN)
    s = ds.resolve_source(f"{f}:FileSource", task="regression", target="y", options={"scale": "0.5"})
    assert type(s).__name__ == "FileSource" and s.options == {"scale": 0.5}
    splits, info = s.load(seed=0)
    ds.validate_loaded(splits, info)
    assert info["source"] == "plugin-file"


def test_plugin_from_an_importable_module(tmp_path, monkeypatch):
    (tmp_path / "plug_mod_xyz.py").write_text(PLUGIN)
    monkeypatch.chdir(tmp_path)
    s = ds.resolve_source("plug_mod_xyz:FileSource", task="regression", target="y")
    assert type(s).__name__ == "FileSource"


def test_plugin_errors_are_explained(tmp_path):
    f = tmp_path / "p.py"
    f.write_text(PLUGIN)
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
    with pytest.raises(ds.DataSourceError, match="has no class 'Nope'"):
        ds.resolve_source(f"{f}:Nope")
    with pytest.raises(ds.DataSourceError, match="not a DataSource subclass"):
        ds.resolve_source(f"{f}:NotASource")
    with pytest.raises(ds.DataSourceError, match="could not import plug-in.*boom"):
        ds.resolve_source(f"{tmp_path / 'broken.py'}:Anything")
    with pytest.raises(ds.DataSourceError, match="could not import plug-in"):
        ds.resolve_source("no_such_module_abc:Thing")
    with pytest.raises(ds.DataSourceError, match="plug-in file not found"):
        ds.resolve_source(f"{tmp_path / 'absent.py'}:Thing")


# ----------------------------------- validation ----------------------------------- #
def good():
    MemorySource.FRAME = frame()
    return MemorySource(target="y", task="classification", id_cols=["row_id"]).load(seed=0)


def broken(mutate, match):
    splits, info = good()
    mutate(splits, info)
    with pytest.raises(ds.DataSourceError, match=match):
        ds.validate_loaded(splits, info)


def test_validate_accepts_good_data_and_names_every_problem():
    splits, info = good()
    ds.validate_loaded(splits, info)
    X, y = splits["val"].tensors
    broken(lambda s, i: i.pop("preproc"), "missing the keys.*preproc")
    broken(lambda s, i: i.update(task="ranking"), "must be one of")
    broken(lambda s, i: i.update(in_dim=i["in_dim"] + 1), "columns but info\\['in_dim'\\]")
    broken(lambda s, i: i.update(out_dim=3), "class_names")
    broken(lambda s, i: s.pop("test"), "exactly the keys")
    broken(lambda s, i: s.update(val=TensorDataset(X.double().numpy().tolist() and X, y.float())), "int64")
    broken(lambda s, i: s.update(val=TensorDataset(X, y + 5)), "lie in 0\\.\\.1")
    broken(lambda s, i: s.update(val=TensorDataset(X.long(), y)), "float tensor")
    broken(lambda s, i: i["preproc"].update(type="unknown"), "'type' is one of")
    broken(lambda s, i: i.pop("example_input"), "example_input")
    broken(lambda s, i: i.update(split_info={"strategy": "x"}), "split_info")
    broken(lambda s, i: s.update(train=TensorDataset(X[:0], y[:0])), "empty")
    with pytest.raises(ds.DataSourceError, match="info is a dict"):
        ds.validate_loaded(splits, "not a dict")


def test_validate_regression_target_shape():
    MemorySource.FRAME = frame(classification=False)
    splits, info = MemorySource(target="y", task="regression", id_cols=["row_id"]).load(seed=0)
    ds.validate_loaded(splits, info)
    X, y = splits["val"].tensors
    splits["val"] = TensorDataset(X, y.view(-1))           # [n] instead of [n, 1]
    with pytest.raises(ds.DataSourceError, match=r"\[n, 1\]"):
        ds.validate_loaded(splits, info)


# ---------------------------------- Parquet source ---------------------------------- #
def test_parquet_source_reads_a_file_and_a_pre_split_test_file(tmp_path):
    df = frame(500)
    df.iloc[:400].to_parquet(tmp_path / "train.parquet")
    df.iloc[400:].to_parquet(tmp_path / "test.parquet")
    s = ds.resolve_source(str(tmp_path / "train.parquet"), task="classification", target="y",
                          options={"id_cols": "row_id", "test_file": str(tmp_path / "test.parquet")})
    assert isinstance(s, ds.ParquetSource) and s.default_name() == "train"
    splits, info = s.load(seed=0)
    ds.validate_loaded(splits, info)
    assert len(splits["test"]) == 100 and info["split_info"]["presplit"] is True
    assert info["lineage"]["main"]["sha256"] and info["lineage"]["test"]["sha256"]
    with pytest.raises(ds.DataSourceError, match="val_file needs test_file"):
        ds.resolve_source(str(tmp_path / "train.parquet"), task="classification", target="y",
                          options={"val_file": str(tmp_path / "test.parquet")}).load(0)


# ------------------------------- whole pipeline runs -------------------------------- #
def test_everything_lands_in_the_result_folder(clean_env):
    frame(600).to_parquet(clean_env / "d.parquet")
    out = clean_env / "my results" / "run1"
    res = run.main(["--data", str(clean_env / "d.parquet"), "--task", "classification", "--target", "y",
                    "--source-opt", "id_cols=row_id", "--out", str(out), *FAST])
    for name in ("model.pt", "results.json", "MODEL_CARD.md", "mlflow.db", "study.db"):
        assert (out / name).exists(), name
    assert any((out / "mlartifacts").rglob("*.png")), "MLflow artifacts should be inside the result folder"
    assert res["mlflow"]["tracking_uri"].startswith("sqlite:///") and str(out.resolve().as_posix()) in res["mlflow"]["tracking_uri"]
    assert not (clean_env / "mlflow.db").exists() and not (clean_env / "mlruns").exists(), "nothing in the working dir"
    assert res["dataset"] == "d" and "parquet:d.parquet" in (out / "MODEL_CARD.md").read_text()

    # the registry belongs to the folder: a second run into it adds version 2, another folder starts at 1
    run.main(["--data", str(clean_env / "d.parquet"), "--task", "classification", "--target", "y",
              "--source-opt", "id_cols=row_id", "--out", str(out), *FAST, "--no-register"][:-1] + ["--trials", "0"])
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').as_posix()}")
    assert sorted(int(v.version) for v in MlflowClient().search_model_versions("name='MLP-d'")) == [1, 2]
    other = clean_env / "run2"
    run.main(["--data", str(clean_env / "d.parquet"), "--task", "classification", "--target", "y",
              "--source-opt", "id_cols=row_id", "--out", str(other), *FAST, "--trials", "0"])
    mlflow.set_tracking_uri(f"sqlite:///{(other / 'mlflow.db').as_posix()}")
    assert sorted(int(v.version) for v in MlflowClient().search_model_versions("name='MLP-d'")) == [1]


def test_setup_does_not_leak_the_store_into_the_environment(clean_env):
    """Two runs in one process (notebook, loop) must each use their own result folder."""
    import tracking
    a = tracking.setup(None, "exp-a", default_dir=clean_env / "A")
    assert "MLFLOW_TRACKING_URI" not in os.environ
    b = tracking.setup(None, "exp-b", default_dir=clean_env / "B")
    assert a != b and b.endswith("B/mlflow.db") and (clean_env / "B" / "mlflow.db").exists()
    # a store the caller chose explicitly is respected, and left in the environment as it was
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(clean_env / 'shared.db').as_posix()}"
    c = tracking.setup(None, "exp-c", default_dir=clean_env / "C")
    assert c.endswith("shared.db") and os.environ["MLFLOW_TRACKING_URI"] == c
    assert not (clean_env / "C" / "mlflow.db").exists()


def test_default_result_folder_is_runs_name(clean_env):
    frame(500, classification=False).to_parquet(clean_env / "prices.parquet")
    run.main(["--data", str(clean_env / "prices.parquet"), "--task", "regression", "--target", "y",
              "--source-opt", "id_cols=row_id", *FAST, "--trials", "0"])
    assert (clean_env / "runs" / "prices" / "model.pt").exists()


def test_sqlite_example_plugin_end_to_end(clean_env):
    df = frame(700, classification=False)
    with sqlite3.connect(clean_env / "sales.db") as con:
        df.to_sql("houses", con, index=False)
    out = clean_env / "sql_out"
    res = run.main(["--data", f"{HERE / 'examples' / 'sqlite_source.py'}:SqliteSource", "--task", "regression",
                    "--target", "y", "--out", str(out), "--source-opt", f"db={clean_env / 'sales.db'}",
                    "--source-opt", "table=houses", "--source-opt", "id_cols=row_id", *FAST])
    assert res["task"] == "regression" and res["dataset"] == "houses"
    assert (out / "model.pt").exists() and res["test_metrics"]["mse_se"] > 0
    assert "sqlite:houses" in (out / "MODEL_CARD.md").read_text()
    # the packaged model scores raw rows through the tolerant path, with no knowledge of SQL
    import infer
    pred = infer.Predictor(str(out / "model.pt"))
    out_df = pred.tabular_frame(df.drop(columns=["y", "row_id"]).head(4))
    assert len(out_df) == 4 and np.isfinite(out_df["prediction"]).all()


# ------------------------------------- the CLI ------------------------------------- #
@pytest.mark.parametrize("argv", [
    ["--data", "california", "--task", "classification"],                     # task contradicts the dataset
    ["--data", "mnist", "--target", "something_else"],                        # fixed target
    ["--data", "definitely-not-a-source"],                                    # cannot interpret
    ["--data", "california", "--source-opt", "bogus=1"],                      # unknown option
    ["--data", "california", "--source-opt", "malformed"],                    # no '='
])
def test_cli_rejects_bad_data_arguments(argv, clean_env):
    with pytest.raises(SystemExit):
        run.main(argv)


def test_cli_explains_data_problems_without_a_traceback(clean_env):
    pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(clean_env / "tiny.csv", index=False)
    with pytest.raises(SystemExit) as e:                                      # target column does not exist
        run.main(["--data", str(clean_env / "tiny.csv"), "--task", "classification", "--target", "nope"])
    assert "target column 'nope'" in str(e.value)
    with pytest.raises(SystemExit) as e:                                      # own data without --task
        run.main(["--data", str(clean_env / "tiny.csv"), "--target", "y"])
    assert "--task" in str(e.value) or e.value.code == 2


def test_help_shows_the_four_core_arguments(capsys):
    with pytest.raises(SystemExit):
        run.main(["--help"])
    text = capsys.readouterr().out
    for flag in ("--data", "--task", "--target", "--out", "--source-opt"):
        assert flag in text
    assert text.index("what to run:") < text.index("advanced: hyperparameter search")
