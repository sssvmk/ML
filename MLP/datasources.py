"""
Pluggable data sources for the MLP pipeline.

The pipeline never reads data itself. It asks a DataSource for ready-to-train data:

    source = resolve_source("train.csv", task="classification", target="label", options={"id_cols": "ID"})
    splits, info = source.load(seed)

There are two levels at which you can plug something in.

1. TableSource (the easy one). Tabular data from ANYWHERE: a database, Parquet files, an API, a data
   lake. Implement ONE method, read_tables(), returning pandas DataFrames that contain a target column.
   The pipeline does everything else: drops rows with a missing target, splits train / validation / test
   (stratified for classification), fits the preprocessing on the TRAINING rows only (imputation,
   scaling, one-hot, dates), scores a linear/logistic reference model, and packages the preprocessing
   inside the registered model so it accepts raw rows.

       class SqlSource(TableSource):
           OPTIONS = {**TableSource.OPTIONS, "db": str, "table": str}     # extra --source-opt keys
           def read_tables(self):
               df = pd.read_sql_query(f"SELECT * FROM {self.options['table']}", connect(self.options["db"]))
               return TableData(main=df, label="sql:" + self.options["table"])

2. DataSource (full control). Implement load(seed) -> (splits, info) yourself, for data that is not a
   table (the built-in MNIST source does this). What you return is checked by validate_loaded(), which
   explains any problem in plain words. See DataSource.load for the contract.

How `--data` is turned into a source (resolve_source):
    california | mnist          built-in sources (see register_source for aliases)
    path/to/file.csv(.gz)       CsvSource
    path/to/file.parquet | dir  ParquetSource
    package.module:ClassName    your own DataSource / TableSource, imported from an installed module
    path/to/file.py:ClassName   ... or from a Python file

Extra settings for a source are passed as --source-opt KEY=VALUE and must be declared in the source's
OPTIONS dict (name -> type), so a typo is an error, not silently ignored.
"""
import abc
import importlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

TASKS = ("classification", "regression")
PROFILES = ("mnist", "housing", "tabular")       # which default hyperparameters / search ranges to use
PREPROC_TYPES = ("mnist", "housing", "tabular")  # input handling the packaged model knows how to apply
RESERVED_NAMES = {"mnist", "housing", "california", "calif", "california_housing"}


class DataSourceError(ValueError):
    """A data source is misconfigured, or returned data the pipeline cannot use."""


def safe_name(text, fallback="data"):
    """A short name usable for an MLflow experiment, a registered model and a folder."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return s or fallback


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
class DataSource(abc.ABC):
    """Anything the pipeline can train on. Subclass it and implement load().

    Class attributes a subclass may set:
      name          default label for the experiment / registered model / result folder
      profile       'mnist' | 'housing' | 'tabular': which default hyperparameters and search ranges to use
      fixed_task    'classification' or 'regression' if the data decide it (built-in sources do)
      fixed_target  the target's name if it is fixed
      OPTIONS       {option name: type}; the extra settings accepted via --source-opt (str, int, float, bool, list)
    """
    name = None
    profile = "tabular"
    fixed_task = None
    fixed_target = None
    OPTIONS = {}

    def __init__(self, location=None, *, target=None, task=None, data_dir="./data", **options):
        cls = type(self).__name__
        unknown = sorted(set(options) - set(self.OPTIONS))
        if unknown:
            raise DataSourceError(f"{cls} has no option(s) {unknown}; valid options: "
                                  f"{', '.join(sorted(self.OPTIONS)) or 'none'}")
        if task not in (None, "auto", *TASKS):
            raise DataSourceError(f"task must be classification, regression or auto (got {task!r})")
        if self.fixed_task and task not in (None, "auto", self.fixed_task):
            raise DataSourceError(f"{self.name or cls} is a {self.fixed_task} dataset; --task {task} does not apply")
        if self.fixed_target and target not in (None, self.fixed_target):
            raise DataSourceError(f"{self.name or cls} has a fixed target ({self.fixed_target!r}); "
                                  f"omit --target or pass --target {self.fixed_target}")
        self.location, self.data_dir, self.options = location, data_dir, dict(options)
        self.task = self.fixed_task or task
        self.target = self.fixed_target or target

    def default_name(self):
        return self.name or safe_name(type(self).__name__)

    def describe(self):
        """Short text recorded as the data source in MLflow and the model card."""
        return f"{type(self).__name__}({self.location})" if self.location else type(self).__name__

    @abc.abstractmethod
    def load(self, seed):
        """Return (splits, info). The contract (checked by validate_loaded):

        splits : {"train": ds, "val": ds, "test": ds}; each a torch TensorDataset (X, y)
                 X float tensor [n, in_dim]  (already preprocessed, model-ready)
                 y classification: int64 [n] with values 0..out_dim-1
                   regression    : float [n, 1], STANDARDIZED with the training mean/std (y_mean, y_std below)
        info   : dict with
                 task        'classification' | 'regression'
                 in_dim, out_dim   (regression: out_dim == 1)
                 y_mean, y_std     (classification: 0.0 and 1.0)
                 class_names list of str (classification) or None
                 baseline    dict of reference-model scores (may be empty)
                 preproc     dict whose "type" is one of PREPROC_TYPES: how raw inputs become X when the
                             packaged model is used. "tabular" (a fitted tabular.TabularPreprocessor
                             .to_dict()) is the one to use for your own data; it also needs
                             info["example_input"], a DataFrame of a few raw rows.
                 source      text describing where the data came from
                 split_info  {"presplit": bool, "strategy": text explaining how the splits were made}
                 optional: notes, lineage, target_label, money_scale, baseline_test_pred
        """


@dataclass
class TableData:
    """What TableSource.read_tables() returns."""
    main: pd.DataFrame                   # features + target column. If `test` is None it is split three ways.
    test: pd.DataFrame = None            # optional labelled test table, kept untouched
    val: pd.DataFrame = None             # optional validation table (needs `test`)
    label: str = "table"                 # recorded as the data source
    lineage: dict = field(default_factory=dict)   # e.g. {"main": {"path": ..., "sha256": ...}}
    notes: list = field(default_factory=list)     # facts worth recording (what was joined, filtered, ...)
    ignore: tuple = ()                   # columns that must not become features (e.g. a join key)


class TableSource(DataSource):
    """Tabular data from anywhere. Implement read_tables(); the pipeline handles the rest."""
    profile = "tabular"
    OPTIONS = {"id_cols": list, "drop_cols": list, "date_cols": list, "categorical_cols": list,
               "val_fraction": float, "test_fraction": float, "max_categories": int}

    @abc.abstractmethod
    def read_tables(self) -> TableData:
        """Return the rows as DataFrames. No splitting or preprocessing here."""

    def default_name(self):
        n = super().default_name()
        return n + "_data" if n.lower() in RESERVED_NAMES else n  # never collide with a built-in dataset

    def load(self, seed):
        from tabular import build_dataset_from_tables

        cls = type(self).__name__
        if not self.target:
            raise DataSourceError(f"{cls} needs a target column (--target)")
        if self.task is None:
            raise DataSourceError(f"{cls} needs --task classification or --task regression (or --task auto)")
        t = self.read_tables()
        if not isinstance(t, TableData) or not isinstance(t.main, pd.DataFrame):
            raise DataSourceError(f"{cls}.read_tables() must return TableData(main=<DataFrame>, ...)")
        o = self.options
        return build_dataset_from_tables(
            t.main, target=self.target, name=safe_name(self.default_name()), source_label=t.label or self.describe(),
            task=self.task, test=t.test, val=t.val, id_cols=o.get("id_cols", ()), drop_cols=o.get("drop_cols", ()),
            date_cols=o.get("date_cols", ()), categorical_cols=o.get("categorical_cols", ()), ignore_extra=t.ignore,
            val_fraction=o.get("val_fraction", 0.2), test_fraction=o.get("test_fraction", 0.2),
            max_categories=o.get("max_categories", 20), seed=seed, lineage=t.lineage, notes=t.notes)


# --------------------------------------------------------------------------- #
# Built-in sources
# --------------------------------------------------------------------------- #
class MnistSource(DataSource):
    """MNIST handwritten digits (torchvision). Pre-split: official 60k train / 10k test; a validation
    set is carved from train."""
    name, profile = "mnist", "mnist"
    fixed_task, fixed_target = "classification", "label"

    def load(self, seed):
        import mlp_core  # looked up at call time so tests can substitute the loader
        return mlp_core._load_mnist(self.data_dir, seed)


class CaliforniaHousingSource(DataSource):
    """California Housing (scikit-learn): ONE table of 20,640 rows, split 60/20/20 by the pipeline."""
    name, profile = "housing", "housing"
    fixed_task, fixed_target = "regression", "MedHouseVal"

    def load(self, seed):
        import mlp_core
        return mlp_core._load_housing(seed)


def _strip_suffixes(filename, suffixes):
    low = filename.lower()
    changed = True
    while changed:
        changed = False
        for s in suffixes:
            if low.endswith(s):
                filename, low, changed = filename[:-len(s)], low[:-len(s)], True
    return filename


class CsvSource(TableSource):
    """A CSV file (optionally gzip-compressed), optionally joined to a second CSV."""
    OPTIONS = {**TableSource.OPTIONS, "test_csv": str, "val_csv": str, "join_csv": str, "join_on": str}

    def default_name(self):
        n = safe_name(_strip_suffixes(Path(self.location or "csv").name, (".gz", ".csv")))
        return n + "_data" if n.lower() in RESERVED_NAMES else n

    def describe(self):
        return f"csv:{Path(self.location).name}" if self.location else "csv"

    def read_tables(self):
        from tabular import read_csv_tables
        if not self.location:
            raise DataSourceError("CsvSource needs a file path")
        o = self.options
        r = read_csv_tables(path=self.location, val_path=o.get("val_csv"), test_path=o.get("test_csv"),
                            join_path=o.get("join_csv"), join_on=o.get("join_on"))
        return TableData(main=r["main"], test=r["test"], val=r["val"], label=self.describe(),
                         lineage=r["lineage"], notes=r["notes"], ignore=r["ignore_extra"])


class ParquetSource(TableSource):
    """A Parquet file or a directory of Parquet files (needs pyarrow or fastparquet)."""
    OPTIONS = {**TableSource.OPTIONS, "test_file": str, "val_file": str}

    def default_name(self):
        n = safe_name(_strip_suffixes(Path(self.location or "parquet").name, (".parquet", ".pq")))
        return n + "_data" if n.lower() in RESERVED_NAMES else n

    def describe(self):
        return f"parquet:{Path(self.location).name}" if self.location else "parquet"

    def read_tables(self):
        from tabular import file_sha256
        if not self.location or not os.path.exists(self.location):
            raise FileNotFoundError(f"Parquet path not found: {self.location}")
        o = self.options
        if o.get("val_file") and not o.get("test_file"):
            raise DataSourceError("val_file needs test_file as well (or give neither and let the pipeline split)")

        def read(p):
            return pd.read_parquet(p)

        def lin(p):
            return {"path": os.path.abspath(p), **({"sha256": file_sha256(p)} if os.path.isfile(p) else {})}

        lineage = {"main": lin(self.location)}
        test = val = None
        if o.get("test_file"):
            test, lineage["test"] = read(o["test_file"]), lin(o["test_file"])
        if o.get("val_file"):
            val, lineage["val"] = read(o["val_file"]), lin(o["val_file"])
        return TableData(main=read(self.location), test=test, val=val, label=self.describe(), lineage=lineage)


# --------------------------------------------------------------------------- #
# Registry and resolution of --data
# --------------------------------------------------------------------------- #
_REGISTRY = {}


def register_source(*names):
    """Class decorator: make a source available under one or more names, e.g. `--data mysource`."""
    def deco(cls):
        if not (isinstance(cls, type) and issubclass(cls, DataSource)):
            raise TypeError("register_source expects a DataSource subclass")
        for n in names:
            _REGISTRY[n.lower()] = cls
        return cls
    return deco


register_source("mnist")(MnistSource)
register_source("california", "calif", "california_housing", "housing")(CaliforniaHousingSource)

_FILE_SOURCES = ((".csv", CsvSource), (".csv.gz", CsvSource), (".parquet", ParquetSource), (".pq", ParquetSource))


def registered_names():
    return sorted(_REGISTRY)


def _load_plugin_class(spec):
    """'package.module:Class' or 'path/to/file.py:Class' -> the class, or None if `spec` is not of that form."""
    module_part, _, cls_name = str(spec).rpartition(":")
    if not module_part or not cls_name.isidentifier():
        return None
    try:
        if module_part.endswith(".py") or os.path.isfile(module_part):
            path = Path(module_part).resolve()
            if not path.is_file():
                raise DataSourceError(f"plug-in file not found: {module_part}")
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            mod_name = f"mlp_plugin_{safe_name(path.stem)}"
            mspec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(mspec)
            sys.modules[mod_name] = module
            mspec.loader.exec_module(module)
        else:
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
            module = importlib.import_module(module_part)
    except DataSourceError:
        raise
    except Exception as e:
        raise DataSourceError(f"could not import plug-in {module_part!r}: {type(e).__name__}: {e}") from e
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise DataSourceError(f"plug-in module {module_part!r} has no class {cls_name!r}")
    if not (isinstance(cls, type) and issubclass(cls, DataSource)):
        raise DataSourceError(f"{spec!r} is not a DataSource subclass (subclass datasources.DataSource or TableSource)")
    return cls


def find_source_class(data):
    """--data value -> (source class, location or None)."""
    key = str(data).lower()
    if key in _REGISTRY:
        return _REGISTRY[key], None
    if os.path.isdir(data):
        return ParquetSource, data
    low = key
    for suffix, cls in _FILE_SOURCES:
        if low.endswith(suffix):
            if not os.path.exists(data):
                raise DataSourceError(f"file not found: {data}")
            return cls, data
    if os.path.isfile(data):
        raise DataSourceError(f"don't know how to read {data!r}: supported files are .csv, .csv.gz, .parquet "
                              f"(or plug in your own source with --data module:Class)")
    cls = _load_plugin_class(data)
    if cls is not None:
        return cls, None
    raise DataSourceError(
        f"cannot interpret --data {data!r}. Use a built-in name ({', '.join(registered_names())}), "
        f"a path to a .csv / .csv.gz / .parquet file or directory, or a plug-in 'module:Class' / 'file.py:Class'.")


def parse_option_pairs(pairs):
    """['a=1', 'b=x,y'] -> {'a': '1', 'b': 'x,y'} (rejects a missing '=' and duplicate keys)."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise DataSourceError(f"--source-opt expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise DataSourceError(f"--source-opt expects KEY=VALUE, got {p!r}")
        if k in out:
            raise DataSourceError(f"--source-opt {k!r} was given twice")
        out[k] = v
    return out


def _coerce(key, value, typ):
    if not isinstance(value, str):
        return value  # already typed (programmatic use)
    try:
        if typ is list:
            return [s.strip() for s in value.split(",") if s.strip()]
        if typ is bool:
            v = value.strip().lower()
            if v in ("1", "true", "yes", "y"):
                return True
            if v in ("0", "false", "no", "n"):
                return False
            raise ValueError("expected true or false")
        if typ in (int, float):
            return typ(value)
        return value
    except ValueError as e:
        raise DataSourceError(f"option {key}={value!r}: expected {getattr(typ, '__name__', typ)} ({e})") from e


def coerce_options(cls, options):
    """Check option names against cls.OPTIONS and convert the strings to their declared types."""
    out = {}
    for k, v in (options or {}).items():
        if k not in cls.OPTIONS:
            raise DataSourceError(f"{cls.__name__} has no option {k!r}; valid options: "
                                  f"{', '.join(sorted(cls.OPTIONS)) or 'none'}")
        out[k] = _coerce(k, v, cls.OPTIONS[k])
    return out


def resolve_source(data, *, task=None, target=None, options=None, data_dir="./data"):
    """Turn the --data value (plus --task, --target, --source-opt) into a configured DataSource."""
    cls, location = find_source_class(data)
    return cls(location, target=target, task=task, data_dir=data_dir, **coerce_options(cls, options))


# --------------------------------------------------------------------------- #
# Checking what a source returned
# --------------------------------------------------------------------------- #
def validate_loaded(splits, info, who="the data source"):
    """Raise DataSourceError listing everything wrong with (splits, info); cheap shape / dtype checks only."""
    if not isinstance(info, dict):
        raise DataSourceError(f"{who} must return (splits, info) where info is a dict")
    problems = []
    need = ("task", "in_dim", "out_dim", "y_mean", "y_std", "baseline", "class_names", "preproc", "source",
            "split_info")
    missing = [k for k in need if k not in info]
    if missing:
        problems.append(f"info is missing the keys {missing}")
    task, out_dim, in_dim = info.get("task"), info.get("out_dim"), info.get("in_dim")
    if task not in TASKS:
        problems.append(f"info['task'] must be one of {TASKS}, got {task!r}")
    if not isinstance(in_dim, int) or in_dim < 1:
        problems.append(f"info['in_dim'] must be a positive int, got {in_dim!r}")
    if task == "classification":
        names = info.get("class_names")
        if not isinstance(out_dim, int) or out_dim < 2:
            problems.append(f"classification needs info['out_dim'] >= 2, got {out_dim!r}")
        if not isinstance(names, (list, tuple)) or (isinstance(out_dim, int) and len(names) != out_dim):
            problems.append("info['class_names'] must be a list with one name per class")
    elif task == "regression":
        if out_dim != 1:
            problems.append(f"regression needs info['out_dim'] == 1, got {out_dim!r}")
    pre = info.get("preproc")
    if not isinstance(pre, dict) or pre.get("type") not in PREPROC_TYPES:
        problems.append(f"info['preproc'] must be a dict whose 'type' is one of {PREPROC_TYPES} (how raw inputs "
                        f"become model inputs when the packaged model is used)")
    elif pre["type"] == "tabular" and not isinstance(info.get("example_input"), pd.DataFrame):
        problems.append("a 'tabular' preproc needs info['example_input']: a DataFrame of a few raw input rows")
    si = info.get("split_info")
    if not isinstance(si, dict) or "presplit" not in si or not isinstance(si.get("strategy"), str):
        problems.append("info['split_info'] must be {'presplit': bool, 'strategy': <text explaining the split>}")
    if not isinstance(info.get("baseline"), dict):
        problems.append("info['baseline'] must be a dict (it may be empty)")

    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        problems.append("splits must be a dict with exactly the keys 'train', 'val' and 'test'")
    else:
        for name, ds in splits.items():
            if not isinstance(ds, TensorDataset) or len(ds.tensors) != 2:
                problems.append(f"split {name!r} must be a torch TensorDataset(X, y)")
                continue
            X, y = ds.tensors
            n = len(ds)
            if n == 0:
                problems.append(f"split {name!r} is empty")
            if not torch.is_floating_point(X) or X.dim() != 2:
                problems.append(f"split {name!r}: X must be a float tensor [n, in_dim], got {X.dtype} {tuple(X.shape)}")
            elif isinstance(in_dim, int) and X.shape[1] != in_dim:
                problems.append(f"split {name!r}: X has {X.shape[1]} columns but info['in_dim'] is {in_dim}")
            if task == "classification":
                if y.dtype != torch.int64 or y.dim() != 1:
                    problems.append(f"split {name!r}: y must be an int64 tensor [n], got {y.dtype} {tuple(y.shape)}")
                elif n and isinstance(out_dim, int) and (int(y.min()) < 0 or int(y.max()) >= out_dim):
                    problems.append(f"split {name!r}: class labels must lie in 0..{out_dim - 1}")
            elif task == "regression":
                if not torch.is_floating_point(y) or tuple(y.shape) != (n, 1):
                    problems.append(f"split {name!r}: y must be a float tensor [n, 1] (standardized), "
                                    f"got {y.dtype} {tuple(y.shape)}")
    if problems:
        raise DataSourceError(f"{who} returned data the pipeline cannot use:\n  - " + "\n  - ".join(problems))
