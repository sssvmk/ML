"""
Generic CSV datasets for the MLP pipeline (classification or regression).

    python run.py --csv-path train.csv --target target --name santander --task classification

What happens
  1. read the CSV (optionally join a second CSV on a key); drop rows whose target is missing
  2. decide the task (--task, or auto-detect) and encode the target
  3. split: ONE file    -> train / validation / test (stratified for classification)
            + --test-csv -> that file is the test set; validation is carved from the main file
            + --val-csv  -> both files used as given (nothing is split)
  4. fit a preprocessor on the TRAINING rows only (median imputation + missing-value
     indicators, standardisation, one-hot for text columns, date expansion), then apply it
     to every split
  5. fit a simple reference model (linear / logistic regression) for comparison

The fitted preprocessor is stored inside the model bundle, so the packaged model accepts RAW
rows (original column names) and applies exactly the training-time preprocessing.

This module needs only numpy, pandas and torch (scikit-learn and splits are imported only
when a dataset is loaded for training), so it is safe to ship inside the MLflow model.
"""
import hashlib
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

MISSING = "__missing__"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(name):
    if not name or not _SAFE_NAME.match(name):
        raise ValueError(f"--name {name!r} must start with a letter or digit and contain only letters, "
                         f"digits, '.', '_' and '-' (it names the MLflow experiment, the registered "
                         f"model and the output folder)")
    return name


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Preprocessor (fit on training rows only; state is plain python for torch.save)
# --------------------------------------------------------------------------- #
class TabularPreprocessor:
    """Raw DataFrame -> float32 matrix.

    Output columns, in order:  standardised numeric  |  missing-value indicators  |  one-hot text
      numeric : coerced to numbers (unparseable / +-inf -> missing), missing -> training median,
                then (x - mean) / std. A 0/1 indicator is added for numeric columns that had
                any missing value in the training rows.
      text    : bool / object / category columns. Up to `max_categories` most frequent levels
                (each seen at least `min_count` times) become one-hot columns; missing is its
                own level; unseen levels at inference become all zeros.
      dates   : columns named in `date_cols` are replaced by year, month, day-of-week.
    Columns that are entirely missing, or ID-like text (more than half of the rows distinct),
    are dropped and listed in `notes`.
    """

    def __init__(self, max_categories=20, min_count=5, date_cols=(), categorical_cols=()):
        self.max_categories, self.min_count = int(max_categories), int(min_count)
        self.date_cols = list(date_cols)
        self.force_cat = list(categorical_cols)  # numeric-looking columns that are really categories
        self.fitted = False

    # ---- helpers ----
    @staticmethod
    def _numeric(s):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    @staticmethod
    def _text(s):
        """Text form used for one-hot levels. Numbers are normalised (1, 1.0 and "1" all become "1")
        so a code column encodes the same way whether or not it contains missing values."""
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = s.astype("float64")
            return v.map(lambda x: MISSING if pd.isna(x)
                         else (str(int(x)) if float(x).is_integer() else repr(float(x)))).astype(str)
        return s.astype(object).where(s.notna(), MISSING).astype(str)

    def _expand_dates(self, df):
        out = df.copy()
        for c in self.date_cols:
            if c not in out.columns:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(out[c], errors="coerce")
            out[f"{c}__year"] = d.dt.year.astype("float64")
            out[f"{c}__month"] = d.dt.month.astype("float64")
            out[f"{c}__dow"] = d.dt.dayofweek.astype("float64")
            out = out.drop(columns=[c])
        return out

    # ---- fit ----
    def fit(self, df):
        for c in self.date_cols:
            if c not in df.columns:
                raise ValueError(f"date column {c!r} is not in the data")
        for c in self.force_cat:
            if c not in df.columns:
                raise ValueError(f"categorical column {c!r} is not in the data")
        d = self._expand_dates(df)
        n = len(d)
        self.notes, num, cat = [], [], []
        for c in d.columns:
            s = d[c]
            if s.isna().all():
                self.notes.append(f"dropped {c!r}: entirely missing in the training rows")
            elif (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                  and c not in self.force_cat):
                num.append(c)
            else:
                nun = int(s.nunique(dropna=True))
                if n >= 50 and nun > 0.5 * n:
                    self.notes.append(f"dropped {c!r}: looks like an ID or free text "
                                      f"({nun} distinct values in {n} rows)")
                else:
                    cat.append(c)

        self.num_cols, self.num_median, self.num_mean, self.num_scale, self.ind_cols = [], [], [], [], []
        for c in num:
            x = self._numeric(d[c])
            if x.isna().all():
                self.notes.append(f"dropped {c!r}: no parseable numbers in the training rows")
                continue
            med = float(x.median())
            xi = x.fillna(med)
            sd = float(xi.std(ddof=0))
            self.num_cols.append(c)
            self.num_median.append(med)
            self.num_mean.append(float(xi.mean()))
            self.num_scale.append(sd if sd > 0 else 1.0)
            if x.isna().any():
                self.ind_cols.append(c)

        self.cat_cols, self.cat_levels = [], {}
        for c in cat:
            vc = self._text(d[c]).value_counts()
            levels = [str(k) for k, cnt in vc.items() if cnt >= self.min_count][: self.max_categories]
            if len(levels) == 0 or (len(levels) == 1 and vc.iloc[0] == n):
                self.notes.append(f"dropped {c!r}: constant or no frequent levels")
                continue
            self.cat_cols.append(c)
            self.cat_levels[c] = levels

        # categorical columns that arrive as numbers (codes): the model signature must keep them numeric
        self.cat_numeric = [c for c in self.cat_cols
                            if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
        derived = {f"{c}__{s}" for c in self.date_cols for s in ("year", "month", "dow")}
        kept = set(self.num_cols) | set(self.cat_cols)
        raw = []
        for c in df.columns:
            if c in self.date_cols:
                if any(f"{c}__{s}" in kept for s in ("year", "month", "dow")):
                    raw.append(c)
            elif c in kept and c not in derived:
                raw.append(c)
        self.raw_inputs = raw
        if not (self.num_cols or self.cat_cols):
            raise ValueError("no usable feature columns remain after preprocessing")
        self.feature_names_out = (list(self.num_cols) + [f"{c}__missing" for c in self.ind_cols]
                                  + [f"{c}={lv}" for c in self.cat_cols for lv in self.cat_levels[c]])
        self.fitted = True
        return self

    # ---- transform ----
    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("preprocessor is not fitted")
        missing = [c for c in self.raw_inputs if c not in df.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        d = self._expand_dates(df[self.raw_inputs])
        n = len(d)
        num = np.empty((n, len(self.num_cols)), dtype=np.float32)
        ind = np.zeros((n, len(self.ind_cols)), dtype=np.float32)
        ind_pos = {c: i for i, c in enumerate(self.ind_cols)}
        for j, c in enumerate(self.num_cols):
            x = self._numeric(d[c])
            if c in ind_pos:
                ind[:, ind_pos[c]] = x.isna().to_numpy(dtype=np.float32)
            num[:, j] = (x.fillna(self.num_median[j]).to_numpy() - self.num_mean[j]) / self.num_scale[j]
        blocks = [num, ind]
        for c in self.cat_cols:
            v = self._text(d[c]).to_numpy()
            blocks.append(np.stack([(v == lv) for lv in self.cat_levels[c]], axis=1).astype(np.float32))
        return np.concatenate(blocks, axis=1)

    @property
    def n_numeric(self):
        return len(self.num_cols)

    # ---- (de)serialisation: python primitives only, so torch.load(weights_only=True) accepts it ----
    def to_dict(self):
        return {
            "max_categories": self.max_categories, "min_count": self.min_count,
            "date_cols": list(self.date_cols), "force_cat": list(self.force_cat),
            "cat_numeric": list(self.cat_numeric),
            "raw_inputs": list(self.raw_inputs),
            "num_cols": list(self.num_cols), "num_median": list(self.num_median),
            "num_mean": list(self.num_mean), "num_scale": list(self.num_scale),
            "ind_cols": list(self.ind_cols), "cat_cols": list(self.cat_cols),
            "cat_levels": {c: list(v) for c, v in self.cat_levels.items()},
            "feature_names_out": list(self.feature_names_out), "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d):
        p = cls(d["max_categories"], d["min_count"], d["date_cols"], d.get("force_cat", ()))
        p.raw_inputs, p.num_cols = list(d["raw_inputs"]), list(d["num_cols"])
        p.num_median, p.num_mean, p.num_scale = list(d["num_median"]), list(d["num_mean"]), list(d["num_scale"])
        p.ind_cols, p.cat_cols = list(d["ind_cols"]), list(d["cat_cols"])
        p.cat_numeric = list(d.get("cat_numeric", []))
        p.cat_levels = {c: list(v) for c, v in d["cat_levels"].items()}
        p.feature_names_out, p.notes = list(d["feature_names_out"]), list(d.get("notes", []))
        p.fitted = True
        return p

    def example_frame(self, df, n=5):
        """First n raw rows with stable dtypes (numeric -> float64, everything else -> str); used as
        the MLflow input example so the model signature does not depend on missing values."""
        ex = df[self.raw_inputs].head(n).copy()
        derived_src = set(self.date_cols)
        for c in ex.columns:
            if c in self.cat_numeric:  # a code column callers send as numbers: keep it numeric (double)
                lv = next((float(v) for v in self.cat_levels[c] if v != MISSING), 0.0)
                ex[c] = self._numeric(ex[c]).fillna(lv).astype("float64")
            elif c in derived_src or c in self.cat_cols:
                ex[c] = self._text(ex[c])
            else:
                j = self.num_cols.index(c)
                ex[c] = self._numeric(ex[c]).fillna(self.num_median[j]).astype("float64")
        return ex.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Reading, joining, task detection
# --------------------------------------------------------------------------- #
def _read(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _join(main, join_path, key):
    """Inner-join `join_path` onto `main` on `key`. The (possibly huge) join file is read in
    chunks and filtered to the keys present in `main`, so memory stays bounded."""
    if key not in main.columns:
        raise ValueError(f"--join-on {key!r} is not a column of the main CSV")
    if not os.path.isfile(join_path):
        raise FileNotFoundError(f"join CSV not found: {join_path}")
    keys = set(main[key].dropna().unique())
    parts = []
    for chunk in pd.read_csv(join_path, chunksize=200_000, low_memory=False):
        if key not in chunk.columns:
            raise ValueError(f"join key {key!r} is not a column of {join_path}")
        parts.append(chunk[chunk[key].isin(keys)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        right = pd.concat(parts, ignore_index=True)
    if right[key].duplicated().any():
        raise ValueError(f"{key!r} is not unique in the join CSV; a join would multiply rows")
    shared = sorted((set(right.columns) & set(main.columns)) - {key})
    right = right.drop(columns=shared)  # the main file's version of a shared column wins
    merged = main.merge(right, on=key, how="inner")
    if len(merged) == 0:
        raise ValueError(f"the join on {key!r} produced 0 rows (key values or dtypes do not match)")
    return merged, {"rows_main": len(main), "rows_after_join": len(merged), "shared_columns_kept_from_main": shared}


def _coerce_numeric_like(df, skip=()):
    """Object columns whose values are all numbers (e.g. read in chunks) become numeric."""
    for c in df.columns:
        if c in skip or df[c].dtype != object:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == df[c].notna().sum() and conv.notna().any():
            df[c] = conv
    return df


def detect_task(y):
    s = y.dropna()
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "classification"
    v = s.to_numpy(dtype="float64")
    if np.all(np.mod(v, 1) == 0) and s.nunique() <= 20:
        return "classification"
    return "regression"


def _label_str(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


# --------------------------------------------------------------------------- #
# Reading (where the rows come from) is separate from building (everything generic)
# --------------------------------------------------------------------------- #
def read_csv_tables(*, path, val_path=None, test_path=None, join_path=None, join_on=None):
    """Read CSV file(s) into DataFrames. Pure I/O: no splitting, no preprocessing.

    Returns a dict: main, test, val (DataFrame or None), lineage (paths + sha256 of every file read),
    notes (what was joined), ignore_extra (columns that must not become features, e.g. the join key).
    """
    if val_path and not test_path:
        raise ValueError("val_csv needs test_csv as well (or give neither and let the pipeline split)")
    if join_path and not join_on:
        raise ValueError("join_csv needs join_on KEY")
    notes, lineage = [], {"main_csv": {"path": os.path.abspath(path), "sha256": file_sha256(path)}}
    main = _read(path, "CSV")
    if join_path:
        main, jinfo = _join(main, join_path, join_on)
        lineage["join_csv"] = {"path": os.path.abspath(join_path), "sha256": file_sha256(join_path)}
        notes.append(f"joined {os.path.basename(join_path)} on {join_on!r}: {jinfo['rows_main']:,} main rows -> "
                     f"{jinfo['rows_after_join']:,} after the inner join"
                     + (f"; shared columns kept from the main file: {jinfo['shared_columns_kept_from_main']}"
                        if jinfo["shared_columns_kept_from_main"] else ""))
    test_df = val_df = None
    if test_path:
        test_df = _read(test_path, "test CSV")
        lineage["test_csv"] = {"path": os.path.abspath(test_path), "sha256": file_sha256(test_path)}
        if val_path:
            val_df = _read(val_path, "validation CSV")
            lineage["val_csv"] = {"path": os.path.abspath(val_path), "sha256": file_sha256(val_path)}
    return dict(main=main, test=test_df, val=val_df, lineage=lineage, notes=notes,
                ignore_extra=(join_on,) if join_on else ())


def build_dataset_from_tables(main, *, target, name, source_label, task="auto", test=None, val=None,
                              id_cols=(), drop_cols=(), date_cols=(), categorical_cols=(), ignore_extra=(),
                              val_fraction=0.2, test_fraction=0.2, max_categories=20, baseline_rows=20000,
                              seed=42, lineage=None, notes=None):
    """DataFrame(s) with a target column -> (splits, info) in the contract the pipeline expects.

    Everything here is independent of where the rows came from (a CSV, Parquet, a database, ...):
    rows with a missing target are dropped, the task is decided and the target encoded, the data are
    split (one table -> train/validation/test; `test` given -> that table is the untouched test set;
    `test` and `val` given -> used as they are), the preprocessor is fit on the TRAINING rows only,
    and a reference model (linear / logistic regression) is scored on the test rows.
    """
    from splits import carve_validation, split_three_way

    validate_name(name)
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"task must be auto, classification or regression (got {task!r})")
    notes, lineage = list(notes or []), dict(lineage or {})
    if val is not None and test is None:
        raise ValueError("a validation table needs a test table as well (or give neither)")
    if target not in main.columns:
        raise ValueError(f"target column {target!r} is not in the data; columns: {list(main.columns)[:15]}...")

    ignore = {target, *id_cols, *drop_cols, *ignore_extra}
    for c in ignore - {target}:
        if c not in main.columns:
            raise ValueError(f"column {c!r} (from id_cols / drop_cols / the join key) is not in the data")
    features = [c for c in main.columns if c not in ignore]
    if not features:
        raise ValueError("no feature columns left after removing target / id / drop columns")

    def prepare(df, what):
        missing = [c for c in [target] + features if c not in df.columns]
        if missing:
            raise ValueError(f"{what} is missing columns present in the main table: {missing[:10]}")
        df = _coerce_numeric_like(df[[target] + features].copy(), skip=(target,))
        n0 = len(df)
        df = df[df[target].notna()].reset_index(drop=True)
        if len(df) < n0:
            notes.append(f"{what}: dropped {n0 - len(df):,} rows with a missing target")
        return df

    main = prepare(main, "main table")
    if task == "auto":
        task = detect_task(main[target])
        notes.append(f"task auto-detected as {task} from the target column (override with --task)")

    test_df = prepare(test, "test table") if test is not None else None
    val_df = prepare(val, "validation table") if val is not None else None

    # ---- target encoding ----
    if task == "classification":
        ycol = main[target]
        if pd.api.types.is_numeric_dtype(ycol) and not pd.api.types.is_bool_dtype(ycol):
            classes = [_label_str(v) for v in sorted(ycol.unique())]
        else:
            classes = sorted({_label_str(v) for v in ycol.unique()})
        if len(classes) < 2:
            raise ValueError(f"classification needs at least 2 classes; the target has {classes}")
        if len(classes) > 50:
            raise ValueError(f"the target has {len(classes)} distinct values; that looks like a regression "
                             f"target. Pass --task regression if so.")
        cidx = {c: i for i, c in enumerate(classes)}

        def enc(df, what):
            labels = df[target].map(_label_str)
            bad = sorted(set(labels) - set(cidx))
            if bad:
                raise ValueError(f"{what} has target values not present in the main table: {bad[:5]}")
            return labels.map(cidx).to_numpy(dtype="int64")
    else:
        if not pd.api.types.is_numeric_dtype(main[target]) or pd.api.types.is_bool_dtype(main[target]):
            raise ValueError(f"regression needs a numeric target; {target!r} has dtype {main[target].dtype}")
        classes = None

        def enc(df, what):
            y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(y).all():
                raise ValueError(f"{what}: the target contains non-numeric or infinite values")
            return y

    y_main = enc(main, "main table")

    # ---- splits ----
    strat = task == "classification"
    try:
        if test_df is not None and val_df is not None:
            parts = {"train": main, "val": val_df, "test": test_df}
            ys = {"train": y_main, "val": enc(val_df, "validation table"), "test": enc(test_df, "test table")}
            how = (f"PRE-SPLIT by you: three tables used as given ({len(main):,} train / {len(val_df):,} "
                   f"validation / {len(test_df):,} test); nothing was re-split.")
            presplit = True
        elif test_df is not None:
            tr_i, va_i = carve_validation(len(main), y_main, val_size=val_fraction, seed=seed, stratify=strat)
            parts = {"train": main.iloc[tr_i], "val": main.iloc[va_i], "test": test_df}
            ys = {"train": y_main[tr_i], "val": y_main[va_i], "test": enc(test_df, "test table")}
            how = (f"PRE-SPLIT by you into train and test. The test table is kept untouched; "
                   f"{len(va_i):,} validation rows were carved from the main table ({'stratified, ' if strat else ''}"
                   f"seed {seed}); {len(tr_i):,} remain for training.")
            presplit = True
        else:
            tr_i, va_i, te_i = split_three_way(len(main), y_main, val_size=val_fraction,
                                               test_size=test_fraction, seed=seed, stratify=strat)
            parts = {"train": main.iloc[tr_i], "val": main.iloc[va_i], "test": main.iloc[te_i]}
            ys = {"train": y_main[tr_i], "val": y_main[va_i], "test": y_main[te_i]}
            how = (f"NOT pre-split: a single table of {len(main):,} rows was split {len(tr_i):,} train / "
                   f"{len(va_i):,} validation / {len(te_i):,} test ({'stratified by class, ' if strat else ''}"
                   f"random, seed {seed}).")
            presplit = False
    except ValueError as e:
        raise ValueError(f"could not split the data ({e}). With a classification target every class needs "
                         f"enough rows to appear in all three splits.") from e
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}

    # ---- preprocessing, fit on the training rows only ----
    prep = TabularPreprocessor(max_categories=max_categories, date_cols=date_cols,
                               categorical_cols=categorical_cols)
    prep.fit(parts["train"][features])
    notes.extend(prep.notes)
    X = {k: prep.transform(v[features]) for k, v in parts.items()}
    how += " Preprocessing (imputation, scaling, one-hot) was fit on the training rows only."

    if task == "regression":
        y_mean, y_std = float(ys["train"].mean()), float(ys["train"].std(ddof=0))
        y_std = y_std if y_std > 0 else 1.0
        tens = {k: TensorDataset(torch.from_numpy(X[k]),
                                 torch.from_numpy(((ys[k] - y_mean) / y_std).astype("float32")).view(-1, 1))
                for k in parts}
        out_dim = 1
    else:
        y_mean, y_std, out_dim = 0.0, 1.0, len(classes)
        tens = {k: TensorDataset(torch.from_numpy(X[k]), torch.from_numpy(np.array(ys[k], dtype="int64")))
                for k in parts}  # np.array copies: pandas 3 hands out read-only arrays

    baseline, base_pred = _baselines(task, X, ys, classes, baseline_rows, seed, notes)

    preproc = {"type": "tabular", "name": name, "target": target, "y_mean": y_mean, "y_std": y_std,
               **prep.to_dict()}
    info = dict(
        task=task, in_dim=X["train"].shape[1], out_dim=out_dim, y_mean=y_mean, y_std=y_std,
        baseline=baseline, class_names=classes, preproc=preproc,
        source=source_label, target_label=target, money_scale=None,
        split_info={"presplit": presplit, "strategy": how}, notes=notes, lineage=lineage,
        example_input=prep.example_frame(parts["test"][features], 5),
    )
    if base_pred is not None:
        info["baseline_test_pred"] = base_pred
    return tens, info


def load_csv_dataset(*, path, target, name, task="auto", id_cols=(), drop_cols=(), date_cols=(),
                     categorical_cols=(), val_path=None, test_path=None, join_path=None, join_on=None,
                     val_fraction=0.2, test_fraction=0.2, max_categories=20, baseline_rows=20000, seed=42):
    """CSV file(s) -> (splits, info). Convenience wrapper: read_csv_tables + build_dataset_from_tables."""
    t = read_csv_tables(path=path, val_path=val_path, test_path=test_path, join_path=join_path, join_on=join_on)
    return build_dataset_from_tables(
        t["main"], target=target, name=name, source_label=f"csv:{os.path.basename(path)}", task=task,
        test=t["test"], val=t["val"], id_cols=id_cols, drop_cols=drop_cols, date_cols=date_cols,
        categorical_cols=categorical_cols, ignore_extra=t["ignore_extra"], val_fraction=val_fraction,
        test_fraction=test_fraction, max_categories=max_categories, baseline_rows=baseline_rows, seed=seed,
        lineage=t["lineage"], notes=t["notes"])


def _baselines(task, X, ys, classes, baseline_rows, seed, notes):
    """Simple reference models fit on the training rows, scored on the test rows."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "regression":
                from sklearn.linear_model import LinearRegression
                lin = LinearRegression().fit(X["train"], ys["train"])
                pred = lin.predict(X["test"])
                mse = float(np.mean((pred - ys["test"]) ** 2))
                return ({"linear_regression_test_mse": mse, "linear_regression_test_rmse": mse ** 0.5,
                         "predict_mean_test_rmse": float(np.sqrt(np.mean((ys["train"].mean() - ys["test"]) ** 2)))},
                        pred)
            from metrics import macro_auc
            from sklearn.linear_model import LogisticRegression
            rng = np.random.default_rng(seed)
            n = len(ys["train"])
            idx = rng.choice(n, size=min(n, int(baseline_rows)), replace=False)
            if len(np.unique(ys["train"][idx])) < len(classes):
                idx = np.arange(n)
            lr = LogisticRegression(max_iter=200).fit(X["train"][idx], ys["train"][idx])
            probs = np.zeros((len(ys["test"]), len(classes)))
            probs[:, lr.classes_] = lr.predict_proba(X["test"])
            maj = int(np.bincount(ys["train"], minlength=len(classes)).argmax())
            return ({"logistic_regression_test_auc": float(macro_auc(probs, ys["test"])[0]),
                     "majority_class_test_accuracy": float((ys["test"] == maj).mean())}, None)
    except Exception as e:  # a reference model must never stop the pipeline
        notes.append(f"reference model skipped: {type(e).__name__}: {e}")
        return {}, None
