"""
Example data-source plug-in: read a table from a SQLite database.

    python run.py --data examples/sqlite_source.py:SqliteSource \\
                  --task regression --target price --out results/houses \\
                  --source-opt db=sales.db --source-opt table=houses

This is all it takes to plug a new place to read data from into the pipeline: subclass TableSource,
declare the extra settings in OPTIONS, and implement read_tables() to return DataFrames. Splitting,
the data audit, preprocessing (missing values, scaling, one-hot, dates), the reference model, tuning,
MLflow tracking, the registry and the packaged model are all done by the pipeline.

Swap sqlite3 for any other reader (Postgres, Snowflake, S3, a REST API...): only read_tables() changes.
"""
import re
import sqlite3

import pandas as pd

from datasources import TableData, TableSource, safe_name


class SqliteSource(TableSource):
    # extra settings, passed on the command line as --source-opt KEY=VALUE
    OPTIONS = {**TableSource.OPTIONS, "db": str, "table": str}

    def default_name(self):
        return safe_name(self.options.get("table") or "sqlite")

    def describe(self):
        return f"sqlite:{self.options.get('table')}"

    def read_tables(self):
        db, table = self.options.get("db"), self.options.get("table")
        if not db or not table:
            raise ValueError("SqliteSource needs --source-opt db=FILE and --source-opt table=NAME")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"table must be a plain identifier (letters, digits, underscore), got {table!r}")
        with sqlite3.connect(db) as con:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
        return TableData(main=df, label=self.describe(),
                         notes=[f"read {len(df):,} rows from table {table!r} of {db}"])
